#!/usr/bin/env python3
"""
absee.py - Pull SEC EDGAR ABS-EE asset-level data and build realized loss curves.

Designed for the ABS Strategy SIP: turn loan-level Reg AB II filings into a tidy
`realized_performance` table you can drop straight into the dashboard.

Subcommands
-----------
  find      Look up candidate trust CIKs by name (ABS-EE filers only).
  inspect   Dump the XML tag names + one sample asset from a filing, so you can
            confirm the charge-off / recovery field names before trusting sums.
  build     Fetch every ABS-EE filing for the configured deals, aggregate
            loan-level charge-offs and recoveries, and write cumulative net
            loss curves to output/realized_performance.csv.
  selftest  Parse a synthetic filing offline and verify the loss math (no network).

Scaling to a whole shelf is just adding {"name", "cik"} entries to deals.json;
no code changes.

SEC fair-access rules: you MUST send a descriptive User-Agent containing a real
contact (name + email). Set EDGAR_USER_AGENT or pass --user-agent. Requests are
throttled below SEC's 10 req/sec limit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

# --- Field names for the AUTO LOAN ABS-EE schema (Reg AB II, Schedule AL). ---
# These are the defaults; confirm with `inspect` and override via CLI if a deal
# uses different tags (e.g. auto LEASE deals differ from auto LOAN deals).
DEFAULT_COLS = {
    "chargeoff": "chargedoffPrincipalAmount",
    "recovery": "recoveredAmount",
    "end_balance": "reportingPeriodActualEndBalanceAmount",
    "original": "originalLoanAmount",
    "period_end": "reportingPeriodEndingDate",
}

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_DIR_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
BROWSE_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

CACHE_DIR = Path("output/cache")


# --------------------------------------------------------------------------- #
# HTTP client                                                                  #
# --------------------------------------------------------------------------- #
class EdgarClient:
    """Polite EDGAR HTTP client: required User-Agent, throttling, retries."""

    def __init__(self, user_agent: str, min_interval: float = 0.2,
                 timeout: int = 30, max_retries: int = 4):
        if not user_agent or "@" not in user_agent:
            raise SystemExit(
                "EDGAR requires a descriptive User-Agent with a contact email.\n"
                "  Set it once:   export EDGAR_USER_AGENT='Your Name your.email@example.com'\n"
                "  Or pass:       --user-agent 'Your Name your.email@example.com'"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, stream: bool = False
            ) -> requests.Response:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, stream=stream,
                                        timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                last_err = RuntimeError(f"HTTP {resp.status_code} for {url}")
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Failed to GET {url}: {last_err}")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return self.get(url, params=params).json()

    def download(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = self.get(url, stream=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
        return dest


# --------------------------------------------------------------------------- #
# EDGAR discovery                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Filing:
    accession: str
    filing_date: str
    report_date: str
    primary_doc: str


def normalize_cik(cik: str | int) -> int:
    return int(str(cik).strip().lstrip("0") or "0")


def find_entities(client: EdgarClient, name: str) -> list[tuple[str, str]]:
    """Best-effort: return (cik, title) pairs for ABS-EE filers matching `name`."""
    resp = client.get(BROWSE_EDGAR_URL, params={
        "action": "getcompany", "company": name, "type": "ABS-EE",
        "dateb": "", "owner": "include", "count": "100", "output": "atom",
    })
    text = resp.text
    pairs: dict[str, str] = {}
    # Atom entries: a title near each CIK=######### link.
    for m in re.finditer(r"CIK=(\d{10})", text):
        cik = m.group(1)
        window = text[max(0, m.start() - 400): m.start()]
        tmatch = re.findall(r"<title>(.*?)</title>", window, re.S)
        title = (tmatch[-1].strip() if tmatch else "").replace("\n", " ")
        pairs.setdefault(cik, title)
    return sorted(pairs.items())


def _filing_arrays(blob: dict) -> dict:
    """submissions.json nests arrays under filings.recent; paged files don't."""
    if "filings" in blob and "recent" in blob["filings"]:
        return blob["filings"]["recent"]
    return blob


def list_absee_filings(client: EdgarClient, cik: int) -> list[Filing]:
    blob = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    sources = [_filing_arrays(blob)]
    for extra in blob.get("filings", {}).get("files", []):
        sources.append(client.get_json(
            SUBMISSIONS_FILE_URL.format(name=extra["name"])))

    filings: list[Filing] = []
    for arr in sources:
        forms = arr.get("form", [])
        for i, form in enumerate(forms):
            if form != "ABS-EE":
                continue
            filings.append(Filing(
                accession=arr["accessionNumber"][i],
                filing_date=arr.get("filingDate", [""] * len(forms))[i],
                report_date=arr.get("reportDate", [""] * len(forms))[i],
                primary_doc=arr.get("primaryDocument", [""] * len(forms))[i],
            ))
    filings.sort(key=lambda f: f.report_date or f.filing_date)
    return filings


_JUNK_XML = ("primary_doc.xml", "metalinks.json")
_JUNK_SUFFIX = (".xsd", "_def.xml", "_lab.xml", "_pre.xml", "_cal.xml")


def asset_xml_url(client: EdgarClient, cik: int, accession: str) -> str:
    """Pick the EX-102 asset-data XML: the largest .xml that isn't boilerplate."""
    acc = accession.replace("-", "")
    base = ARCHIVE_DIR_URL.format(cik=cik, acc=acc)
    index = client.get_json(base + "index.json")
    best_name, best_size = None, -1
    for item in index.get("directory", {}).get("item", []):
        name = item.get("name", "")
        low = name.lower()
        if not low.endswith(".xml"):
            continue
        if low in _JUNK_XML or low.endswith(_JUNK_SUFFIX):
            continue
        if re.match(r"r\d+\.xml$", low):  # XBRL viewer fragments
            continue
        size = int(item.get("size", 0) or 0)
        if size > best_size:
            best_name, best_size = name, size
    if not best_name:
        raise RuntimeError(f"No asset XML found in {base}")
    return base + best_name


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_assets(source):
    """Stream per-loan records from an ABS-EE XML file, yielding flat dicts.

    Each direct child of the root element is treated as one asset record. The
    SEC auto schemas name this element <assets> (plural), but detecting it by
    position rather than name works across asset classes and avoids guessing.
    Namespaces are stripped and each leaf descendant becomes one key. Memory
    stays flat because each record is cleared after it is yielded.
    """
    depth = 0
    for event, elem in ET.iterparse(source, events=("start", "end")):
        if event == "start":
            depth += 1
            continue
        depth -= 1
        if depth != 1:  # only fire when a direct child of the root closes
            continue
        row: dict[str, str] = {}
        for leaf in elem.iter():
            if leaf is elem:
                continue
            text = (leaf.text or "").strip()
            if text:
                row[_localname(leaf.tag)] = text
        if row:
            yield row
        elem.clear()


def to_num(value: str | None) -> float:
    """Parse a money/number string. Handles commas and (parenthesized) negatives."""
    if not value:
        return 0.0
    s = value.strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        n = float(s)
    except ValueError:
        return 0.0
    return -n if neg else n


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class PeriodAgg:
    deal_name: str
    cik: int
    accession: str
    period_end: str
    n_assets: int = 0
    chargeoff: float = 0.0
    recovery: float = 0.0
    end_balance: float = 0.0
    original: float = 0.0
    seen_cols: set = field(default_factory=set)


def aggregate_period(rows, cols: dict, meta: dict) -> PeriodAgg:
    """Sum charge-offs / recoveries / balances across loan-level rows."""
    agg = PeriodAgg(meta["deal_name"], meta["cik"], meta["accession"],
                    meta.get("period_end", ""))
    for row in rows:
        agg.n_assets += 1
        agg.seen_cols.update(row.keys())
        agg.chargeoff += to_num(row.get(cols["chargeoff"]))
        agg.recovery += to_num(row.get(cols["recovery"]))
        agg.end_balance += to_num(row.get(cols["end_balance"]))
        agg.original += to_num(row.get(cols["original"]))
        if not agg.period_end:
            agg.period_end = row.get(cols["period_end"], "")
    return agg


def build_curve(periods: list[PeriodAgg]) -> list[dict]:
    """Add cumulative net loss + loss rate to chronologically ordered periods."""
    periods = sorted(periods, key=lambda p: p.period_end)
    original_pool = next((p.original for p in periods if p.original > 0), 0.0)
    cum = 0.0
    out: list[dict] = []
    for p in periods:
        net = p.chargeoff - p.recovery
        cum += net
        out.append({
            "deal_name": p.deal_name,
            "cik": p.cik,
            "accession": p.accession,
            "period_end": p.period_end,
            "n_assets": p.n_assets,
            "period_chargeoff": round(p.chargeoff, 2),
            "period_recovery": round(p.recovery, 2),
            "period_net_loss": round(net, 2),
            "period_end_balance": round(p.end_balance, 2),
            "cum_net_loss": round(cum, 2),
            "original_pool_balance": round(original_pool, 2),
            "cum_net_loss_rate": round(cum / original_pool, 6) if original_pool else None,
        })
    return out


# --------------------------------------------------------------------------- #
# Subcommands                                                                  #
# --------------------------------------------------------------------------- #
def cmd_find(args, client: EdgarClient) -> None:
    matches = find_entities(client, args.name)
    if not matches:
        print(f"No ABS-EE filers matched {args.name!r}.")
        return
    print(f"{'CIK':<12} Entity")
    print("-" * 60)
    for cik, title in matches:
        print(f"{cik:<12} {title}")
    print(f"\n{len(matches)} match(es). Add the relevant CIKs to deals.json.")


def _cached_xml(client: EdgarClient, cik: int, accession: str,
                use_cache: bool) -> Path:
    dest = CACHE_DIR / f"{cik}_{accession.replace('-', '')}.xml"
    if use_cache and dest.exists() and dest.stat().st_size > 0:
        return dest
    url = asset_xml_url(client, cik, accession)
    print(f"  downloading {url}")
    return client.download(url, dest)


def cmd_inspect(args, client: EdgarClient) -> None:
    if args.file:
        path = Path(args.file)
    else:
        cik = normalize_cik(args.cik)
        filings = list_absee_filings(client, cik)
        if not filings:
            raise SystemExit(f"No ABS-EE filings for CIK {cik}.")
        target = next((f for f in filings if f.accession == args.accession),
                      filings[-1]) if args.accession else filings[-1]
        print(f"Inspecting {target.accession} (report {target.report_date})")
        path = _cached_xml(client, cik, target.accession, not args.no_cache)

    tags: set[str] = set()
    sample: dict | None = None
    for i, row in enumerate(parse_assets(str(path))):
        tags.update(row.keys())
        if sample is None:
            sample = row
        if i + 1 >= args.limit:
            break

    print(f"\n{len(tags)} distinct asset fields:\n")
    for t in sorted(tags):
        marks = [k for k, v in DEFAULT_COLS.items() if v == t]
        flag = f"   <-- used as {', '.join(marks)}" if marks else ""
        print(f"  {t}{flag}")

    missing = [f"{k}={v}" for k, v in DEFAULT_COLS.items()
               if v not in tags and k != "period_end"]
    if missing:
        print("\nWARNING: expected field(s) not present in this filing:")
        for m in missing:
            print(f"  {m}")
        print("Override with --col-* flags on `build` (e.g. auto-lease deals).")

    if sample:
        print("\nSample asset:")
        for k in sorted(sample):
            print(f"  {k}: {sample[k]}")


def load_config(path: str) -> list[dict]:
    cfg = json.loads(Path(path).read_text())
    deals = cfg.get("deals", [])
    if not deals:
        raise SystemExit(f"No deals in {path}. Add entries to the 'deals' list.")
    return deals


def cmd_build(args, client: EdgarClient) -> None:
    import pandas as pd

    cols = {
        "chargeoff": args.col_chargeoff,
        "recovery": args.col_recovery,
        "end_balance": args.col_end_balance,
        "original": args.col_original,
        "period_end": DEFAULT_COLS["period_end"],
    }
    deals = load_config(args.config)
    all_rows: list[dict] = []

    for deal in deals:
        cik = normalize_cik(deal["cik"])
        name = deal.get("name", str(cik))
        print(f"\n=== {name} (CIK {cik}) ===")
        if cik == 0:
            print("  SKIP: placeholder CIK — put a real CIK in deals.json")
            continue
        try:
            filings = list_absee_filings(client, cik)
        except Exception as exc:
            print(f"  SKIP: could not list filings ({exc})")
            continue
        if args.limit_filings:
            filings = filings[-args.limit_filings:]
        print(f"  {len(filings)} ABS-EE filing(s)")
        if not filings:
            continue

        periods: list[PeriodAgg] = []
        for f in filings:
            try:
                path = _cached_xml(client, cik, f.accession, not args.no_cache)
            except Exception as exc:  # one bad filing shouldn't kill the run
                print(f"  SKIP {f.accession}: {exc}")
                continue
            meta = {"deal_name": name, "cik": cik, "accession": f.accession,
                    "period_end": f.report_date}
            agg = aggregate_period(parse_assets(str(path)), cols, meta)
            periods.append(agg)
            print(f"  {agg.period_end}  assets={agg.n_assets:>7}  "
                  f"net_loss={agg.chargeoff - agg.recovery:>14,.0f}")
            if periods and not (set(cols.values()) & agg.seen_cols):
                print("  WARNING: none of the configured columns were found; "
                      "run `inspect` to confirm field names.")

        all_rows.extend(build_curve(periods))

    if not all_rows:
        raise SystemExit("No data produced. Check CIKs and field names.")

    df = pd.DataFrame(all_rows).sort_values(["deal_name", "period_end"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows for {df['deal_name'].nunique()} deal(s) -> {out}")


SYNTHETIC = """<?xml version="1.0" encoding="UTF-8"?>
<assetData xmlns="http://www.sec.gov/edgar/document/absee/autoloan/assetdata">
  <asset>
    <assetNumber>A1</assetNumber>
    <reportingPeriodEndingDate>2023-01-31</reportingPeriodEndingDate>
    <originalLoanAmount>20000.00</originalLoanAmount>
    <reportingPeriodActualEndBalanceAmount>18000.00</reportingPeriodActualEndBalanceAmount>
    <chargedoffPrincipalAmount>0.00</chargedoffPrincipalAmount>
    <recoveredAmount>0.00</recoveredAmount>
  </asset>
  <asset>
    <assetNumber>A2</assetNumber>
    <reportingPeriodEndingDate>2023-01-31</reportingPeriodEndingDate>
    <originalLoanAmount>30000.00</originalLoanAmount>
    <reportingPeriodActualEndBalanceAmount>0.00</reportingPeriodActualEndBalanceAmount>
    <chargedoffPrincipalAmount>5000.00</chargedoffPrincipalAmount>
    <recoveredAmount>1500.00</recoveredAmount>
  </asset>
</assetData>
"""


def cmd_selftest(args, client) -> None:
    """Offline check of parsing + loss math against a known synthetic filing."""
    fixture = Path("output/sample_absee.xml")
    fixture.write_text(SYNTHETIC)
    rows = list(parse_assets(str(fixture)))
    assert len(rows) == 2, f"expected 2 assets, got {len(rows)}"

    meta = {"deal_name": "TEST 2023-1", "cik": 9999999, "accession": "x"}
    agg = aggregate_period(iter(rows), DEFAULT_COLS, meta)
    assert agg.n_assets == 2
    assert agg.chargeoff == 5000.0, agg.chargeoff
    assert agg.recovery == 1500.0, agg.recovery
    assert agg.original == 50000.0, agg.original

    curve = build_curve([agg])
    r = curve[0]
    assert r["period_net_loss"] == 3500.0, r
    assert r["cum_net_loss"] == 3500.0, r
    assert r["original_pool_balance"] == 50000.0, r
    assert abs(r["cum_net_loss_rate"] - 0.07) < 1e-9, r
    assert to_num("(1,234.50)") == -1234.50
    assert to_num("$2,000") == 2000.0
    print("selftest OK: parsing, charge-off/recovery sums, cumulative net loss "
          "rate (3,500 / 50,000 = 7.0%) all correct.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pull SEC ABS-EE asset-level data into realized loss curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--user-agent", default=os.environ.get("EDGAR_USER_AGENT", ""),
                   help="SEC-required 'Name email' (or set EDGAR_USER_AGENT).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("find", help="Find trust CIKs by name.")
    pf.add_argument("name")
    pf.set_defaults(func=cmd_find, net=True)

    pi = sub.add_parser("inspect", help="Show asset field names + a sample.")
    pi.add_argument("--cik", help="Trust CIK (latest ABS-EE used if no accession).")
    pi.add_argument("--accession", help="Specific accession number.")
    pi.add_argument("--file", help="Parse a local XML instead of fetching.")
    pi.add_argument("--limit", type=int, default=2000)
    pi.add_argument("--no-cache", action="store_true")
    pi.set_defaults(func=cmd_inspect, net=True)

    pb = sub.add_parser("build", help="Build realized loss curves from deals.json.")
    pb.add_argument("--config", default="deals.json")
    pb.add_argument("--out", default="output/realized_performance.csv")
    pb.add_argument("--limit-filings", type=int, default=0,
                    help="Only the most recent N filings per deal (0 = all).")
    pb.add_argument("--no-cache", action="store_true")
    pb.add_argument("--col-chargeoff", default=DEFAULT_COLS["chargeoff"])
    pb.add_argument("--col-recovery", default=DEFAULT_COLS["recovery"])
    pb.add_argument("--col-end-balance", default=DEFAULT_COLS["end_balance"])
    pb.add_argument("--col-original", default=DEFAULT_COLS["original"])
    pb.set_defaults(func=cmd_build, net=True)

    ps = sub.add_parser("selftest", help="Offline check of the loss math.")
    ps.set_defaults(func=cmd_selftest, net=False)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # inspect with --file needs no network either
    needs_net = getattr(args, "net", True) and not (
        args.cmd == "inspect" and getattr(args, "file", None))
    client = EdgarClient(args.user_agent) if needs_net else None
    args.func(args, client)


if __name__ == "__main__":
    main()
