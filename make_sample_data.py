#!/usr/bin/env python3
"""
make_sample_data.py - Generate a realistic synthetic dataset for the CE dashboard.

Writes three CSVs to data/ that match the exact schema the real pipeline uses, so
you can build and demo the dashboard before your real data lands:

  data/deals.csv                deal-level structure + at-issuance CE features
  data/tranches.csv             tranche-level capital structure
  data/realized_performance.csv SAME columns as absee.py output (drop-in swap)

SCOPE: subprime auto-loan ABS only. Prime shelves were dropped so the whole
dashboard speaks one collateral language -- the cross-deal comparisons (credibility
vs realized loss, trigger tightness, CE adequacy) are only meaningful within a
single collateral grade. The numbers are plausible subprime auto-ABS values but
invented. Replace realized_performance.csv with absee.py's output and hand-curate
deals/tranches from prospectuses to go live. Deal names mirror deals.json so the
swap lines up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA = Path("data")

# (deal_name, originator, closing, term_months, terminal_loss_rate,
#  assumed_pd, assumed_lgd, target_oc, total_sub, trigger_strength 0-1, quality 0-1)
# All deals are subprime auto loan. terminal_loss_rate is the REALIZED lifetime CNL;
# assumed_pd x assumed_lgd is the loss PRICED at issuance. Strong shelves come in
# at/under priced (CE holds, triggers clear); weak 2022 deep-subprime vintages run
# hot (triggers breach, OC steps up).
#
# `quality` (0-1) is the deal's latent underwriting strength. It is the SINGLE knob
# behind the dashboard's central thesis -- structure reveals confidence, confidence
# predicts outcomes -- so it drives BOTH the realized loss (already baked into
# terminal_loss_rate / trigger_strength above) AND the at-issuance structural
# choices below: stronger deals voluntarily over-collateralize, hold bigger
# reserves, lean less on (soft) excess spread, and set OC nearer target. That makes
# credibility (Tab 2) correlate negatively with realized loss, as it should.
#  name, originator, closing, term, terminal, pd, lgd, target_oc, total_sub, trig, quality
DEALS = [
    ("AmeriCredit Auto Receivables 2021-3 (subprime)",      "AmeriCredit",                 "2021-08-12", 60, 0.072, 0.155, 0.550, 0.080, 0.250, 0.58, 0.92),
    ("AmeriCredit Auto Receivables 2022-2 (subprime)",      "AmeriCredit",                 "2022-06-18", 60, 0.088, 0.165, 0.575, 0.080, 0.250, 0.56, 0.82),
    ("AmeriCredit Auto Receivables 2023-1 (subprime)",      "AmeriCredit",                 "2023-02-15", 60, 0.080, 0.160, 0.560, 0.082, 0.255, 0.57, 0.88),
    ("Santander Drive Auto Receivables 2022-2 (subprime)",  "Santander Consumer",          "2022-04-25", 60, 0.140, 0.200, 0.550, 0.090, 0.283, 0.46, 0.48),
    ("Santander Drive Auto Receivables 2023-2 (subprime)",  "Santander Consumer",          "2023-05-17", 60, 0.130, 0.197, 0.550, 0.090, 0.285, 0.47, 0.55),
    ("Exeter Automobile Receivables 2022-2 (subprime)",     "Exeter Finance",              "2022-04-27", 60, 0.175, 0.210, 0.620, 0.100, 0.420, 0.42, 0.28),
    ("Exeter Automobile Receivables 2023-1 (subprime)",     "Exeter Finance",              "2023-01-22", 60, 0.165, 0.205, 0.610, 0.100, 0.410, 0.43, 0.34),
    ("DriveTime Auto Owner Trust 2022-2 (subprime)",        "DriveTime",                   "2022-08-10", 66, 0.185, 0.220, 0.636, 0.110, 0.430, 0.40, 0.22),
    ("Westlake Automobile Receivables 2022-3 (subprime)",   "Westlake Financial",          "2022-09-14", 60, 0.092, 0.180, 0.555, 0.085, 0.300, 0.50, 0.80),
    ("GLS Auto Receivables Trust 2022-4 (subprime)",        "Global Lending Services",     "2022-11-09", 60, 0.165, 0.205, 0.610, 0.105, 0.400, 0.43, 0.35),
    ("Flagship Credit Auto Trust 2022-3 (subprime)",        "Flagship Credit",             "2022-07-20", 66, 0.100, 0.190, 0.568, 0.090, 0.320, 0.49, 0.74),
    ("CPS Auto Receivables Trust 2022-C (subprime)",        "Consumer Portfolio Services", "2022-06-15", 66, 0.158, 0.200, 0.600, 0.095, 0.360, 0.45, 0.40),
    ("American Credit Acceptance 2022-3 (subprime)",        "American Credit Acceptance",  "2022-09-21", 60, 0.200, 0.240, 0.625, 0.115, 0.450, 0.39, 0.15),
]


def structural_features(target_oc: float, total_sub: float, expected_loss: float,
                        quality: float) -> dict:
    """Derive at-issuance CE features from a deal's latent quality (0-1).

    Stronger deals (higher quality) set OC nearer/above target, hold larger
    reserves, lean less on excess spread, and leave a wider voluntary CE gap over
    the rating-agency requirement.
    """
    initial_oc = round(target_oc * (0.85 + 0.28 * quality), 4)
    reserve = round(0.008 + 0.022 * quality, 4)
    excess_spread = round(0.040 + 0.090 * (1.0 - quality), 4)
    available_ce = initial_oc + total_sub
    required_ce = round(available_ce * (0.78 - 0.16 * quality), 4)
    return {
        "initial_oc_pct": initial_oc,
        "reserve_fund_pct": reserve,
        "excess_spread_pct": excess_spread,
        "required_ce_senior_pct": required_ce,
    }

# Tranche template: (suffix, rating, balance_share, coupon, attach, detach)
# attach/detach are fractions of the pool from the BOTTOM of the capital stack.
# One subprime template keeps the synthetic structures comparable; the deal-level
# CE features (OC, reserve, required CE, trigger strength) carry the variation.
SUBPRIME_TRANCHES = [
    ("E", "BB",  0.060, 0.105, 0.080, 0.140),
    ("D", "BBB", 0.080, 0.078, 0.140, 0.220),
    ("C", "A",   0.090, 0.060, 0.220, 0.310),
    ("B", "AA",  0.110, 0.046, 0.310, 0.420),
    ("A", "AAA", 0.660, 0.034, 0.420, 1.000),
]


def logistic_curve(terminal: float, months: int) -> np.ndarray:
    """S-shaped cumulative loss curve ramping to `terminal` over `months`."""
    t = np.arange(1, months + 1)
    midpoint = months * 0.45
    steepness = 0.18
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    s = (s - s.min()) / (s.max() - s.min())
    noise = 1.0 + RNG.normal(0, 0.03, months).cumsum() * 0.01
    return terminal * s * noise


def build() -> None:
    DATA.mkdir(exist_ok=True)
    deal_rows, tranche_rows, perf_rows = [], [], []

    for (name, orig, closing, term, terminal, pd_, lgd, toc, sub, trig,
         quality) in DEALS:
        pool = float(RNG.integers(700, 2200)) * 1e6  # $0.7B-$2.2B
        feats = structural_features(toc, sub, pd_ * lgd, quality)
        deal_rows.append({
            "deal_name": name, "sector": "auto_loan", "originator": orig,
            "grade": "subprime", "closing_date": closing,
            "original_pool_balance": round(pool, 2),
            "initial_oc_pct": feats["initial_oc_pct"], "target_oc_pct": toc,
            "reserve_fund_pct": feats["reserve_fund_pct"],
            "excess_spread_pct": feats["excess_spread_pct"],
            "total_subordination_pct": sub,
            "trigger_strength": trig,
            "required_ce_senior_pct": feats["required_ce_senior_pct"],
            "assumed_pd": pd_, "assumed_lgd": lgd,
        })

        for suffix, rating, share, coupon, attach, detach in SUBPRIME_TRANCHES:
            tranche_rows.append({
                "deal_name": name, "tranche": suffix, "rating": rating,
                "original_balance": round(pool * share, 2),
                "coupon_pct": coupon,
                "attachment_pct": attach, "detachment_pct": detach,
                "initial_ce_pct": attach,         # CE = everything below it
                "required_ce_pct": round(attach * RNG.uniform(0.75, 0.92), 4),
            })

        # Monthly realized performance (matches absee.py columns exactly)
        curve = logistic_curve(terminal, term)
        start = pd.Timestamp(closing) + pd.offsets.MonthEnd(1)
        prev_cum = 0.0
        for m in range(term):
            period_end = (start + pd.offsets.MonthEnd(m)).strftime("%Y-%m-%d")
            cum_rate = float(curve[m])
            cum_loss = cum_rate * pool
            net = cum_loss - prev_cum
            prev_cum = cum_loss
            end_bal = pool * max(0.0, 1.0 - (m + 1) / term) ** 1.3
            perf_rows.append({
                "deal_name": name, "cik": 0, "accession": f"sample-{m:03d}",
                "period_end": period_end,
                "n_assets": int(40000 * max(0.05, 1 - m / term)),
                "period_chargeoff": round(net * 1.25, 2),       # gross
                "period_recovery": round(net * 0.25, 2),        # ~20% recovery
                "period_net_loss": round(net, 2),
                "period_end_balance": round(end_bal, 2),
                "cum_net_loss": round(cum_loss, 2),
                "original_pool_balance": round(pool, 2),
                "cum_net_loss_rate": round(cum_rate, 6),
            })

    pd.DataFrame(deal_rows).to_csv(DATA / "deals.csv", index=False)
    pd.DataFrame(tranche_rows).to_csv(DATA / "tranches.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(DATA / "realized_performance.csv", index=False)
    print(f"Wrote {len(deal_rows)} subprime deals, {len(tranche_rows)} tranches, "
          f"{len(perf_rows)} performance rows to {DATA}/")


if __name__ == "__main__":
    build()
