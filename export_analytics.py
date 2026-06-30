#!/usr/bin/env python3
"""
export_analytics.py - Export every metric the dashboard computes, per deal, to CSV.

Writes two files to output/:
  output/deal_analytics.csv     one row per deal (structure, pricing, credibility,
                                Monte Carlo, and realized backtest where available)
  output/tranche_analytics.csv  one row per tranche (CE, pricing, stress loss)

Realized-loss columns are filled only for deals present in
data/realized_performance.csv; others are left blank. Run:

    python export_analytics.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from engine import data, formulas, montecarlo, scoring, capital_stack, triggers, dynamics

N_SIMS = 50_000
SEED = 7
STRESS_LOSS_LEVELS = [0.15, 0.30]   # collateral-loss shocks for tranche stress


def realized_terminal() -> pd.Series:
    """Last cumulative net loss rate per deal, or empty if no realized file."""
    try:
        r = data.load_realized()
    except FileNotFoundError:
        return pd.Series(dtype=float)
    return (r.sort_values("period_end").groupby("deal_name")["cum_net_loss_rate"]
            .last())


def build_deal_rows(deals, tranches, realized, perf_full) -> list[dict]:
    rows = []
    for _, d in deals.iterrows():
        t = tranches[tranches.deal_name == d.deal_name]
        sr = t.loc[t.attachment_pct.idxmax()]
        jr = t.loc[t.attachment_pct.idxmin()]
        senior_ce = float(sr.attachment_pct)
        exp_loss = formulas.expected_loss(d.assumed_pd, d.assumed_lgd)
        first_loss_ce = d.initial_oc_pct + d.reserve_fund_pct
        cs = scoring.score_deal(d)

        ysac = jr.coupon_pct - sr.coupon_pct
        incr = senior_ce - jr.attachment_pct
        cost = formulas.cost_of_incremental_protection(ysac, incr)

        row = {
            "deal": d.deal_name, "grade": d.grade, "originator": d.originator,
            "closing_date": d.closing_date, "sector": d.sector,
            "pool_balance": d.original_pool_balance, "n_tranches": len(t),
            # structure / CE
            "senior_ce": round(senior_ce, 4),
            "total_subordination": d.total_subordination_pct,
            "initial_oc": d.initial_oc_pct, "target_oc": d.target_oc_pct,
            "reserve": d.reserve_fund_pct, "excess_spread": d.excess_spread_pct,
            # Lens 1: pricing
            "expected_loss": round(exp_loss, 4),
            "ce_coverage_x": round(senior_ce / exp_loss, 2) if exp_loss else None,
            "senior_coupon": round(float(sr.coupon_pct), 5),
            "junior_coupon": round(float(jr.coupon_pct), 5),
            "yield_sacrificed": round(ysac, 5),
            "cost_of_protection": round(cost, 3) if cost else None,
            # Lens 2: signal
            "credibility_score": cs.score,
            "deferred_loss_score": cs.deferred_loss_score,
            "n_red_flags": sum(cs.flags.values()),
            **{f"flag_{k}": v for k, v in cs.flags.items()},
        }

        # Forward-looking Monte Carlo across stress regimes. Asset correlation
        # follows PD (Basel retail) + regime uplift, not a flat assumption.
        for name, mult in montecarlo.SHOCK_REGIMES.items():
            mc = montecarlo.simulate(d.assumed_pd, d.assumed_lgd, senior_ce,
                                     correlation=montecarlo.rho_for_regime(d.assumed_pd, name),
                                     n_sims=N_SIMS, shock_multiplier=mult, seed=SEED)
            key = name.lower().split()[0]
            row[f"p_exhaust_{key}"] = round(mc.p_ce_exhaustion, 4)
            if name in ("Base case", "Global Financial Crisis"):
                row[f"tail99_{key}"] = round(mc.tail_loss_99, 4)

        # Triggers + dynamic CE (only meaningful where we have a realized series)
        perf = perf_full[perf_full.deal_name == d.deal_name]
        ev = triggers.evaluate(d, perf)
        row["trigger_terminal_limit"] = round(ev["terminal_limit"], 4)
        row["trigger_breached"] = ev["breached"]
        row["trigger_breach_month"] = ev["breach_month"]
        row["trigger_min_headroom"] = (round(ev["min_headroom"], 4)
                                       if ev["min_headroom"] is not None else None)
        cep = dynamics.ce_path(d, perf, senior_ce, breach_month=ev["breach_month"])
        if len(cep):
            row["senior_ce_current_pool_end"] = round(float(cep.structural_ce_pct.iloc[-1]), 4)
            row["available_cushion_end"] = round(float(cep.available_ce_pct.iloc[-1]), 4)

        # Realized backtest (only if we have realized data for this deal)
        rl = realized.get(d.deal_name)
        if rl is not None and pd.notna(rl):
            row["realized_cum_loss"] = round(float(rl), 4)
            row["backtest_error"] = round(formulas.backtest_error(rl, exp_loss), 4)
            row["first_loss_ce"] = round(first_loss_ce, 4)
            row["ce_surplus_firstloss"] = round(
                formulas.ce_surplus_shortfall(first_loss_ce, rl), 4)
            row["senior_ce_adequacy_ratio"] = formulas.ce_adequacy_ratio(senior_ce, rl)
            row["senior_ce_held"] = bool(senior_ce > rl)
        rows.append(row)
    return rows


def build_tranche_rows(deals, tranches) -> list[dict]:
    rows = []
    grade = deals.set_index("deal_name")["grade"]
    for _, t in tranches.iterrows():
        row = {
            "deal": t.deal_name, "grade": grade.get(t.deal_name, ""),
            "tranche": t.tranche, "rating": t.rating,
            "original_balance": t.original_balance, "coupon": t.coupon_pct,
            "attachment": t.attachment_pct, "detachment": t.detachment_pct,
            "initial_ce": t.initial_ce_pct, "required_ce": t.required_ce_pct,
            "rating_cushion": round(formulas.rating_cushion(
                t.initial_ce_pct, t.required_ce_pct), 4),
        }
        for L in STRESS_LOSS_LEVELS:
            row[f"loss_pct_at_{int(L*100)}"] = round(
                capital_stack.tranche_loss(L, t.attachment_pct, t.detachment_pct), 4)
        rows.append(row)
    return rows


def main() -> None:
    deals = data.load_deals()
    tranches = data.load_tranches()
    realized = realized_terminal()
    try:
        perf_full = data.load_realized()
    except FileNotFoundError:
        perf_full = pd.DataFrame(columns=["deal_name"])

    out = Path("output")
    out.mkdir(exist_ok=True)

    deal_df = pd.DataFrame(build_deal_rows(deals, tranches, realized, perf_full))
    deal_df = deal_df.sort_values(["grade", "credibility_score"],
                                  ascending=[True, False])
    deal_df.to_csv(out / "deal_analytics.csv", index=False)

    tr_df = pd.DataFrame(build_tranche_rows(deals, tranches))
    tr_df.to_csv(out / "tranche_analytics.csv", index=False)

    n_real = deal_df["realized_cum_loss"].notna().sum() if "realized_cum_loss" in deal_df else 0
    print(f"Wrote output/deal_analytics.csv ({len(deal_df)} deals, "
          f"{n_real} with realized backtest)")
    print(f"Wrote output/tranche_analytics.csv ({len(tr_df)} tranches)")


if __name__ == "__main__":
    main()
