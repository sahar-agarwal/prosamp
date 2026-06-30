"""Originator credibility scoring and deferred-loss detection (Eqs. 12, 27).

Rule-based, fixed-weight scoring (no fitted model) -- deliberately transparent so
the dashboard can explain *why* a deal scores the way it does. Inputs are taken
from the deal-level structural fields; features are normalized to roughly 0-1 so
the weights are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

# Eq. 12 weights: reward conservative structure, penalize fragile assumptions.
# Features are RISK-NORMALIZED (CE relative to expected loss / requirements), not
# absolute. This matters: subprime deals carry huge absolute CE precisely because
# their collateral is risky, so an absolute-CE score would rank them "most
# conservative" while they have the worst losses. Normalizing by expected loss
# measures conservatism *relative to the risk taken*, which is the real signal.
CREDIBILITY_WEIGHTS = {
    "ce_coverage": 0.5,            # available CE / expected loss (Eq. 3)
    "oc_cushion": 0.4,            # OC above target, relative
    "voluntary_ce_gap": 0.4,      # CE above rating-agency requirement, relative
    "reserve_coverage": 0.3,      # reserve / expected loss
    "excess_spread_reliance": -0.9,   # penalty (soft CE)
    "collateral_deterioration": -1.0,  # penalty (needs time-series)
}
# NOTE: trigger_strength is intentionally excluded from scoring. Subprime deals
# carry MORE protective trigger machinery (CNL/OC step-ups) precisely because they
# need it, so "more triggers" is not a clean credibility signal. The field is kept
# in deals.csv as documentation only. See [[project-abs-strategy-sip]].

# Eq. 27 deferred-loss flag weights.
DEFERRED_WEIGHTS = {
    "excess_spread_reliance_flag": 1.0,
    "small_reserve_flag": 0.8,
    "thin_voluntary_ce_flag": 1.2,
}


@dataclass
class CredibilityScore:
    deal_name: str
    score: float
    components: dict
    flags: dict
    deferred_loss_score: float


def _excess_spread_reliance(deal: pd.Series) -> float:
    """Share of total CE that leans on (soft) excess spread vs (hard) structure."""
    xs = deal["excess_spread_pct"]
    hard = deal["initial_oc_pct"] + deal["reserve_fund_pct"] + deal["total_subordination_pct"]
    total = xs + hard
    return float(xs / total) if total else 0.0


def score_deal(deal: pd.Series) -> CredibilityScore:
    """Compute Eq. 12 credibility score + Eq. 27 deferred-loss score for one deal."""
    eps = 1e-3
    expected_loss = max(deal["assumed_pd"] * deal["assumed_lgd"], eps)
    available_ce = deal["initial_oc_pct"] + deal["total_subordination_pct"]
    xs_reliance = _excess_spread_reliance(deal)

    ce_coverage = available_ce / expected_loss
    oc_cushion_rel = (deal["initial_oc_pct"] - deal["target_oc_pct"]) / max(
        deal["target_oc_pct"], eps)
    vol_ce_gap_rel = (available_ce - deal["required_ce_senior_pct"]) / max(
        deal["required_ce_senior_pct"], eps)
    reserve_coverage = deal["reserve_fund_pct"] / expected_loss

    comp = {
        "ce_coverage": ce_coverage,
        "oc_cushion": oc_cushion_rel,
        "voluntary_ce_gap": vol_ce_gap_rel,
        "reserve_coverage": reserve_coverage,
        "excess_spread_reliance": xs_reliance,
        "collateral_deterioration": 0.0,  # needs time-series; 0 in static demo
    }
    score = sum(CREDIBILITY_WEIGHTS[k] * v for k, v in comp.items())

    flags = {
        "excess_spread_reliance_flag": xs_reliance > 0.25,
        "small_reserve_flag": reserve_coverage < 0.50,
        "thin_voluntary_ce_flag": vol_ce_gap_rel < 0.05,
    }
    deferred = sum(DEFERRED_WEIGHTS[k] * (1.0 if v else 0.0) for k, v in flags.items())

    return CredibilityScore(
        deal_name=deal["deal_name"], score=round(score, 3),
        components={k: round(v, 4) for k, v in comp.items()},
        flags=flags, deferred_loss_score=round(deferred, 3),
    )


def score_all(deals: pd.DataFrame) -> pd.DataFrame:
    """Score every deal; returns a ranking table sorted by credibility."""
    rows = []
    for _, deal in deals.iterrows():
        cs = score_deal(deal)
        rows.append({
            "deal_name": cs.deal_name,
            "credibility_score": cs.score,
            "deferred_loss_score": cs.deferred_loss_score,
            "n_red_flags": sum(cs.flags.values()),
            **{f"flag_{k}": v for k, v in cs.flags.items()},
        })
    return pd.DataFrame(rows).sort_values("credibility_score", ascending=False)
