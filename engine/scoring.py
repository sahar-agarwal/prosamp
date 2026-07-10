"""Originator credibility scoring and deferred-loss detection (Eqs. 12, 27).

Rule-based, fixed-weight scoring (no fitted model) -- deliberately transparent.

DESIGN (de-confounded): credibility measures *voluntary structural conservatism*
using only fields independent of our own loss assumption. It normalizes credit
enhancement by the RATING-AGENCY required CE (an external, risk-adjusted benchmark),
NOT by our assumed PD x LGD. An earlier version divided available CE by expected
loss (PD x LGD); because PD/LGD were assigned by credit grade, and realized loss
also tracks grade, that made the credibility-vs-realized-loss relationship partly
circular. Using the external required-CE benchmark removes that confound: the score
reflects how much protection a sponsor provided ABOVE what the ratings required -- a
genuine structural choice, not a restatement of the loss we penciled in.

Consequence to keep honest: credibility is a measure of *conservatism*, NOT a proxy
for collateral quality. It relates to loss WITHIN a grade; loss ACROSS grades is
driven by the collateral itself. The Tab 2 scatter is labeled accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

# Weights over features INDEPENDENT of our PD/LGD assumption (see module docstring).
# Dominant signal: voluntary CE above the rating-agency required level. Reserve is a
# hard-cash cushion; excess-spread reliance is penalized as "soft" CE.
# Deliberately EXCLUDED:
#   * ce_coverage (available CE / expected loss) and reserve_coverage -- expected loss
#     is our assumed PD x LGD, so they would confound the credibility-vs-loss test;
#   * oc_cushion (initial - target OC) -- sign is ambiguous, since conservative deals
#     often BUILD OC from a low initial toward a high target;
#   * trigger_strength -- subprime carry more triggers because they need them, so
#     "more triggers" is not a clean credibility signal.
CREDIBILITY_WEIGHTS = {
    "voluntary_ce_gap": 1.5,      # CE above rating-agency required CE
    "reserve_ratio": 1.0,         # hard-cash reserve cushion
    "excess_spread_reliance": -1.0,   # penalty (soft CE)
    "collateral_deterioration": -1.0,  # penalty (needs time-series)
}
# Each feature is min-max normalized to [0,1] against a fixed, plausible upper bound
# BEFORE weighting, so the weights express IMPORTANCE only -- not a scale artifact of
# features that live on very different raw ranges. The components reported to the
# dashboard are the signed, weighted contributions, so the score is exactly their sum
# and the driver bar reads as "what moved the score."
NORM_BOUNDS = {
    "voluntary_ce_gap": 1.0,         # 100% CE above the requirement = full credit
    "reserve_ratio": 0.03,           # 3% reserve = full credit
    "excess_spread_reliance": 0.40,  # 40% of CE from excess spread = full penalty
}

# Eq. 27 deferred-loss flag weights.
DEFERRED_WEIGHTS = {
    "excess_spread_reliance_flag": 1.0,
    "small_reserve_flag": 0.8,
    "thin_voluntary_ce_flag": 1.2,
}

# Deferred-loss flag thresholds (tunable). A deal trips a flag only when its
# structure is genuinely fragile on that dimension; a conservatively structured
# deal legitimately shows zero flags. Tighten these to make the detector more
# sensitive, loosen to make it stricter.
FLAG_XS_RELIANCE = 0.15       # >15% of the CE stack leans on soft excess spread
FLAG_THIN_VOL_CE = 0.10       # <10% CE above the rating requirement (thin cushion)
FLAG_SMALL_RESERVE = 0.0075   # reserve < 0.75% of pool


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
    eps = 1e-6
    available_ce = deal["initial_oc_pct"] + deal["total_subordination_pct"]
    required = max(float(deal["required_ce_senior_pct"]), eps)
    xs_reliance = _excess_spread_reliance(deal)

    # Raw features (normalized by the EXTERNAL rating requirement, not our PD*LGD).
    vol_ce_gap_rel = (available_ce - required) / required
    reserve_ratio = float(deal["reserve_fund_pct"])
    raw = {
        "voluntary_ce_gap": vol_ce_gap_rel,
        "reserve_ratio": reserve_ratio,
        "excess_spread_reliance": xs_reliance,
        "collateral_deterioration": 0.0,  # needs time-series; 0 in static demo
    }

    # Min-max normalize each feature to [0,1], then weight -> signed contribution.
    # Components are the signed contributions, so score == sum(comp.values()).
    comp = {}
    for k, w in CREDIBILITY_WEIGHTS.items():
        bound = NORM_BOUNDS.get(k, 1.0)
        norm = min(max(raw[k] / bound, 0.0), 1.0) if bound else 0.0
        comp[k] = round(w * norm, 4)
    score = sum(comp.values())

    flags = {
        "excess_spread_reliance_flag": xs_reliance > FLAG_XS_RELIANCE,
        "small_reserve_flag": reserve_ratio < FLAG_SMALL_RESERVE,
        "thin_voluntary_ce_flag": vol_ce_gap_rel < FLAG_THIN_VOL_CE,
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
