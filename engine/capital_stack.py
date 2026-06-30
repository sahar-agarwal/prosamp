"""Capital-stack loss allocation and stakeholder risk translation (Eqs. 13-18).

Given a collateral loss (fraction of pool), waterfall it bottom-up through the
structure and report how each tranche and stakeholder experiences it. attachment_pct
/ detachment_pct are fractions of the pool measured from the BOTTOM of the stack.
"""
from __future__ import annotations

import pandas as pd


def tranche_loss(collateral_loss: float, attach: float, detach: float) -> float:
    """Fraction of a tranche wiped out by a given collateral loss (0-1)."""
    width = detach - attach
    if width <= 0:
        return 0.0
    return float(min(max((collateral_loss - attach) / width, 0.0), 1.0))


def allocate(tranches: pd.DataFrame, collateral_loss: float) -> pd.DataFrame:
    """Per-tranche loss for one collateral-loss level, ordered senior->junior."""
    out = tranches.copy()
    out["loss_fraction"] = out.apply(
        lambda r: tranche_loss(collateral_loss, r["attachment_pct"],
                               r["detachment_pct"]), axis=1)
    out["loss_amount"] = out["loss_fraction"] * out["original_balance"]
    return out.sort_values("attachment_pct", ascending=False)


def senior_exposure(collateral_loss: float, ce_beneath_senior: float,
                    senior_balance_frac: float) -> dict:
    """Eqs. 13-14: senior noteholder view."""
    exposure = max(0.0, collateral_loss - ce_beneath_senior)
    loss_rate = exposure / senior_balance_frac if senior_balance_frac else 0.0
    return {"ce_adjusted_exposure": exposure, "loss_rate": loss_rate}


def residual_value(pv_excess_spread: float, first_loss_absorption: float) -> float:
    """Eq. 15: residual holder economics."""
    return pv_excess_spread - first_loss_absorption


def residual_erosion(base_residual: float, stressed_residual: float) -> float:
    """Eq. 16."""
    return base_residual - stressed_residual


def stakeholder_view(deal: pd.Series, tranches: pd.DataFrame,
                     collateral_loss: float) -> dict:
    """Translate one collateral shock into each stakeholder's exposure."""
    senior = tranches.sort_values("attachment_pct").iloc[-1]
    ce_beneath_senior = senior["attachment_pct"]
    senior_frac = senior["original_balance"] / deal["original_pool_balance"]

    sr = senior_exposure(collateral_loss, ce_beneath_senior, senior_frac)

    # Residual: PV of excess spread (rough) minus first-loss it absorbs.
    pv_xs = deal["excess_spread_pct"] * 2.5  # ~2.5y of excess spread, undiscounted-ish
    first_loss = min(collateral_loss, deal["initial_oc_pct"] + deal["reserve_fund_pct"])
    base_resid = residual_value(pv_xs, deal["initial_oc_pct"] + deal["reserve_fund_pct"]
                                - min(deal["assumed_pd"] * deal["assumed_lgd"],
                                      deal["initial_oc_pct"] + deal["reserve_fund_pct"]))
    stressed_resid = residual_value(pv_xs, first_loss)

    # Originator retained-risk cost (Eq. 17, simplified): reserve + retained sub.
    orig_cost = deal["reserve_fund_pct"] + 0.05 * deal["total_subordination_pct"]

    # Rating agency: cushion of available CE over requirement (Eqs. 19-20).
    avail_ce = deal["initial_oc_pct"] + deal["total_subordination_pct"]
    rating_cushion = avail_ce - deal["required_ce_senior_pct"]

    return {
        "senior": sr,
        "residual": {
            "base_value": base_resid,
            "stressed_value": stressed_resid,
            "erosion": residual_erosion(base_resid, stressed_resid),
        },
        "originator": {"retained_risk_cost": orig_cost},
        "rating_agency": {
            "available_ce": avail_ce,
            "required_ce": deal["required_ce_senior_pct"],
            "cushion": rating_cushion,
        },
    }
