"""Performance triggers: cumulative-net-loss (CNL) trigger schedules and breaches.

Subprime auto ABS protect the senior notes with *triggers* -- contractual loss
(and delinquency) limits that ramp up over the deal's life. While realized CNL
stays under the schedule, cash flows normally and excess spread is released to the
residual. The first month realized CNL breaches the limit, the deal fails its
trigger: excess spread is trapped, the OC target steps up, and CE rebuilds faster.
That is the mechanical link to the dynamic-CE view (engine.dynamics): a breach here
drives the OC step-up there.

We don't have prospectus trigger tables in the synthetic dataset, so the schedule
is modeled parametrically from two fields already on each deal:

  * the priced loss expectation (assumed_pd x assumed_lgd), which sets the terminal
    level the schedule ramps toward, and
  * trigger_strength (0-1): a tighter trigger (higher strength) sits closer to the
    priced expectation -- less headroom before it bites, i.e. more protective.

Swap `cnl_trigger_schedule` for real per-deal step tables when you have them; the
breach logic downstream is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import formulas

# Headroom of the terminal CNL limit over priced expectation. A strength-0 deal
# gets the loosest limit; strength-1 the tightest. Tuned so subprime strengths
# (~0.40-0.58) land around 1.5x-1.8x priced loss.
HEADROOM_BASE = 1.08
HEADROOM_SPAN = 0.35

# CNL triggers don't bind during the initial seasoning window -- losses are near
# zero and the schedule is near zero, so a comparison there is just noise. Ignore
# breaches before this month, matching how real deals carve out an early period.
SEASONING_MONTHS = 6


# Weight on the linear ramp when shaping the trigger schedule. Real CNL triggers
# carry generous headroom early (losses should be minimal while the deal is young)
# and tighten as it seasons, so the schedule is front-loaded relative to the
# S-shaped loss curve. A hot deal therefore starts under its trigger and crosses it
# mid-life -- not in month 2 -- which is how breaches actually look.
LINEAR_BLEND = 0.45


def _loss_shape(n_months: int) -> np.ndarray:
    """Normalized S-shaped ramp (0->1) matching a typical CNL accumulation curve."""
    t = np.arange(1, n_months + 1)
    midpoint = n_months * 0.45
    s = 1.0 / (1.0 + np.exp(-0.18 * (t - midpoint)))
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s


def _schedule_shape(n_months: int) -> np.ndarray:
    """Front-loaded trigger ramp (0->1): blend of a linear ramp and the loss S-curve."""
    linear = np.linspace(1.0 / n_months, 1.0, n_months)
    return LINEAR_BLEND * linear + (1.0 - LINEAR_BLEND) * _loss_shape(n_months)


def terminal_limit(deal: pd.Series) -> float:
    """Lifetime CNL trigger ceiling for a deal (fraction of original pool)."""
    priced = formulas.expected_loss(deal["assumed_pd"], deal["assumed_lgd"])
    strength = float(np.clip(deal.get("trigger_strength", 0.5), 0.0, 1.0))
    headroom = HEADROOM_BASE + HEADROOM_SPAN * (1.0 - strength)
    return float(priced * headroom)


def cnl_trigger_schedule(deal: pd.Series, n_months: int) -> np.ndarray:
    """Month-by-month CNL trigger limits (fraction of original pool)."""
    return terminal_limit(deal) * _schedule_shape(n_months)


def evaluate(deal: pd.Series, perf: pd.DataFrame) -> dict:
    """Compare realized CNL to the modeled trigger schedule for one deal.

    Returns a dict with a per-period table plus summary fields:
      table        : DataFrame[month, period_end, cnl_limit, realized_cnl,
                     headroom, breached]
      breach_month : 1-based month of first breach, or None
      breached     : bool
      terminal_limit, min_headroom
    """
    if perf is None or len(perf) == 0:
        return {"table": pd.DataFrame(
            columns=["month", "period_end", "cnl_limit", "realized_cnl",
                     "headroom", "breached"]),
            "breach_month": None, "breached": False,
            "terminal_limit": terminal_limit(deal), "min_headroom": None}

    perf = perf.sort_values("period_end").reset_index(drop=True)
    n = len(perf)
    months = np.arange(1, n + 1)
    limits = cnl_trigger_schedule(deal, n)
    realized = perf["cum_net_loss_rate"].fillna(0.0).to_numpy(dtype=float)
    headroom = limits - realized
    seasoned = months >= SEASONING_MONTHS
    breached = (realized > limits) & seasoned

    table = pd.DataFrame({
        "month": months,
        "period_end": perf["period_end"].to_numpy(),
        "cnl_limit": limits,
        "realized_cnl": realized,
        "headroom": headroom,
        "breached": breached,
    })
    breach_month = int(months[breached][0]) if breached.any() else None
    seasoned_hr = headroom[seasoned]

    return {
        "table": table,
        "breach_month": breach_month,
        "breached": bool(breached.any()),
        "terminal_limit": float(limits[-1]),
        "min_headroom": float(seasoned_hr.min()) if seasoned_hr.size else None,
    }
