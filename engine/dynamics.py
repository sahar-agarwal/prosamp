"""Time evolution of credit enhancement (the 'dynamic CE' view).

CE is not a static number set at closing -- it is a *living* quantity. As the pool
amortizes, the senior notes pay down fastest while the subordinate notes and the
overcollateralization (OC) account are locked out, so subordination measured as a
fraction of the *current* pool grows month over month. Excess spread is also
trapped early to build OC up to its target. Realized losses run the other way,
eroding the first-loss pieces from the bottom.

This module turns a deal's realized performance time series into a CE path so the
dashboard can show, at any month since closing:

  * structural senior CE (% of current pool) -- grows as the pool amortizes / OC builds
  * realized cumulative net loss (% of original pool) -- what has actually burned
  * available cushion (senior CE at issuance minus realized loss) -- the honest
    adequacy test, on a consistent original-pool basis

Pure functions, plain numbers / DataFrames in and out, so every tab can reuse them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OC_RAMP_MONTHS = 12        # months over which OC builds from initial toward target
OC_STEP_MULTIPLIER = 1.5   # OC target step-up once a CNL trigger is breached


def ce_path(deal: pd.Series, perf: pd.DataFrame, senior_ce0: float,
            breach_month: int | None = None,
            oc_step_multiplier: float = OC_STEP_MULTIPLIER) -> pd.DataFrame:
    """Build the month-by-month CE path for one deal.

    Parameters
    ----------
    deal : deal-level row (needs initial_oc_pct, target_oc_pct,
           original_pool_balance).
    perf : that deal's realized_performance rows (needs period_end,
           period_end_balance, cum_net_loss_rate); may be empty.
    senior_ce0 : senior credit enhancement at issuance (fraction of pool) --
           typically the senior tranche's attachment point.
    breach_month : 1-based month a CNL trigger breaches (from engine.triggers),
           after which the OC target steps up. None = no breach.
    """
    cols = ["month", "period_end", "pool_factor", "oc_pct",
            "structural_ce_pct", "realized_cnl", "available_ce_pct",
            "trigger_active"]
    if perf is None or len(perf) == 0:
        return pd.DataFrame(columns=cols)

    perf = perf.sort_values("period_end").reset_index(drop=True)
    pool0 = float(deal["original_pool_balance"])
    init_oc = float(deal["initial_oc_pct"])
    target_oc = float(deal["target_oc_pct"])

    rows = []
    for i, r in perf.iterrows():
        m = i + 1
        pf = float(np.clip(r["period_end_balance"] / pool0, 0.05, 1.0)) if pool0 else 1.0

        active = breach_month is not None and m >= breach_month
        target_eff = target_oc * (oc_step_multiplier if active else 1.0)
        oc_t = min(target_eff,
                   init_oc + (target_eff - init_oc) * min(1.0, m / OC_RAMP_MONTHS))
        incr_oc = max(0.0, oc_t - init_oc)

        # Subordination $ is ~locked while the pool shrinks -> CE / current pool
        # rises like senior_ce0 / pool_factor; trapped OC adds on top.
        structural = float(min(0.99, senior_ce0 / pf + incr_oc))

        cnl = float(r["cum_net_loss_rate"]) if pd.notna(r["cum_net_loss_rate"]) else 0.0
        available = senior_ce0 - cnl  # honest cushion, original-pool basis

        rows.append({
            "month": m,
            "period_end": r["period_end"],
            "pool_factor": pf,
            "oc_pct": oc_t,
            "structural_ce_pct": structural,
            "realized_cnl": cnl,
            "available_ce_pct": available,
            "trigger_active": bool(active),
        })
    return pd.DataFrame(rows, columns=cols)


def snapshot(path: pd.DataFrame, month: int) -> dict:
    """CE state at a chosen month (clamped to the available range)."""
    if path is None or len(path) == 0:
        return {}
    m = int(np.clip(month, int(path["month"].min()), int(path["month"].max())))
    row = path.loc[path["month"] == m].iloc[0]
    return row.to_dict()
