"""Monte Carlo collateral-loss simulation (single-factor Vasicek model).

The Vasicek large-homogeneous-pool model is the market-standard way to turn a
single PD + asset correlation into a full loss *distribution* (it underlies Basel
IRB and most ABS loss modeling). We simulate the systematic factor, derive a
conditional default rate per draw, and multiply by LGD to get a portfolio loss
rate. Stress = scaling PD by a shock multiplier (Eq. 26).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class MCResult:
    losses: np.ndarray          # simulated portfolio loss rates (fraction of pool)
    pd_used: float
    expected_loss: float
    p_ce_exhaustion: float      # Eq. 25: P(loss > available CE)
    tail_loss_99: float         # 99th percentile loss
    expected_shortfall_99: float

    def as_dict(self) -> dict:
        return {
            "pd_used": self.pd_used,
            "expected_loss": self.expected_loss,
            "p_ce_exhaustion": self.p_ce_exhaustion,
            "tail_loss_99": self.tail_loss_99,
            "expected_shortfall_99": self.expected_shortfall_99,
        }


def simulate(pd_: float, lgd: float, available_ce: float,
             correlation: float = 0.15, n_sims: int = 50_000,
             shock_multiplier: float = 1.0, seed: int | None = None) -> MCResult:
    """Simulate collateral loss paths and summarize against available CE.

    Parameters
    ----------
    pd_ : base-case probability of default (fraction).
    lgd : loss given default (fraction).
    available_ce : credit enhancement available to the position (fraction of pool).
    correlation : asset correlation (Vasicek rho); higher = fatter tail.
    shock_multiplier : multiply PD for stress scenarios (Eq. 26).
    """
    rng = np.random.default_rng(seed)
    pd_stressed = float(np.clip(pd_ * shock_multiplier, 1e-6, 0.999))
    rho = float(np.clip(correlation, 1e-4, 0.99))

    z = rng.standard_normal(n_sims)            # systematic factor
    cond_default = norm.cdf(
        (norm.ppf(pd_stressed) + np.sqrt(rho) * z) / np.sqrt(1.0 - rho)
    )
    losses = cond_default * lgd

    exhaust = float(np.mean(losses > available_ce))
    var99 = float(np.quantile(losses, 0.99))
    tail = losses[losses >= var99]
    es99 = float(tail.mean()) if tail.size else var99

    return MCResult(
        losses=losses,
        pd_used=pd_stressed,
        expected_loss=float(losses.mean()),
        p_ce_exhaustion=exhaust,
        tail_loss_99=var99,
        expected_shortfall_99=es99,
    )


# Named stress regimes (PD multipliers) for Tab 1 / Tab 3 controls.
SHOCK_REGIMES = {
    "Base case": 1.0,
    "Mild recession": 1.5,
    "COVID disruption": 2.0,
    "Global Financial Crisis": 3.0,
    "Severe / tail": 4.0,
}

# Asset correlation (Vasicek rho) is NOT a free knob. Its BASE level is grounded in
# the Basel II/III IRB "other retail" supervisory formula -- the regulatory standard
# for granular retail pools, which auto loans fall under. That formula makes rho a
# DECREASING function of PD (diversified, high-default retail pools have low asset
# correlation) bounded between 3% and 16%:
#     R(PD) = R_min * w(PD) + R_max * (1 - w(PD)),
#     w(PD) = (1 - e^(-k*PD)) / (1 - e^(-k)),   k = 35  for other retail.
BASEL_RETAIL_RHO_MIN = 0.03
BASEL_RETAIL_RHO_MAX = 0.16
BASEL_RETAIL_K = 35.0


def basel_retail_correlation(pd_: float) -> float:
    """Basel IRB 'other retail' asset correlation as a function of PD (3%-16%)."""
    p = float(np.clip(pd_, 1e-6, 0.999))
    w = (1.0 - np.exp(-BASEL_RETAIL_K * p)) / (1.0 - np.exp(-BASEL_RETAIL_K))
    return float(BASEL_RETAIL_RHO_MIN * w + BASEL_RETAIL_RHO_MAX * (1.0 - w))


# Basel's correlation is point-in-time and does NOT capture the well-documented
# widening of correlations in a downturn (defaults cluster when one macro shock hits
# every borrower at once). Each stress regime therefore adds an uplift on top of the
# Basel base. This uplift is the one piece that is judgement rather than standard --
# a stylized downturn overlay, calibrate to your own loss co-movement before use.
REGIME_RHO_UPLIFT = {
    "Base case": 0.00,
    "Mild recession": 0.05,
    "COVID disruption": 0.12,
    "Global Financial Crisis": 0.22,
    "Severe / tail": 0.32,
}


def rho_for_regime(pd_: float, regime: str, adjustment: float = 0.0) -> float:
    """Asset correlation for a deal under a stress regime.

    Basel 'other retail' base correlation at the deal's PD, plus the regime's
    downturn uplift, plus a manual adjustment, clipped to a sane (0.02, 0.90) band.
    """
    base = basel_retail_correlation(pd_)
    uplift = REGIME_RHO_UPLIFT.get(regime, 0.0)
    return float(np.clip(base + uplift + adjustment, 0.02, 0.90))
