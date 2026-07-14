"""CE-centric ABS tranche valuation from a simulated collateral-loss distribution.

A tranche is economically a call spread on collateral losses: its attachment and
detachment points are the strikes, and **the attachment point IS the credit
enhancement beneath it**. So credit enhancement is the central pricing input here,
not an add-on.

Given a simulated distribution of portfolio (collateral) loss rates -- from the
Vasicek engine in engine.montecarlo -- we compute a tranche's expected loss and its
tail (unexpected) loss via the standard attachment/detachment loss function, then
price it with the fundamental credit identity:

    fair spread  ≈  annualized expected loss  +  λ · annualized unexpected loss

where λ is the market price of risk (premium demanded per unit of tail risk). Price
is the PV of the tranche's promised cashflows discounted at (risk-free + fair
spread). Every number traces to inputs -- nothing is a black box.

DEFENSIBILITY NOTE: this is the standard structural / loss-distribution approach to
tranche valuation. Its known failure mode -- the one that broke Gaussian-copula
pricing in 2008 -- is UNDERESTIMATING correlation and the tail. So this module never
trusts one correlation: callers should value across a range of correlations / stress
regimes and read the tail explicitly, not quote a single point. See value_surface().
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TrancheValue:
    attachment: float
    detachment: float
    coupon: float
    wal: float
    tranche_el: float       # lifetime expected loss (fraction of the tranche)
    tranche_ul: float       # tail loss beyond EL (ES99 - EL), fraction of tranche
    fair_spread: float      # required annual credit spread (decimal)
    el_spread: float        # expected-loss component of the spread
    rp_spread: float        # risk-premium (tail) component
    fair_price: float       # intrinsic value per 100, given the tranche coupon

    def as_dict(self) -> dict:
        return asdict(self)


def tranche_loss_fraction(losses: np.ndarray, attach: float, detach: float) -> np.ndarray:
    """Fraction of a tranche wiped out per simulated collateral loss (0-1)."""
    width = detach - attach
    if width <= 0:
        return np.zeros_like(losses, dtype=float)
    return np.clip((losses - attach) / width, 0.0, 1.0)


def value_tranche(losses: np.ndarray, attach: float, detach: float, coupon: float,
                  wal: float, risk_free: float = 0.04, price_of_risk: float = 1.0,
                  tail_q: float = 0.99) -> TrancheValue:
    """Value one tranche off a simulated collateral-loss distribution.

    Parameters
    ----------
    losses : array of simulated portfolio loss rates (fraction of pool).
    attach, detach : tranche attachment/detachment (fractions of pool from bottom);
        `attach` is the credit enhancement beneath the tranche.
    coupon : the tranche's coupon (decimal).
    wal : weighted average life in years (the tenor used to annualize and discount).
    risk_free : benchmark discount rate.
    price_of_risk : λ, the premium demanded per unit of annualized tail loss.
    """
    losses = np.asarray(losses, dtype=float)
    tl = tranche_loss_fraction(losses, attach, detach)

    el = float(tl.mean())                                    # lifetime expected loss
    var = float(np.quantile(tl, tail_q))
    tail = tl[tl >= var]
    es = float(tail.mean()) if tail.size else var            # expected shortfall
    ul = max(0.0, es - el)                                   # unexpected (tail) loss

    wal = max(float(wal), 1e-6)
    el_spread = el / wal                                     # annualized expected loss
    rp_spread = price_of_risk * ul / wal                    # annualized risk premium
    fair_spread = el_spread + rp_spread

    # Intrinsic value: PV of the tranche's promised cashflows (coupon annuity +
    # principal) discounted at the required yield = risk-free + fair spread.
    r = risk_free + fair_spread
    if r > 0:
        annuity = coupon * (1.0 - (1.0 + r) ** (-wal)) / r
        fair_price = 100.0 * (annuity + (1.0 + r) ** (-wal))
    else:
        fair_price = 100.0

    return TrancheValue(
        attachment=attach, detachment=detach, coupon=coupon, wal=wal,
        tranche_el=el, tranche_ul=ul, fair_spread=fair_spread,
        el_spread=el_spread, rp_spread=rp_spread, fair_price=fair_price)


def ce_value_bps(losses: np.ndarray, attach: float, detach: float, wal: float,
                 risk_free: float = 0.04, price_of_risk: float = 1.0) -> float:
    """Spread (bps) the credit enhancement saves the tranche.

    The extra spread it would demand if the same-width tranche sat at the BOTTOM of
    the stack (attachment 0) instead of behind its actual enhancement.
    """
    with_ce = value_tranche(losses, attach, detach, 0.0, wal,
                            risk_free, price_of_risk).fair_spread
    width = detach - attach
    no_ce = value_tranche(losses, 0.0, max(width, 1e-6), 0.0, wal,
                          risk_free, price_of_risk).fair_spread
    return (no_ce - with_ce) * 1e4


def value_surface(loss_sets: dict, attach: float, detach: float, coupon: float,
                  wal: float, risk_free: float = 0.04,
                  price_of_risk: float = 1.0) -> dict:
    """Fair spread under several loss distributions (e.g. one per stress regime).

    `loss_sets` maps a label -> loss array. Returns label -> fair_spread, so the
    valuation is reported as a RANGE across correlation/stress, never a single point.
    """
    return {label: value_tranche(losses, attach, detach, coupon, wal,
                                 risk_free, price_of_risk).fair_spread
            for label, losses in loss_sets.items()}
