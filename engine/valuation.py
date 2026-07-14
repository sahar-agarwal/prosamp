"""
CE-centric ABS tranche valuation from a simulated collateral-loss distribution.

A tranche is economically a call spread on collateral losses: its attachment and
detachment points are the strikes, and the attachment point IS the credit
enhancement beneath it. So credit enhancement is the central pricing input here,
not an add-on.

Given a simulated distribution of portfolio (collateral) loss rates -- from the
Vasicek engine in engine.montecarlo -- we compute a tranche's expected loss and
its tail (unexpected) loss via the standard attachment/detachment loss function,
then price it with the fundamental credit identity:

    fair spread ~= annualized expected loss + lambda * annualized unexpected loss

where lambda is the market price of risk, i.e. the premium demanded per unit of
tail risk.

This module supports two valuation modes:

1. Simple WAL valuation:
   - treats the tranche like a simplified bond with coupon annuity plus principal
     discounted at risk-free + fair spread.

2. Period-by-period cashflow valuation:
   - explicitly projects expected coupon and principal cashflows through time,
     including amortization and loss timing.
   - because expected losses are already reflected by reduced expected cashflows,
     those cashflows are discounted at risk-free + risk-premium spread, not
     risk-free + expected-loss spread + risk-premium spread. This avoids
     double-counting expected loss.

DEFENSIBILITY NOTE:
This is the standard structural / loss-distribution approach to tranche
valuation. Its known failure mode -- the one that broke Gaussian-copula pricing
in 2008 -- is underestimating correlation and the tail. So this module never
requires callers to trust one correlation: callers should value across a range
of correlations / stress regimes and read the tail explicitly, not quote a
single point. See value_surface() and value_cashflow_surface().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


CouponBasis = Literal["beginning", "post_loss", "average"]


@dataclass
class TrancheValue:
    attachment: float
    detachment: float
    coupon: float
    wal: float

    tranche_el: float       # lifetime expected loss, fraction of tranche par
    tranche_ul: float       # ES tail loss beyond EL, fraction of tranche par

    fair_spread: float      # required annual credit spread, decimal
    el_spread: float        # expected-loss component of spread, decimal
    rp_spread: float        # risk-premium / tail component of spread, decimal

    fair_price: float       # intrinsic value per 100 par, given tranche coupon

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CashflowPeriod:
    period: int
    time: float

    scheduled_principal: float

    expected_begin_balance: float
    expected_loss: float
    expected_coupon: float
    expected_principal: float
    expected_ending_balance: float

    expected_total_cashflow: float
    discount_factor: float
    expected_pv: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrancheCashflowValue:
    attachment: float
    detachment: float
    coupon: float
    wal: float

    tranche_el: float       # lifetime expected loss after timing/amortization
    tranche_ul: float       # ES tail loss beyond EL after timing/amortization

    fair_spread: float      # EL spread + risk-premium spread
    el_spread: float        # annualized expected loss
    rp_spread: float        # annualized tail-risk premium

    discount_rate: float    # risk_free + rp_spread for expected cashflows
    fair_price: float       # PV of expected cashflows per 100 par

    cashflows: list[CashflowPeriod]

    def as_dict(self) -> dict:
        return asdict(self)


def _as_1d_float_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a one-dimensional array.")

    if arr.size == 0:
        raise ValueError(f"`{name}` must not be empty.")

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` contains non-finite values.")

    return arr


def _validate_nonnegative_vector(
    x: np.ndarray,
    name: str,
    n: int | None = None,
    require_sum_one: bool = False,
    tol: float = 1e-8,
) -> np.ndarray:
    arr = _as_1d_float_array(x, name)

    if n is not None and arr.size != n:
        raise ValueError(f"`{name}` must have length {n}; got {arr.size}.")

    if np.any(arr < 0.0):
        raise ValueError(f"`{name}` must be non-negative.")

    if require_sum_one and not np.isclose(arr.sum(), 1.0, atol=tol):
        raise ValueError(f"`{name}` must sum to 1.0; got {arr.sum():.12f}.")

    return arr


def tranche_loss_fraction(
    losses: np.ndarray,
    attach: float,
    detach: float,
) -> np.ndarray:
    """Fraction of a tranche wiped out per simulated collateral loss.

    Parameters
    ----------
    losses:
        Simulated portfolio/collateral loss rates, as fractions of the pool.

    attach:
        Tranche attachment point, as fraction of the pool. This is the credit
        enhancement beneath the tranche.

    detach:
        Tranche detachment point, as fraction of the pool.

    Returns
    -------
    np.ndarray
        Tranche loss fractions, clipped between 0 and 1.
    """
    losses = np.asarray(losses, dtype=float)

    width = detach - attach
    if width <= 0.0:
        return np.zeros_like(losses, dtype=float)

    return np.clip((losses - attach) / width, 0.0, 1.0)


def value_tranche(
    losses: np.ndarray,
    attach: float,
    detach: float,
    coupon: float,
    wal: float,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
    tail_q: float = 0.99,
) -> TrancheValue:
    """Value one tranche off a simulated collateral-loss distribution.

    This is the simple WAL-based valuation mode. It maps lifetime collateral
    losses into lifetime tranche losses, computes EL and tail UL, converts those
    into a fair annual spread, and discounts a simplified coupon annuity plus
    principal repayment at risk-free + fair spread.

    Parameters
    ----------
    losses:
        Simulated portfolio loss rates, as fractions of the pool.

    attach, detach:
        Tranche attachment/detachment points, as fractions of the pool.
        `attach` is the credit enhancement beneath the tranche.

    coupon:
        Tranche coupon, annual decimal rate. For example, 8% is 0.08.

    wal:
        Weighted average life in years. Used to annualize losses and discount
        the simplified cashflows.

    risk_free:
        Benchmark discount rate.

    price_of_risk:
        Lambda, the premium demanded per unit of annualized tail loss.

    tail_q:
        Quantile used for expected shortfall. Default is 0.99.

    Returns
    -------
    TrancheValue
    """
    losses = _as_1d_float_array(losses, "losses")

    if not 0.0 <= tail_q <= 1.0:
        raise ValueError("`tail_q` must be between 0 and 1.")

    tl = tranche_loss_fraction(losses, attach, detach)

    el = float(tl.mean())

    var = float(np.quantile(tl, tail_q))
    tail = tl[tl >= var]
    es = float(tail.mean()) if tail.size else var

    ul = max(0.0, es - el)

    wal = max(float(wal), 1e-6)

    el_spread = el / wal
    rp_spread = price_of_risk * ul / wal
    fair_spread = el_spread + rp_spread

    # Simplified intrinsic value:
    # PV of promised coupon annuity + promised principal, discounted at the
    # required yield = risk-free + fair spread.
    r = risk_free + fair_spread

    if r > 0.0:
        annuity = coupon * (1.0 - (1.0 + r) ** (-wal)) / r
        fair_price = 100.0 * (annuity + (1.0 + r) ** (-wal))
    else:
        # Basic fallback for zero/negative discount rates in the simplified mode.
        fair_price = 100.0

    return TrancheValue(
        attachment=float(attach),
        detachment=float(detach),
        coupon=float(coupon),
        wal=float(wal),
        tranche_el=el,
        tranche_ul=ul,
        fair_spread=fair_spread,
        el_spread=el_spread,
        rp_spread=rp_spread,
        fair_price=fair_price,
    )


def value_tranche_cashflows(
    losses: np.ndarray,
    attach: float,
    detach: float,
    coupon: float,
    principal_schedule: np.ndarray,
    period_lengths: np.ndarray | None = None,
    loss_timing: np.ndarray | None = None,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
    tail_q: float = 0.99,
    coupon_basis: CouponBasis = "beginning",
) -> TrancheCashflowValue:
    """Value an amortizing ABS tranche using period-by-period expected cashflows.

    This is the more realistic amortizing-tranche valuation mode. It explicitly
    projects expected coupon and principal cashflows period by period.

    Important:
    Because this function explicitly reduces expected cashflows for expected
    losses, it discounts those expected cashflows at:

        risk_free + risk-premium spread

    not:

        risk_free + expected-loss spread + risk-premium spread

    This avoids double-counting expected loss.

    Parameters
    ----------
    losses:
        Simulated lifetime collateral loss rates, as fractions of the pool.

    attach, detach:
        Tranche attachment and detachment points, as fractions of the pool.
        `attach` is the credit enhancement beneath the tranche.

    coupon:
        Annual coupon rate, decimal. For example, 8% is 0.08.

    principal_schedule:
        Scheduled principal payments by period, as fractions of original tranche
        principal. Must be non-negative and sum to 1.0.

        Example: quarterly straight-line amortization over 5 years:

            np.full(20, 1.0 / 20)

    period_lengths:
        Year fractions for each period. If None, assumes annual periods.

        Example: quarterly periods:

            np.full(20, 0.25)

    loss_timing:
        Distribution of lifetime tranche losses through time. Must sum to 1.0.
        If None, losses are assumed to arrive uniformly across periods.

        Example: front-loaded losses:

            np.array([0.20, 0.15, 0.12, ...])

    risk_free:
        Risk-free discount rate.

    price_of_risk:
        Lambda, the premium demanded per unit of annualized tail loss.

    tail_q:
        Tail quantile used for expected shortfall. Default is 0.99.

    coupon_basis:
        Balance convention for coupon calculation.

        - "beginning": coupon on beginning-of-period balance
        - "post_loss": coupon on balance after period losses
        - "average": coupon on average of beginning and ending balance

    Returns
    -------
    TrancheCashflowValue
    """
    losses = _as_1d_float_array(losses, "losses")

    if not 0.0 <= tail_q <= 1.0:
        raise ValueError("`tail_q` must be between 0 and 1.")

    if detach <= attach:
        raise ValueError("`detach` must be greater than `attach`.")

    if coupon_basis not in {"beginning", "post_loss", "average"}:
        raise ValueError(
            "`coupon_basis` must be one of: 'beginning', 'post_loss', 'average'."
        )

    principal_schedule = _validate_nonnegative_vector(
        principal_schedule,
        "principal_schedule",
        require_sum_one=True,
    )

    n_periods = principal_schedule.size

    if period_lengths is None:
        period_lengths = np.ones(n_periods, dtype=float)
    else:
        period_lengths = _validate_nonnegative_vector(
            period_lengths,
            "period_lengths",
            n=n_periods,
        )

    if np.any(period_lengths <= 0.0):
        raise ValueError("`period_lengths` must be strictly positive.")

    if loss_timing is None:
        loss_timing = np.full(n_periods, 1.0 / n_periods, dtype=float)
    else:
        loss_timing = _validate_nonnegative_vector(
            loss_timing,
            "loss_timing",
            n=n_periods,
            require_sum_one=True,
        )

    payment_times = np.cumsum(period_lengths)

    # Scheduled WAL based on promised principal, not loss-adjusted principal.
    wal = float(np.sum(payment_times * principal_schedule))
    wal = max(wal, 1e-6)

    # Convert each simulated lifetime collateral loss into terminal tranche loss.
    terminal_tranche_loss = tranche_loss_fraction(losses, attach, detach)
    n_sims = terminal_tranche_loss.size

    # Scenario-level outstanding balance, as fraction of original tranche par.
    outstanding = np.ones(n_sims, dtype=float)

    expected_begin_balances = np.zeros(n_periods, dtype=float)
    expected_losses = np.zeros(n_periods, dtype=float)
    expected_coupons = np.zeros(n_periods, dtype=float)
    expected_principals = np.zeros(n_periods, dtype=float)
    expected_ending_balances = np.zeros(n_periods, dtype=float)

    total_realized_loss = np.zeros(n_sims, dtype=float)

    for i in range(n_periods):
        begin_balance = outstanding.copy()

        # Allocate each scenario's terminal tranche loss through time.
        raw_period_loss = terminal_tranche_loss * loss_timing[i]

        # Loss cannot exceed remaining tranche balance.
        period_loss = np.minimum(begin_balance, raw_period_loss)
        balance_after_loss = begin_balance - period_loss

        # Scheduled principal is stated as a fraction of original tranche par.
        # It cannot exceed remaining live balance after losses.
        scheduled_principal = principal_schedule[i]
        principal_paid = np.minimum(balance_after_loss, scheduled_principal)

        ending_balance = balance_after_loss - principal_paid

        if coupon_basis == "beginning":
            coupon_balance = begin_balance
        elif coupon_basis == "post_loss":
            coupon_balance = balance_after_loss
        else:
            coupon_balance = 0.5 * (begin_balance + ending_balance)

        coupon_paid = coupon * period_lengths[i] * coupon_balance

        expected_begin_balances[i] = float(begin_balance.mean())
        expected_losses[i] = float(period_loss.mean())
        expected_coupons[i] = float(coupon_paid.mean())
        expected_principals[i] = float(principal_paid.mean())
        expected_ending_balances[i] = float(ending_balance.mean())

        total_realized_loss += period_loss
        outstanding = ending_balance

    # Lifetime expected tranche loss after considering amortization and loss timing.
    el = float(total_realized_loss.mean())

    var = float(np.quantile(total_realized_loss, tail_q))
    tail = total_realized_loss[total_realized_loss >= var]
    es = float(tail.mean()) if tail.size else var

    ul = max(0.0, es - el)

    el_spread = el / wal
    rp_spread = price_of_risk * ul / wal
    fair_spread = el_spread + rp_spread

    # Expected losses are already inside the projected cashflows, so only the
    # tail-risk premium is added to risk-free for discounting expected cashflows.
    discount_rate = risk_free + rp_spread

    if discount_rate <= -1.0:
        raise ValueError("`risk_free + rp_spread` must be greater than -100%.")

    discount_factors = (1.0 + discount_rate) ** (-payment_times)

    cashflows: list[CashflowPeriod] = []
    pv_total = 0.0

    for i in range(n_periods):
        expected_total_cf = expected_coupons[i] + expected_principals[i]
        expected_pv = expected_total_cf * discount_factors[i]
        pv_total += expected_pv

        cashflows.append(
            CashflowPeriod(
                period=i + 1,
                time=float(payment_times[i]),
                scheduled_principal=float(principal_schedule[i]),
                expected_begin_balance=float(expected_begin_balances[i]),
                expected_loss=float(expected_losses[i]),
                expected_coupon=float(expected_coupons[i]),
                expected_principal=float(expected_principals[i]),
                expected_ending_balance=float(expected_ending_balances[i]),
                expected_total_cashflow=float(expected_total_cf),
                discount_factor=float(discount_factors[i]),
                expected_pv=float(expected_pv),
            )
        )

    fair_price = 100.0 * float(pv_total)

    return TrancheCashflowValue(
        attachment=float(attach),
        detachment=float(detach),
        coupon=float(coupon),
        wal=float(wal),
        tranche_el=el,
        tranche_ul=ul,
        fair_spread=fair_spread,
        el_spread=el_spread,
        rp_spread=rp_spread,
        discount_rate=float(discount_rate),
        fair_price=fair_price,
        cashflows=cashflows,
    )


def ce_value_bps(
    losses: np.ndarray,
    attach: float,
    detach: float,
    wal: float,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
) -> float:
    """Spread, in bps, that credit enhancement saves the tranche.

    This compares the actual tranche to a same-width tranche sitting at the
    bottom of the capital stack with no credit enhancement.

    The attachment point is the tranche's credit enhancement.
    """
    with_ce = value_tranche(
        losses=losses,
        attach=attach,
        detach=detach,
        coupon=0.0,
        wal=wal,
        risk_free=risk_free,
        price_of_risk=price_of_risk,
    ).fair_spread

    width = detach - attach

    no_ce = value_tranche(
        losses=losses,
        attach=0.0,
        detach=max(width, 1e-6),
        coupon=0.0,
        wal=wal,
        risk_free=risk_free,
        price_of_risk=price_of_risk,
    ).fair_spread

    return float((no_ce - with_ce) * 1e4)


def ce_value_bps_cashflows(
    losses: np.ndarray,
    attach: float,
    detach: float,
    coupon: float,
    principal_schedule: np.ndarray,
    period_lengths: np.ndarray | None = None,
    loss_timing: np.ndarray | None = None,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
    tail_q: float = 0.99,
    coupon_basis: CouponBasis = "beginning",
) -> float:
    """Spread, in bps, that credit enhancement saves under cashflow valuation.

    This compares the actual amortizing tranche to a same-width amortizing
    tranche sitting at the bottom of the capital stack.
    """
    with_ce = value_tranche_cashflows(
        losses=losses,
        attach=attach,
        detach=detach,
        coupon=coupon,
        principal_schedule=principal_schedule,
        period_lengths=period_lengths,
        loss_timing=loss_timing,
        risk_free=risk_free,
        price_of_risk=price_of_risk,
        tail_q=tail_q,
        coupon_basis=coupon_basis,
    ).fair_spread

    width = detach - attach

    no_ce = value_tranche_cashflows(
        losses=losses,
        attach=0.0,
        detach=max(width, 1e-6),
        coupon=coupon,
        principal_schedule=principal_schedule,
        period_lengths=period_lengths,
        loss_timing=loss_timing,
        risk_free=risk_free,
        price_of_risk=price_of_risk,
        tail_q=tail_q,
        coupon_basis=coupon_basis,
    ).fair_spread

    return float((no_ce - with_ce) * 1e4)


def value_surface(
    loss_sets: dict[str, np.ndarray],
    attach: float,
    detach: float,
    coupon: float,
    wal: float,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
    tail_q: float = 0.99,
) -> dict[str, float]:
    """Fair spread under several loss distributions.

    `loss_sets` maps label -> loss array. This returns label -> fair_spread, so
    valuation can be reported as a range across correlation/stress regimes rather
    than a single point estimate.
    """
    return {
        label: value_tranche(
            losses=losses,
            attach=attach,
            detach=detach,
            coupon=coupon,
            wal=wal,
            risk_free=risk_free,
            price_of_risk=price_of_risk,
            tail_q=tail_q,
        ).fair_spread
        for label, losses in loss_sets.items()
    }


def value_cashflow_surface(
    loss_sets: dict[str, np.ndarray],
    attach: float,
    detach: float,
    coupon: float,
    principal_schedule: np.ndarray,
    period_lengths: np.ndarray | None = None,
    loss_timing: np.ndarray | None = None,
    risk_free: float = 0.04,
    price_of_risk: float = 1.0,
    tail_q: float = 0.99,
    coupon_basis: CouponBasis = "beginning",
) -> dict[str, TrancheCashflowValue]:
    """Cashflow valuation under several loss distributions.

    `loss_sets` maps label -> loss array.

    Returns label -> TrancheCashflowValue, including:
    - fair spread
    - fair price
    - expected loss
    - unexpected/tail loss
    - full expected cashflow table
    """
    return {
        label: value_tranche_cashflows(
            losses=losses,
            attach=attach,
            detach=detach,
            coupon=coupon,
            principal_schedule=principal_schedule,
            period_lengths=period_lengths,
            loss_timing=loss_timing,
            risk_free=risk_free,
            price_of_risk=price_of_risk,
            tail_q=tail_q,
            coupon_basis=coupon_basis,
        )
        for label, losses in loss_sets.items()
    }


def cashflow_table(value: TrancheCashflowValue) -> list[dict]:
    """Return the projected expected cashflow table as list[dict]."""
    return [row.as_dict() for row in value.cashflows]


if __name__ == "__main__":
    # Minimal smoke test / example.
    rng = np.random.default_rng(7)

    # Toy simulated collateral losses.
    # Replace this with your Vasicek engine output.
    losses = np.clip(rng.beta(a=2.0, b=20.0, size=50_000), 0.0, 1.0)

    attach = 0.05
    detach = 0.15
    coupon = 0.08

    # Five-year quarterly amortization.
    n_periods = 20
    principal_schedule = np.full(n_periods, 1.0 / n_periods)
    period_lengths = np.full(n_periods, 0.25)

    # Example front-loaded loss timing.
    loss_timing = np.array(
        [
            0.12, 0.11, 0.10, 0.09,
            0.08, 0.07, 0.06, 0.05,
            0.045, 0.04, 0.035, 0.03,
            0.025, 0.02, 0.018, 0.015,
            0.012, 0.01, 0.007, 0.003,
        ],
        dtype=float,
    )
    loss_timing = loss_timing / loss_timing.sum()

    result = value_tranche_cashflows(
        losses=losses,
        attach=attach,
        detach=detach,
        coupon=coupon,
        principal_schedule=principal_schedule,
        period_lengths=period_lengths,
        loss_timing=loss_timing,
        risk_free=0.04,
        price_of_risk=1.0,
        tail_q=0.99,
        coupon_basis="beginning",
    )

    print("Fair price:", result.fair_price)
    print("Fair spread:", result.fair_spread)
    print("EL spread:", result.el_spread)
    print("Risk-premium spread:", result.rp_spread)
    print("Tranche EL:", result.tranche_el)
    print("Tranche UL:", result.tranche_ul)
    print("First cashflow row:", result.cashflows[0].as_dict())
