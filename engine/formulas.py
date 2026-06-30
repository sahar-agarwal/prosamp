"""Core CE metrics from the project proposal (Eqs. 1-23).

All functions are pure and take plain numbers, so they are trivially testable and
reused identically by every dashboard tab. Rates are fractions of the pool (0.05 =
5%), not percents.
"""
from __future__ import annotations


def expected_loss(pd_: float, lgd: float) -> float:
    """Eq. 1: Expected Loss = PD x LGD."""
    return pd_ * lgd


def ce_surplus_shortfall(initial_ce: float, realized_cum_loss: float) -> float:
    """Eq. 2: positive = CE held up; negative = CE was breached."""
    return initial_ce - realized_cum_loss


def ce_coverage_ratio(available_ce: float, expected_loss_: float) -> float | None:
    """Eq. 3: how many times expected loss is covered by CE."""
    return available_ce / expected_loss_ if expected_loss_ else None


def yield_sacrificed(yield_lower_rated: float, yield_higher_rated: float) -> float:
    """Eq. 4: extra yield given up to move up the stack."""
    return yield_lower_rated - yield_higher_rated


def incremental_ce(ce_higher_rated: float, ce_lower_rated: float) -> float:
    """Eq. 5: extra protection gained moving up the stack."""
    return ce_higher_rated - ce_lower_rated


def cost_of_incremental_protection(yield_sac: float, incr_ce: float) -> float | None:
    """Eq. 6: yield given up per unit of CE gained."""
    return yield_sac / incr_ce if incr_ce else None


def backtest_error(realized_loss: float, expected_loss_: float) -> float:
    """Eq. 21: positive = losses worse than expected at issuance."""
    return realized_loss - expected_loss_


def ce_adequacy(available_ce: float, realized_loss: float) -> float:
    """Eq. 22."""
    return available_ce - realized_loss


def ce_adequacy_ratio(available_ce: float, realized_loss: float) -> float | None:
    """Eq. 23."""
    return available_ce / realized_loss if realized_loss else None


def rating_cushion(available_ce: float, required_ce: float) -> float:
    """Eq. 19."""
    return available_ce - required_ce


def rating_cushion_ratio(available_ce: float, required_ce: float) -> float | None:
    """Eq. 20."""
    return available_ce / required_ce if required_ce else None


# --- Originator-confidence structural metrics (Eqs. 7-11) -------------------- #
def oc_cushion(actual_oc: float, required_oc: float) -> float:
    """Eq. 7."""
    return actual_oc - required_oc


def reserve_fund_ratio(reserve_balance: float, pool_balance: float) -> float | None:
    """Eq. 8."""
    return reserve_balance / pool_balance if pool_balance else None


def voluntary_ce_gap(actual_ce: float, required_ce: float) -> float:
    """Eq. 10: CE provided above the rating-agency minimum."""
    return actual_ce - required_ce


def voluntary_ce_gap_ratio(actual_ce: float, required_ce: float) -> float | None:
    """Eq. 11."""
    return voluntary_ce_gap(actual_ce, required_ce) / required_ce if required_ce else None
