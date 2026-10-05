from math import isfinite
from numbers import Integral, Real

from packfolio.types import Action, MarketSnapshot, PackOutcome, PortfolioSnapshot, PortfolioUpdate


def _check_selling_fee(selling_fee: float) -> float:
    if isinstance(selling_fee, bool) or not isinstance(selling_fee, Real):
        raise TypeError("selling_fee must be a number")
    fee_rate = float(selling_fee)
    if not isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("selling_fee must be finite and in [0, 1)")
    return fee_rate


def _sale_amounts(gross_value: float, selling_fee: float) -> tuple[float, float]:
    fee_rate = _check_selling_fee(selling_fee)
    if not isfinite(gross_value):
        raise ValueError("gross sale value must be finite")
    proceeds = gross_value * (1 - fee_rate)
    fee_paid = gross_value * fee_rate
    return proceeds, fee_paid


def apply_action(portfolio: PortfolioSnapshot, action: Action, market: MarketSnapshot, *, selling_fee: float, inventory_capacity: int, pack_outcome: PackOutcome | None = None,
) -> PortfolioUpdate:
    if not isinstance(action, Action):
        raise TypeError("action must be Action")
    if isinstance(inventory_capacity, bool) or not isinstance(inventory_capacity, Integral):
        raise TypeError("inventory_capacity must be an integer")
    if inventory_capacity <= 0:
        raise ValueError("inventory_capacity must be positive")
    if portfolio.sealed_count > inventory_capacity:
        raise ValueError("sealed_count exceeds inventory_capacity")
    fee_rate = _check_selling_fee(selling_fee)
    if action is Action.OPEN_AND_SELL:
        raise NotImplementedError("OPEN_AND_SELL is not implemented yet")
    if pack_outcome is not None:
        raise ValueError("pack_outcome is only allowed for OPEN_AND_SELL")

    cash = portfolio.cash
    sealed_count = portfolio.sealed_count
    fee_paid = 0.0

    if action is Action.BUY_PACK:
        if cash < market.pack_ask:
            raise ValueError("cash does not cover the pack ask")
        if sealed_count == inventory_capacity:
            raise ValueError("inventory is at capacity")
        cash -= market.pack_ask
        sealed_count += 1
    elif action is Action.SELL_PACK:
        if sealed_count == 0:
            raise ValueError("SELL_PACK requires a sealed pack")
        proceeds, fee_paid = _sale_amounts(market.pack_ask, fee_rate)
        cash += proceeds
        sealed_count -= 1
    updated = PortfolioSnapshot(cash=cash, sealed_count=sealed_count)
    return PortfolioUpdate(portfolio=updated, fee_paid=fee_paid)


def liquidation_value(portfolio: PortfolioSnapshot, market: MarketSnapshot, *, selling_fee: float,) -> float:
    gross_value = portfolio.sealed_count * market.pack_ask
    proceeds, _ = _sale_amounts(gross_value, selling_fee)
    value = portfolio.cash + proceeds
    if not isfinite(value):
        raise ValueError("portfolio value must be finite")
    return value


def liquidate(portfolio: PortfolioSnapshot, market: MarketSnapshot, *, selling_fee: float,) -> PortfolioUpdate:
    gross_value = portfolio.sealed_count * market.pack_ask
    proceeds, fee_paid = _sale_amounts(gross_value, selling_fee)
    updated = PortfolioSnapshot(
        cash=portfolio.cash + proceeds,
        sealed_count=0,
    )
    return PortfolioUpdate(portfolio=updated, fee_paid=fee_paid)

__all__ = ["apply_action", "liquidate", "liquidation_value"]
