from math import isfinite
from numbers import Real

from packfolio.types import MarketSnapshot, PortfolioSnapshot, PortfolioUpdate


def _sale_amounts(gross_value: float, selling_fee: float) -> tuple[float, float]:
    if isinstance(selling_fee, bool) or not isinstance(selling_fee, Real):
        raise TypeError("selling_fee must be a number")
    fee_rate = float(selling_fee)
    if not isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("selling_fee must be finite and in [0, 1)")
    if not isfinite(gross_value):
        raise ValueError("gross sale value must be finite")

    proceeds = gross_value * (1 - fee_rate)
    fee_paid = gross_value * fee_rate
    return proceeds, fee_paid


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

__all__ = ["liquidate", "liquidation_value"]
