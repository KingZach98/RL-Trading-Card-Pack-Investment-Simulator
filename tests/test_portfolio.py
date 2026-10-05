import pytest

from packfolio.portfolio import liquidate, liquidation_value
from packfolio.types import MarketRegime, MarketSnapshot, PortfolioSnapshot


@pytest.fixture
def market():
    return MarketSnapshot(regime=MarketRegime.NORMAL, pack_ask=10.0, card_value_multiplier=1.0,)


def test_liquidation_value_matches_hand_calculation(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    value = liquidation_value(portfolio, market, selling_fee=0.05)
    assert value == pytest.approx(128.50)
    assert portfolio == PortfolioSnapshot(cash=100.0, sealed_count=3)


def test_liquidate_sells_all_packs_and_charges_one_fee(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    update = liquidate(portfolio, market, selling_fee=0.05)
    assert update.portfolio.cash == pytest.approx(128.50)
    assert update.portfolio.sealed_count == 0
    assert update.fee_paid == pytest.approx(1.50)
    assert portfolio == PortfolioSnapshot(cash=100.0, sealed_count=3)
    assert market == MarketSnapshot(regime=MarketRegime.NORMAL, pack_ask=10.0,card_value_multiplier=1.0,)


@pytest.mark.parametrize( "pack_ask, expected_cash, expected_fee", [(8.0, 122.80, 1.20), (10.0, 128.50, 1.50), (12.0, 134.20, 1.80)],)
def test_value_equals_cash_after_liquidation(pack_ask, expected_cash, expected_fee):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    market = MarketSnapshot(regime=MarketRegime.NORMAL,pack_ask=pack_ask,card_value_multiplier=2.0,)
    value = liquidation_value(portfolio, market, selling_fee=0.05)
    update = liquidate(portfolio, market, selling_fee=0.05)
    assert value == pytest.approx(expected_cash)
    assert update.portfolio.cash == pytest.approx(expected_cash)
    assert update.fee_paid == pytest.approx(expected_fee)
    assert update.portfolio.cash == pytest.approx(value)
    assert liquidation_value(update.portfolio, market, selling_fee=0.05) == pytest.approx(value)


def test_empty_inventory_keeps_cash_and_pays_no_fee(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=0)
    value = liquidation_value(portfolio, market, selling_fee=0.05)
    update = liquidate(portfolio, market, selling_fee=0.05)
    assert value == 100.0
    assert update.portfolio == portfolio
    assert update.fee_paid == 0.0


def test_liquidating_twice_does_not_charge_another_fee(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    first = liquidate(portfolio, market, selling_fee=0.05)
    second = liquidate(first.portfolio, market, selling_fee=0.05)
    assert second.portfolio == first.portfolio
    assert second.fee_paid == 0.0


def test_zero_fee_returns_the_full_sale_value(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    value = liquidation_value(portfolio, market, selling_fee=0.0)
    update = liquidate(portfolio, market, selling_fee=0.0)
    assert value == 130.0
    assert update.portfolio.cash == 130.0
    assert update.fee_paid == 0.0


@pytest.mark.parametrize("operation", [liquidation_value, liquidate])
@pytest.mark.parametrize("selling_fee",[-0.01, 1.0, 1.01, float("nan"), float("inf"), float("-inf")],)
def test_invalid_fee_rates_are_rejected(operation, selling_fee, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    with pytest.raises(ValueError, match="selling_fee"):
        operation(portfolio, market, selling_fee=selling_fee)


@pytest.mark.parametrize("operation", [liquidation_value, liquidate])
@pytest.mark.parametrize("selling_fee", [True, "0.05", None])
def test_wrong_fee_types_are_rejected(operation, selling_fee, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)

    with pytest.raises(TypeError, match="selling_fee must be a number"):
        operation(portfolio, market, selling_fee=selling_fee)


@pytest.mark.parametrize("operation", [liquidation_value, liquidate])
@pytest.mark.parametrize("cash, sealed_count", [(0.0, 2), (1e308, 1)])
def test_accounting_overflow_is_rejected(operation, cash, sealed_count):
    portfolio = PortfolioSnapshot(cash=cash, sealed_count=sealed_count)
    market = MarketSnapshot(regime=MarketRegime.NORMAL,pack_ask=1e308, card_value_multiplier=1.0,)
    with pytest.raises(ValueError, match="finite"):
        operation(portfolio, market, selling_fee=0.05)
