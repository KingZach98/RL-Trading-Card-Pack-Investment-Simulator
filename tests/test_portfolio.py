import pytest

from packfolio.portfolio import apply_action, liquidate, liquidation_value
from packfolio.types import Action, MarketRegime, MarketSnapshot, PackOutcome, PackOutcomeId, PortfolioSnapshot


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


@pytest.mark.parametrize("cash, sealed_count", [(0.0, 0), (100.0, 0), (100.0, 2)])
def test_hold_keeps_cash_and_inventory(cash, sealed_count, market):
    portfolio = PortfolioSnapshot(cash=cash, sealed_count=sealed_count)
    update = apply_action(
        portfolio, Action.HOLD, market, selling_fee=0.05, inventory_capacity=2,)
    assert update.portfolio == portfolio
    assert update.fee_paid == 0.0


@pytest.mark.parametrize("cash, sealed_count, expected_cash, expected_count", [(100.0, 0, 90.0, 1), (10.0, 0, 0.0, 1), (100.0, 1, 90.0, 2)],)
def test_buy_exchanges_cash_for_one_pack(cash, sealed_count, expected_cash, expected_count, market):
    portfolio = PortfolioSnapshot(cash=cash, sealed_count=sealed_count)
    update = apply_action( portfolio, Action.BUY_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert update.portfolio.cash == expected_cash
    assert update.portfolio.sealed_count == expected_count
    assert update.fee_paid == 0.0
    assert portfolio == PortfolioSnapshot(cash=cash, sealed_count=sealed_count)


def test_buy_does_not_allow_spending_more_than_available_cash(market):
    portfolio = PortfolioSnapshot(cash=9.99, sealed_count=0)
    with pytest.raises(ValueError, match="cash does not cover"):
        apply_action(portfolio, Action.BUY_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert portfolio == PortfolioSnapshot(cash=9.99, sealed_count=0)


def test_buy_rejects_full_inventory(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=2)
    with pytest.raises(ValueError, match="inventory is at capacity"):
        apply_action(portfolio, Action.BUY_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert portfolio == PortfolioSnapshot(cash=100.0, sealed_count=2)


def test_zero_price_buy_needs_no_cash():
    portfolio = PortfolioSnapshot(cash=0.0, sealed_count=0)
    market = MarketSnapshot(regime=MarketRegime.NORMAL, pack_ask=0.0, card_value_multiplier=1.0)
    update = apply_action(portfolio, Action.BUY_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert update.portfolio == PortfolioSnapshot(cash=0.0, sealed_count=1)
    assert update.fee_paid == 0.0


@pytest.mark.parametrize("sealed_count", [1, 2])
def test_sealed_sale_removes_one_pack_and_charges_one_fee(sealed_count, market):
    portfolio = PortfolioSnapshot(cash=90.0, sealed_count=sealed_count)
    update = apply_action(portfolio, Action.SELL_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert update.portfolio.cash == pytest.approx(99.50)
    assert update.portfolio.sealed_count == sealed_count - 1
    assert update.fee_paid == pytest.approx(0.50)
    assert portfolio == PortfolioSnapshot(cash=90.0, sealed_count=sealed_count)
    assert market == MarketSnapshot(regime=MarketRegime.NORMAL, pack_ask=10.0, card_value_multiplier=1.0)


def test_sealed_sale_requires_a_pack(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=0)
    with pytest.raises(ValueError, match="SELL_PACK requires a sealed pack"):
        apply_action(portfolio, Action.SELL_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert portfolio == PortfolioSnapshot(cash=100.0, sealed_count=0)


def test_buy_then_sell_loses_only_the_sale_fee(market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=0)
    bought = apply_action(portfolio, Action.BUY_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert liquidation_value(bought.portfolio, market, selling_fee=0.05) == pytest.approx(99.50)
    sold = apply_action(bought.portfolio, Action.SELL_PACK, market, selling_fee=0.05, inventory_capacity=2,)
    assert sold.portfolio.cash == pytest.approx(99.50)
    assert sold.portfolio.sealed_count == 0
    assert bought.fee_paid + sold.fee_paid == pytest.approx(0.50)
    assert portfolio == PortfolioSnapshot(cash=100.0, sealed_count=0)


@pytest.mark.parametrize("action", [True, 0, 1, 3, 4, "BUY_PACK"])
def test_apply_action_requires_the_action_enum(action, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=1)
    with pytest.raises(TypeError, match="action must be Action"):
        apply_action(portfolio, action, market, selling_fee=0.05, inventory_capacity=2,)


@pytest.mark.parametrize("capacity, expected_error",[(0, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError), ("2", TypeError)],)
def test_apply_action_rejects_invalid_capacity(capacity, expected_error, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=1)
    with pytest.raises(expected_error, match="inventory_capacity"):
        apply_action(portfolio, Action.HOLD, market, selling_fee=0.05, inventory_capacity=capacity,)


@pytest.mark.parametrize("action", [Action.HOLD, Action.BUY_PACK, Action.SELL_PACK])
def test_apply_action_rejects_inventory_above_capacity(action, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=3)
    with pytest.raises(ValueError, match="sealed_count exceeds inventory_capacity"):
        apply_action(portfolio, action, market, selling_fee=0.05, inventory_capacity=2,)


@pytest.mark.parametrize("action", [Action.HOLD, Action.BUY_PACK, Action.SELL_PACK])
def test_apply_action_checks_the_fee_even_when_no_sale_occurs(action, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=1)
    with pytest.raises(ValueError, match="selling_fee"):
        apply_action(portfolio, action, market, selling_fee=1.0, inventory_capacity=2,)


@pytest.mark.parametrize("action", [Action.HOLD, Action.BUY_PACK, Action.SELL_PACK])
def test_nonopening_actions_reject_pack_outcomes(action, market):
    portfolio = PortfolioSnapshot(cash=100.0, sealed_count=1)
    outcome = PackOutcome(outcome_id=PackOutcomeId.MEDIUM_VALUE, base_gross_value=20.0, gross_value=20.0)
    with pytest.raises(ValueError, match="pack_outcome is only allowed for OPEN_AND_SELL"):
        apply_action(portfolio, action, market,selling_fee=0.05, inventory_capacity=2, pack_outcome=outcome,)
