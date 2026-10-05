import json
from pathlib import Path

import numpy as np
import pytest

from packfolio.agents.dqn_agent import (
    ACTION_COUNT,
    FEATURE_SCALE,
    FEATURE_ORDER,
    HIDDEN_LAYER_WIDTHS,
    DQNAgent,
    preprocess_observations,
)
from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "contract_exchange_v1.json"
)


def load_contract_observation() -> np.ndarray:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    return np.asarray(fixture["observation"], dtype=np.float32)


def test_preprocessing_uses_fixed_scales_and_preserves_profitable_cash():
    observation = load_contract_observation()
    profitable = observation.copy()
    profitable[0] = 25.0

    processed = preprocess_observations(profitable)

    assert processed.shape == (OBSERVATION_SIZE,)
    assert processed.dtype == np.float32
    assert processed[0] == pytest.approx(2.5)
    assert processed[0] > preprocess_observations(observation)[0]
    assert FEATURE_SCALE[0] == pytest.approx(0.1)
    assert not np.shares_memory(processed, profitable)


def test_training_batch_and_single_observation_use_identical_preprocessing():
    observation = load_contract_observation()
    batch = np.stack((observation, observation * 2.0)).astype(np.float32)

    processed_batch = preprocess_observations(batch)
    processed_single = preprocess_observations(observation)

    np.testing.assert_array_equal(processed_batch[0], processed_single)
    np.testing.assert_array_equal(processed_batch[1], processed_single * 2.0)


def test_policy_uses_contract_fixture_and_returns_an_action_id():
    seen: list[np.ndarray] = []

    def q_function(features: np.ndarray) -> np.ndarray:
        seen.append(features.copy())
        return np.asarray([[0.0, 1.0, 4.0, 2.0]], dtype=np.float32)

    agent = DQNAgent(q_function)
    action = agent.choose_action(load_contract_observation())

    assert action is Action.OPEN_AND_SELL
    assert int(action) in range(ACTION_COUNT)
    assert seen[0].shape == (1, OBSERVATION_SIZE)
    assert seen[0].dtype == np.float32


def test_policy_breaks_q_value_ties_by_lowest_action_id():
    agent = DQNAgent(lambda _: np.ones((1, ACTION_COUNT), dtype=np.float32))

    assert agent.choose_action(load_contract_observation()) is Action.HOLD


@pytest.mark.parametrize(
    "observation, error, message",
    [
        (np.zeros(7, dtype=np.float32), ValueError, "shape"),
        (np.zeros(8, dtype=np.float64), TypeError, "float32"),
        (np.full(8, np.nan, dtype=np.float32), ValueError, "finite"),
        (np.zeros((0, 8), dtype=np.float32), ValueError, "empty"),
    ],
)
def test_preprocessing_rejects_invalid_observations(observation, error, message):
    with pytest.raises(error, match=message):
        preprocess_observations(observation)


def test_action_adapter_rejects_batch_input_and_invalid_q_values():
    observation = load_contract_observation()
    agent = DQNAgent(lambda _: np.zeros((1, ACTION_COUNT), dtype=np.float32))
    with pytest.raises(ValueError, match="one observation"):
        agent.choose_action(np.stack((observation, observation)))

    wrong_shape = DQNAgent(lambda _: np.zeros((ACTION_COUNT,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        wrong_shape.choose_action(observation)

    non_finite = DQNAgent(
        lambda _: np.asarray([[0.0, 1.0, np.inf, 2.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="non-finite"):
        non_finite.choose_action(observation)


def test_network_configuration_is_compact_for_eight_inputs_and_four_actions():
    assert FEATURE_ORDER == tuple(index.name for index in ObservationIndex)
    assert HIDDEN_LAYER_WIDTHS == (64, 64)
    assert ACTION_COUNT == 4
