import numpy as np

from packfolio.agents.dqn_agent import (
    DEFAULT_HIDDEN_LAYER_WIDTHS,
    HIDDEN_LAYER_WIDTHS,
    DQNAgent,
    preprocess_observations,
)
from packfolio.types import Action


def test_original_hidden_layer_constant_remains_available():
    assert HIDDEN_LAYER_WIDTHS == DEFAULT_HIDDEN_LAYER_WIDTHS == (64, 64)


def test_training_and_evaluation_scaling_preserves_profitable_cash():
    observation = np.array([25, 0, 1, 1, 1, 0, 1, 0], dtype=np.float32)
    single = preprocess_observations(observation)
    batch = preprocess_observations(observation[None, :])
    np.testing.assert_array_equal(single, batch[0])
    assert single[0] == 2.5


def test_original_adapter_api_returns_deterministic_action():
    agent = DQNAgent(lambda _: np.array([[0, 1, 4, 2]], dtype=np.float32))
    observation = np.array([1, 0, 1, 1, 1, 0, 1, 0], dtype=np.float32)
    assert agent.choose_action(observation) is Action.OPEN_AND_SELL
    assert agent.choose_action(observation) is Action.OPEN_AND_SELL
