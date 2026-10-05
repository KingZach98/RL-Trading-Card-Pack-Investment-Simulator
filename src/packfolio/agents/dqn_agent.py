"""DQN network and deterministic policy adapter for Packfolio."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


ACTION_COUNT = len(Action)
DEFAULT_HIDDEN_LAYER_WIDTHS = (64, 64)
HIDDEN_LAYER_WIDTHS = DEFAULT_HIDDEN_LAYER_WIDTHS
FEATURE_SCALE: NDArray[np.float32] = np.asarray(
    (0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32
)
FEATURE_SCALE.setflags(write=False)
FEATURE_ORDER = tuple(index.name for index in ObservationIndex)
ACTION_ORDER = tuple(action.name for action in Action)


class QNetwork(Protocol):
    def __call__(self, observations: object, /) -> Any: ...


def preprocess_observations(
    observations: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply fixed feature scaling shared by training and evaluation.

    Cash is already expressed as a ratio of initial cash. Scaling it by ten
    keeps typical values moderate without clipping higher, profitable states.
    """
    if not isinstance(observations, np.ndarray):
        raise TypeError("observations must be a NumPy array")
    if observations.dtype != np.float32:
        raise TypeError("observations must use float32")
    if observations.ndim not in (1, 2) or observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(
            f"observations must have shape ({OBSERVATION_SIZE},) or "
            f"(batch, {OBSERVATION_SIZE})"
        )
    if observations.ndim == 2 and observations.shape[0] == 0:
        raise ValueError("observation batch must not be empty")
    if not np.isfinite(observations).all():
        raise ValueError("observations must contain only finite values")
    return observations * FEATURE_SCALE


def build_q_network(
    hidden_layer_widths: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_WIDTHS,
) -> QNetwork:
    """Build a compact Q network using the optional PyTorch extra."""
    if (
        not isinstance(hidden_layer_widths, tuple)
        or not hidden_layer_widths
        or any(type(width) is not int or width < 1 for width in hidden_layer_widths)
    ):
        raise ValueError("hidden_layer_widths must be a nonempty tuple of positive integers")
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError(
            "DQN training requires PyTorch; install Packfolio with its 'agent' extra."
        ) from error

    layers: list[Any] = []
    input_width = OBSERVATION_SIZE
    for width in hidden_layer_widths:
        layers.extend((torch.nn.Linear(input_width, width), torch.nn.ReLU()))
        input_width = width
    layers.append(torch.nn.Linear(input_width, ACTION_COUNT))
    return torch.nn.Sequential(*layers)


class _TorchQFunction:
    def __init__(self, network: QNetwork) -> None:
        try:
            self._torch = import_module("torch")
        except ImportError as error:
            raise ImportError(
                "A PyTorch-backed DQN policy requires PyTorch; install "
                "Packfolio with its 'agent' extra."
            ) from error
        self._network = network

    def __call__(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        torch = self._torch
        with torch.inference_mode():
            values = self._network(torch.as_tensor(observations, dtype=torch.float32))
        return values.detach().cpu().numpy()


class DQNAgent:
    """Deterministic evaluation adapter for a four-output Q function.

    The Q function accepts a preprocessed ``(batch, 8)`` float32 array and
    returns one Q-value per action in ID order. Use :meth:`from_torch` for a
    PyTorch network built or trained by the caller.
    """

    def __init__(self, q_function: Callable[[NDArray[np.float32]], object]) -> None:
        if not callable(q_function):
            raise TypeError("q_function must be callable")
        self._q_function = q_function

    @classmethod
    def from_torch(cls, network: QNetwork) -> DQNAgent:
        """Create an inference adapter for a PyTorch Q network."""
        return cls(_TorchQFunction(network))

    def choose_action(self, observation: NDArray[np.float32]) -> Action:
        features = preprocess_observations(observation)
        if features.ndim != 1:
            raise ValueError("choose_action expects one observation, not a batch")
        q_values = np.asarray(self._q_function(features[np.newaxis, :]))
        if q_values.shape != (1, ACTION_COUNT):
            raise ValueError(
                f"q_function must return shape (1, {ACTION_COUNT}), got {q_values.shape}"
            )
        if not np.isfinite(q_values).all():
            raise ValueError("q_function returned non-finite Q-values")
        return Action(int(np.argmax(q_values[0])))


def load_trained_agent(checkpoint_path: str | Path) -> DQNAgent:
    """Load a saved Q network using the checkpoint's architecture metadata."""
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError(
            "Loading a DQN checkpoint requires PyTorch; install Packfolio with "
            "its 'agent' extra."
        ) from error

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported or malformed DQN checkpoint")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint is missing architecture metadata")
    if architecture.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("checkpoint observation size does not match the interface")
    if architecture.get("action_count") != ACTION_COUNT:
        raise ValueError("checkpoint action count does not match the interface")
    if tuple(architecture.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("checkpoint feature order does not match the interface")
    if tuple(architecture.get("action_order", ())) != ACTION_ORDER:
        raise ValueError("checkpoint action order does not match the interface")
    if tuple(architecture.get("feature_scale", ())) != tuple(float(x) for x in FEATURE_SCALE):
        raise ValueError("checkpoint feature scaling does not match this agent")

    hidden_widths = architecture.get("hidden_layer_widths")
    if not isinstance(hidden_widths, list) or not hidden_widths:
        raise ValueError("checkpoint has invalid hidden layer widths")
    network = build_q_network(tuple(hidden_widths))
    state_dict = checkpoint.get("q_network_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing Q-network weights")
    network.load_state_dict(state_dict)
    network.eval()
    return DQNAgent.from_torch(network)


__all__ = [
    "ACTION_COUNT",
    "ACTION_ORDER",
    "DEFAULT_HIDDEN_LAYER_WIDTHS",
    "HIDDEN_LAYER_WIDTHS",
    "FEATURE_ORDER",
    "FEATURE_SCALE",
    "DQNAgent",
    "QNetwork",
    "build_q_network",
    "load_trained_agent",
    "preprocess_observations",
]