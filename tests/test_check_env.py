from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest

import packfolio.check_env as checks
from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
    return load_environment_config(path)


@pytest.mark.parametrize("horizon", [1, 6, 100])
def test_pytorch_can_run_and_replay_an_episode(config, horizon):
    torch = pytest.importorskip("torch")
    config = replace(config, horizon=horizon)
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    random_state = torch.get_rng_state().clone()
    assert checks.check_pytorch(env, horizon=horizon) == horizon
    assert torch.equal(torch.get_rng_state(), random_state)


def test_pytorch_check_rejects_the_wrong_observation_shape(config, monkeypatch):
    pytest.importorskip("torch")
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    original_reset = env.reset

    def wrong_shape(**kwargs):
        observation, info = original_reset(**kwargs)
        return np.append(observation, np.float32(0)), info

    monkeypatch.setattr(env, "reset", wrong_shape)
    with pytest.raises(AssertionError, match="observation shape"):
        checks.check_pytorch(env, horizon=config.horizon)


def test_command_fails_with_setup_help_when_pytorch_is_missing(monkeypatch, capsys):
    def missing_pytorch(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(checks, "check_pytorch", missing_pytorch)
    assert checks.main([]) == 1
    output = capsys.readouterr()
    assert "PyTorch is not installed" in output.err
    assert "--locked --extra dev --extra agent" in output.err
    assert "checks passed" not in output.out


def test_compatibility_command_reports_both_checks(capsys, recwarn):
    pytest.importorskip("torch")
    assert checks.main([]) == 0
    output = capsys.readouterr().out
    assert "Gymnasium and PyTorch boundary checks passed (100 steps, replay matched)" in output
    assert "not DQN training or agent quality" in output
    assert not recwarn


def test_command_keeps_other_checker_warnings_visible(monkeypatch):
    def warn(*args, **kwargs):
        warnings.warn("another environment warning", UserWarning)

    monkeypatch.setattr(checks, "check_env", warn)
    monkeypatch.setattr(checks, "check_pytorch", lambda *args, **kwargs: 100)
    with pytest.warns(UserWarning, match="another environment warning"):
        assert checks.main([]) == 0
