import json
from pathlib import Path

import numpy as np
import pytest

from packfolio.evaluate import (
    EvaluationMetadata,
    evaluate_episode,
    read_evaluation_rows_jsonl,
    read_step_traces_jsonl,
    write_evaluation_rows_jsonl,
    write_step_traces_jsonl,
)
from packfolio.types import (
    OBSERVATION_SIZE,
    Action,
    EvaluationRow,
    MarketRegime,
    StepInfo,
    TerminationReason,
)


FIXTURE_DIR = Path(__file__).with_name("fixtures")


def load_fixture(name: str) -> dict[str, object]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def metadata_from_fixture(fixture: dict[str, object]) -> EvaluationMetadata:
    row = fixture["evaluation_row"]
    return EvaluationMetadata(
        simulator_version=row["simulator_version"],
        policy_id=row["policy_id"],
        model_id=row["model_id"],
        training_seed=row["training_seed"],
        scenario_id=row["scenario_id"],
        scenario_seed=row["scenario_seed"],
        config_hash=row["config_hash"],
        git_commit=row["git_commit"],
    )


class FixturePolicy:
    def __init__(self, action: Action):
        self.action = action

    def choose_action(self, observation: np.ndarray) -> Action:
        assert observation.shape == (OBSERVATION_SIZE,)
        assert observation.dtype == np.float32
        return self.action


class SequencePolicy:
    def __init__(self, actions: list[Action]):
        self._actions = actions
        self._index = 0

    def choose_action(self, observation: np.ndarray) -> Action:
        action = self._actions[self._index]
        self._index += 1
        return action


class FrozenPolicy:
    policy_id = "FROZEN"

    def choose_action(self, observation: np.ndarray) -> Action:
        return Action.HOLD


class FakeEnv:
    def __init__(
        self,
        observation: np.ndarray,
        step_infos: list[StepInfo],
        *,
        truncated: bool = False,
        reward_overrides: list[float] | None = None,
        terminate_with_truncation: bool = False,
    ):
        self._initial_observation = observation
        self._step_infos = step_infos
        self._truncated = truncated
        self._reward_overrides = reward_overrides
        self._terminate_with_truncation = terminate_with_truncation
        self.reset_seed = None
        self.actions: list[Action] = []
        self._step_index = 0

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self._step_index = 0
        self.actions = []
        return self._initial_observation.copy(), {}

    def step(self, action: Action):
        self.actions.append(action)
        step = self._step_infos[self._step_index]
        reward = (
            self._reward_overrides[self._step_index]
            if self._reward_overrides is not None
            else step.reward
        )
        self._step_index += 1
        terminated = self._step_index == len(self._step_infos) and (
            not self._truncated or self._terminate_with_truncation
        )
        truncated = self._step_index == len(self._step_infos) and self._truncated
        return (
            self._initial_observation.copy(),
            reward,
            terminated,
            truncated,
            step.to_dict(),
        )


def make_observation() -> np.ndarray:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[0] = 1.0
    observation[4] = 1.0
    observation[6] = 1.0
    return observation


def make_step(**changes: object) -> StepInfo:
    values = {
        "step_index": 0,
        "requested_action": Action.HOLD,
        "executed_action": Action.HOLD,
        "action_was_infeasible": False,
        "regime_before": MarketRegime.NORMAL,
        "regime_after": MarketRegime.NORMAL,
        "cash_before": 10_000.0,
        "cash_after": 10_000.0,
        "sealed_count_before": 0,
        "sealed_count_after": 0,
        "pack_outcome_id": None,
        "gross_opened_value": None,
        "fee_paid": 0.0,
        "portfolio_value_before": 10_000.0,
        "portfolio_value_after": 10_000.0,
        "reward": 0.0,
        "termination_reason": TerminationReason.HORIZON,
    }
    values.update(changes)
    return StepInfo(**values)


def test_fixture_policy_completes_full_fake_episode():
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    env = FakeEnv(observation, [step])
    policy = FixturePolicy(Action.HOLD)

    result = evaluate_episode(policy, env, metadata_from_fixture(fixture))

    assert result.row == EvaluationRow.from_dict(fixture["evaluation_row"])
    assert len((result.row,)) == 1
    assert result.step_traces == (step,)
    assert env.reset_seed == fixture["evaluation_row"]["scenario_seed"]
    assert env.actions == [Action.HOLD]


def test_evaluate_episode_counts_actions_rewards_and_identity():
    observation = make_observation()
    steps = [
        make_step(
            step_index=0,
            requested_action=Action.BUY_PACK,
            executed_action=Action.HOLD,
            action_was_infeasible=True,
            portfolio_value_before=10_000.0,
            portfolio_value_after=10_010.0,
            reward=0.1,
            termination_reason=None,
        ),
        make_step(
            step_index=1,
            requested_action=Action.HOLD,
            executed_action=Action.HOLD,
            portfolio_value_before=10_010.0,
            portfolio_value_after=10_030.0,
            reward=0.2,
            termination_reason=TerminationReason.HORIZON,
        ),
    ]
    metadata = EvaluationMetadata(
        simulator_version="0.1.0",
        policy_id="fixture-policy",
        model_id="model-a",
        training_seed=7,
        scenario_id="scenario-a",
        scenario_seed=123,
        config_hash="c" * 64,
        git_commit="d" * 40,
    )
    env = FakeEnv(observation, steps)

    result = evaluate_episode(
        SequencePolicy([Action.BUY_PACK, Action.HOLD]),
        env,
        metadata,
    )
    row = result.row

    assert row.policy_id == "fixture-policy"
    assert row.model_id == "model-a"
    assert row.training_seed == 7
    assert row.scenario_id == "scenario-a"
    assert row.scenario_seed == 123
    assert row.episode_steps == 2
    assert row.requested_hold_count == 1
    assert row.requested_buy_pack_count == 1
    assert row.requested_open_and_sell_count == 0
    assert row.requested_sell_pack_count == 0
    assert row.executed_hold_count == 2
    assert row.infeasible_action_count == 1
    assert row.cumulative_reward == 0.30000000000000004
    assert row.initial_portfolio_value == 10_000.0
    assert row.final_portfolio_value == 10_030.0
    assert row.terminated is True
    assert row.truncated is False
    assert row.termination_reason is TerminationReason.HORIZON
    assert result.step_traces == tuple(steps)


def test_policy_receives_only_observation_and_evaluation_does_not_modify_policy():
    class ObservationOnlyPolicy:
        def __init__(self):
            self.seen_shape = None

        def choose_action(self, observation: np.ndarray) -> Action:
            self.seen_shape = observation.shape
            return Action.HOLD

        def fit(self, *_args, **_kwargs):
            raise AssertionError("evaluate_episode must not call fit")

        def update(self, *_args, **_kwargs):
            raise AssertionError("evaluate_episode must not call update")

        def train(self, *_args, **_kwargs):
            raise AssertionError("evaluate_episode must not call train")

        def fit_preprocessing(self, *_args, **_kwargs):
            raise AssertionError("evaluate_episode must not fit preprocessing")

    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    env = FakeEnv(observation, [step])
    policy = ObservationOnlyPolicy()

    evaluate_episode(policy, env, metadata_from_fixture(fixture))

    assert policy.seen_shape == (OBSERVATION_SIZE,)

    frozen_policy = FrozenPolicy()
    before = dict(frozen_policy.__dict__)
    evaluate_episode(frozen_policy, FakeEnv(observation, [step]), metadata_from_fixture(fixture))
    assert frozen_policy.__dict__ == before


def test_reward_mismatch_between_env_return_and_step_info_is_rejected():
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    env = FakeEnv(observation, [step], reward_overrides=[step.reward + 1.0])

    with pytest.raises(
        ValueError,
        match="environment reward and StepInfo.reward disagree",
    ):
        evaluate_episode(FixturePolicy(Action.HOLD), env, metadata_from_fixture(fixture))


def test_requested_action_mismatch_is_rejected():
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = make_step(
        requested_action=Action.BUY_PACK,
        executed_action=Action.HOLD,
        action_was_infeasible=True,
    )
    env = FakeEnv(observation, [step])

    with pytest.raises(
        ValueError,
        match="StepInfo.requested_action must match",
    ):
        evaluate_episode(FixturePolicy(Action.HOLD), env, metadata_from_fixture(fixture))


def test_truncated_episode_is_rejected_by_pf03_evaluation_row_schema():
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    env = FakeEnv(
        observation,
        [step],
        truncated=True,
        terminate_with_truncation=True,
    )

    with pytest.raises(ValueError, match="cannot be truncated"):
        evaluate_episode(FixturePolicy(Action.HOLD), env, metadata_from_fixture(fixture))


def test_jsonl_output_records_validate_with_pf03_schemas(tmp_path):
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    result = evaluate_episode(
        FixturePolicy(Action.HOLD),
        FakeEnv(observation, [step]),
        metadata_from_fixture(fixture),
    )
    rows_path = tmp_path / "rows.jsonl"
    traces_path = tmp_path / "traces.jsonl"

    write_evaluation_rows_jsonl(rows_path, [result.row])
    write_step_traces_jsonl(traces_path, result.step_traces)

    assert read_evaluation_rows_jsonl(rows_path) == (result.row,)
    assert read_step_traces_jsonl(traces_path) == result.step_traces


def test_multiple_rows_keep_model_seed_and_scenario_identity_distinct():
    fixture = load_fixture("contract_exchange_v2.json")
    observation = np.asarray(fixture["observation"], dtype=np.float32)
    step = StepInfo.from_dict(fixture["step_info"])
    metadata_a = metadata_from_fixture(fixture)
    metadata_b = EvaluationMetadata(
        simulator_version=metadata_a.simulator_version,
        policy_id=metadata_a.policy_id,
        model_id="checkpoint-b",
        training_seed=99,
        scenario_id="fixture-002",
        scenario_seed=456,
        config_hash=metadata_a.config_hash,
        git_commit=metadata_a.git_commit,
    )

    row_a = evaluate_episode(
        FixturePolicy(Action.HOLD),
        FakeEnv(observation, [step]),
        metadata_a,
    ).row
    row_b = evaluate_episode(
        FixturePolicy(Action.HOLD),
        FakeEnv(observation, [step]),
        metadata_b,
    ).row

    identities = {
        (row.model_id, row.training_seed, row.scenario_id, row.scenario_seed)
        for row in (row_a, row_b)
    }
    assert identities == {
        (None, None, "fixture-001", 123),
        ("checkpoint-b", 99, "fixture-002", 456),
    }
