# EV_ONE_STEP (PF-13)

EV_ONE_STEP is a **model-informed**, deterministic, myopic comparator. It knows
the published transition probabilities, whole-pack distribution, selling fee,
and observation normalization. It is not an uninformed learner or a claim of
optimality. The implementation follows Project Bible v0.1, pages 13 and 26.

## Equations

For cash C, sealed inventory N, current regime m, ask B(m), and selling fee f:

```text
b  = (1 - f) B(m)
x  = (1 - f) expected_gross_value(pack_config, g(m))
mu = sum_j P(m,j) (1 - f) B(j)

HOLD: C + N mu
BUY:  C - B(m) + (N + 1) mu
SELL: C + b + (N - 1) mu
OPEN: C + x + (N - 1) mu
```

Score only observationally eligible actions. HOLD is always eligible; SELL and
OPEN require owned inventory; BUY requires room and PF-12 affordability.
Select the largest score, breaking exact ties in this order:
**HOLD, SELL_PACK, OPEN_AND_SELL, BUY_PACK**.

Selection compares the equivalent advantages over HOLD: 0, b-mu, x-mu,
mu-B(m), respectively. This cancels the shared C+N*mu and avoids losing small
action differences when cash is large. No arbitrary score tolerance is used.
`score_actions()` still exposes the absolute equations above.

Buying and opening are separate decisions. No lookahead opening bonus, reward
discount, historical purchase-cost subtraction, or terminal bonus is added.
The equations are unchanged on the final decision: the environment first
advances the market and then liquidates remaining inventory at that net quote.

## Information Boundary and Precision

Construct `EVOneStepPolicy` with the same immutable `EnvironmentConfig`, capacity,
and reference price as its environment. Capacity defaults to the shared value
10; reference price defaults to the configured NORMAL ask, matching
`build_environment()`. `choose_action(observation) -> Action` reads only the
public eight-field float32 observation. Regime quotes and multipliers are
deterministic published model values; inconsistent observation features fail
clearly instead of silently combining different configurations.

Analytical pack EV uses the sampler's normalized probabilities. Policy scoring
never calls the sampler, advances a market, reads a scenario/seed/StepInfo,
consumes a random stream, or changes portfolio/environment state. Configuration
and precomputed expectations are immutable. Selling fees are applied once.
The policy rejects a 100% selling fee locally because the existing environment
and accounting require f in [0,1); shared configuration validation is unchanged.

Cash is decoded from its rounded ratio; inventory is reconstructed as an integer
and checked against its normalized float32 representation. Current quotes and
opening multipliers come from static configuration selected by the observed
one-hot regime. Normalization settings must match the environment.

BUY eligibility directly reuses PF-12's `_can_buy()` rounding-interval convention,
without changing PF-12. Independently rounded cash and ask features may have
overlapping intervals, treated as affordable. Distinct affordable/unaffordable
float64 ledger states can have identical float32 observations. Consequently an
eligible BUY may still be rejected by the strict environment and recorded as
HOLD/infeasible. This limitation remains unresolved for team review; it is not
a guarantee of exact ledger feasibility or hidden-state access. Absolute scores
also reflect float32-decoded cash rather than recovering unknowable exact cash.
Very large capacities may similarly make distinct counts indistinguishable;
the current capacity-10 contract does not encounter that ambiguity.

## Evaluation and Classification

The policy works directly with the existing `evaluate_episode()` runner. To
export explicitly classified evidence, `evaluate_ev_episode()` calls that same
runner and its existing JSONL writers. It creates:

- `evaluation_rows.jsonl`: unchanged shared `EvaluationRow` schema.
- `step_traces.jsonl`: unchanged shared `StepInfo` schema.
- `policy_metadata.json`: PF-13-owned companion metadata, version 1, linking
  canonical policy ID `EV_ONE_STEP` to information class `model-informed`, the
  scenario/configuration, normalization, tie order, feasibility convention,
  and output filenames.

Metadata must identify EV_ONE_STEP and the same configuration; model ID and
training seed must be absent. The adapter records evaluator-only identity
outside the policy and never passes it to `choose_action()`. Each output
directory represents one episode; use distinct directories for multiple
scenarios. No shared schema, DQN checkpoint, simulator version, configuration
hash recipe, scenario identity, or random-stream recipe is changed.

```python
from packfolio.env import build_environment
from packfolio.ev_baseline import EVOneStepPolicy, evaluate_ev_episode

# config and metadata are the caller's existing EnvironmentConfig and
# EvaluationMetadata; metadata.policy_id must be "EV_ONE_STEP".
policy = EVOneStepPolicy(config)
result = evaluate_ev_episode(
    policy, build_environment(config), metadata, output_directory="outputs/ev/scenario-1"
)
```

## Verification

`tests/test_ev_baseline.py` includes independent hand calculations for the
Bible reference distribution and current configured distribution, one-fee
accounting, ties, eligibility, normalization, precision, and sampling guards.
`tests/test_ev_baseline_integration.py` checks real environment traces, final
liquidation, weighted enumeration of actual final-step outcomes for all four
actions, no stream/state disturbance, and classified output round trips through
the existing shared readers. Known precision-limit tests explicitly preserve
the unresolved behavior instead of treating successful tests as its resolution.

The shared horizon remains 100. Whether the group selects 52 or 100 is pending
approval and is not a PF-13 change. Current pack values and probabilities are
used as configured; the Bible's reference fixture is used only in analytical tests.
