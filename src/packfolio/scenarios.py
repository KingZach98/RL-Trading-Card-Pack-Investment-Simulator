"""Versioned scenario recipes and simulator-only random streams."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np

from packfolio.config import EnvironmentConfig, _integer, _json_record, load_environment_config
from packfolio.market import advance_market, snapshot_for
from packfolio.packs import PackDraw, sample_pack
from packfolio.types import MarketSnapshot

SIMULATOR_VERSION = "packfolio-scenarios-v1"
Split = Literal["training", "validation", "final_test"]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Replay recipe. Its identifier encodes the config hash and root seed."""

    config_hash: str
    seed: int
    simulator_version: str = SIMULATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.config_hash):
            raise ValueError("config_hash must be a lowercase SHA-256 digest")
        _integer(self.seed, "seed")
        if self.seed >= 2**64:
            raise ValueError("seed must be less than 2**64")
        if self.simulator_version != SIMULATOR_VERSION:
            raise ValueError(f"unsupported simulator_version: {self.simulator_version!r}")

    @property
    def scenario_id(self) -> str:
        return f"{self.config_hash}:{self.seed}"

    @classmethod
    def from_id(cls, scenario_id: str, simulator_version: str) -> "ScenarioSpec":
        if not isinstance(scenario_id, str):
            raise TypeError("scenario_id must be a string")
        match = re.fullmatch(r"([0-9a-f]{64}):(0|[1-9][0-9]*)", scenario_id)
        if match is None:
            raise ValueError("invalid scenario_id")
        return cls(match[1], int(match[2]), simulator_version)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    split: Split
    owner: str
    config: EnvironmentConfig
    scenarios: tuple[ScenarioSpec, ...]

    def __post_init__(self) -> None:
        if self.split not in ("training", "validation", "final_test"):
            raise ValueError("split must be training, validation, or final_test")
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("owner must be a nonempty string")
        if not isinstance(self.config, EnvironmentConfig):
            raise TypeError("config must be EnvironmentConfig")
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValueError("scenarios must be a nonempty tuple")
        if any(not isinstance(spec, ScenarioSpec) for spec in self.scenarios):
            raise TypeError("scenarios must contain ScenarioSpec objects")
        if any(spec.config_hash != self.config.config_hash for spec in self.scenarios):
            raise ValueError("scenario config_hash does not match environment configuration")
        if len({spec.seed for spec in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate seeds within split")


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load only the explicitly requested split; never auto-load final test."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        data = _json_record(
            json.load(file),
            {"split", "owner", "environment_config", "config_hash", "simulator_version", "seeds"},
            "split manifest",
        )
    split = data["split"]
    if split not in ("training", "validation", "final_test"):
        raise ValueError("invalid split")
    parsed_split: Split
    if split == "training":
        parsed_split = "training"
    elif split == "validation":
        parsed_split = "validation"
    else:
        parsed_split = "final_test"
    owner = data["owner"]
    config_path = data["environment_config"]
    config_hash = data["config_hash"]
    version = data["simulator_version"]
    if not isinstance(owner, str) or not isinstance(config_path, str):
        raise TypeError("owner and environment_config must be strings")
    if not isinstance(config_hash, str) or not isinstance(version, str):
        raise TypeError("config_hash and simulator_version must be strings")
    if not config_path.strip():
        raise ValueError("environment_config must be a nonempty path")
    seeds = data["seeds"]
    if not isinstance(seeds, list):
        raise TypeError("seeds must be a JSON array")
    config = load_environment_config(path.parent / config_path)
    specs = tuple(ScenarioSpec(config_hash, _integer(seed, "seed"), version) for seed in seeds)
    return SplitManifest(parsed_split, owner, config, specs)


def validate_split_manifests(*manifests: SplitManifest) -> None:
    """Require all three splits and disjoint seeds/identities across them."""
    if len(manifests) != 3 or {manifest.split for manifest in manifests} != {
        "training", "validation", "final_test"
    }:
        raise ValueError("provide exactly one training, validation, and final_test manifest")
    seeds: set[int] = set()
    identities: set[str] = set()
    for manifest in manifests:
        for spec in manifest.scenarios:
            if spec.seed in seeds or spec.scenario_id in identities:
                raise ValueError("scenario splits overlap")
            seeds.add(spec.seed)
            identities.add(spec.scenario_id)


def _generator(spec: ScenarioSpec, domain: str, *coordinates: int) -> np.random.Generator:
    payload = json.dumps(
        [spec.simulator_version, spec.scenario_id, domain, *coordinates], separators=(",", ":")
    )
    entropy = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
    return np.random.Generator(np.random.PCG64(entropy))


class Scenario:
    """Trusted simulator state, not a policy observation or a security boundary.

    Time starts at zero. Openings within a timestep consume sequential draw
    slots; advance resets the slot index. No API reveals future draws or RNGs.
    """

    __slots__ = ("__config", "__spec", "__market_rng", "__market", "__timestep", "__opening_index")

    def __init__(self, config: EnvironmentConfig, spec: ScenarioSpec) -> None:
        if not isinstance(config, EnvironmentConfig) or not isinstance(spec, ScenarioSpec):
            raise TypeError("Scenario requires EnvironmentConfig and ScenarioSpec")
        if config.config_hash != spec.config_hash:
            raise ValueError("scenario config_hash does not match environment configuration")
        self.__config = config
        self.__spec = spec
        self.__market_rng = _generator(spec, "market")
        self.__market = snapshot_for(config.market.initial_regime, config.market)
        self.__timestep = 0
        self.__opening_index = 0

    @property
    def timestep(self) -> int:
        return self.__timestep

    @property
    def current_market(self) -> MarketSnapshot:
        return self.__market

    @property
    def done(self) -> bool:
        return self.__timestep == self.__config.horizon

    def open_pack(self) -> PackDraw:
        """Reveal only the next opening slot at the current timestep."""
        if self.done:
            raise RuntimeError("scenario has reached its horizon")
        rng = _generator(self.__spec, "pack", self.__timestep, self.__opening_index)
        draw = sample_pack(self.__config.pack, self.__market.card_value_multiplier, rng)
        self.__opening_index += 1
        return draw

    def advance(self) -> MarketSnapshot:
        """Consume exactly one market transition, independent of openings."""
        if self.done:
            raise RuntimeError("scenario has reached its horizon")
        market = advance_market(self.__market.regime, self.__market_rng, self.__config.market)
        self.__market = market
        self.__timestep += 1
        self.__opening_index = 0
        return market


__all__ = [
    "SIMULATOR_VERSION", "Scenario", "ScenarioSpec", "SplitManifest",
    "load_split_manifest", "validate_split_manifests",
]