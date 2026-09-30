import importlib

import packfolio


def test_package_imports():
    assert packfolio.__name__ == "packfolio"
    for module in ("config", "env", "market", "packs", "portfolio", "agents.dqn_agent"):
        assert importlib.import_module(f"packfolio.{module}") is not None