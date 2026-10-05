

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# a .ipynb is JSON, not a module, so exec the code cells into a namespace and
# fake it as statarb_nb.pairs etc. cells tagged # skip-on-import are the ones
# that download data or plot -- skipping them keeps import cheap.
#
# The fake package used to be called "src", which was fine while this sleeve
# was its own repository. It is not fine here: the allocator in src/ is a real
# package, and registering a fake one under the same name in sys.modules broke
# every allocator test that happened to run after this file. Renaming it is the
# fix; the collision only existed because two projects with the same directory
# conventions got merged, and it failed loudly rather than quietly, which was
# luck.

NOTEBOOK = (Path(__file__).resolve().parents[2]
            / "sleeves" / "statarb" / "notebook" / "pairs.ipynb")
SKIP_MARKER = "# skip-on-import"


def _run_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "pairs_notebook"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if SKIP_MARKER in source:
            continue
        exec(compile(source, str(path), "exec"), namespace)

    return namespace


def _register_as_modules(namespace: dict, names: tuple[str, ...]) -> None:
    package = types.ModuleType("statarb_nb")
    package.__path__ = []  # marks it a package so `statarb_nb.pairs` resolves
    sys.modules["statarb_nb"] = package

    for name in names:
        module = types.ModuleType(f"statarb_nb.{name}")
        module.__dict__.update(namespace)
        sys.modules[f"statarb_nb.{name}"] = module
        setattr(package, name, module)


if not NOTEBOOK.exists():
    raise FileNotFoundError(f"cannot find the notebook at {NOTEBOOK}")

_register_as_modules(_run_notebook(NOTEBOOK), ("pairs", "signals", "backtest", "data"))


N = 1500
SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def cointegrated_pair(rng):
    # x is a random walk, y = 10 + 2.5x + AR(1) noise -> spread mean-reverts
    idx = pd.bdate_range("2018-01-01", periods=N)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, N)), index=idx, name="x")

    noise = np.zeros(N)
    for t in range(1, N):  # AR(1), phi=0.85 -> stationary
        noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 1)
    y = pd.Series(10 + 2.5 * x.values + noise, index=idx, name="y")
    return y, x


@pytest.fixture
def independent_walks(rng):
    # two unrelated walks, should NOT come back cointegrated
    idx = pd.bdate_range("2018-01-01", periods=N)
    a = pd.Series(100 + np.cumsum(rng.normal(0, 1, N)), index=idx, name="a")
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, N)), index=idx, name="b")
    return a, b
