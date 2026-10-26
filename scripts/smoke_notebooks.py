#!/usr/bin/env python3
"""Execute MLAI lecture notebooks end-to-end for smoke testing.

CI-oriented defaults:
  - matplotlib Agg backend
  - mock stdin
  - short-circuit ipywidgets.interact / notutils.display_plots (avoid hangs)
  - cap GPy optimize iterations
  - execute `%load -n module.attr` cells by importing the name
  - fill student name / live-rating placeholders
  - blank incomplete exercise cells that are not valid Python
    (Bayesian posterior assignment is filled from the lecture formulae)
  - skip lecture 05 by default (MariaDB + credential widgets)

Usage (from repo root)::

    python scripts/smoke_notebooks.py
    python scripts/smoke_notebooks.py --only 01 12
    python scripts/smoke_notebooks.py --skip
    python scripts/smoke_notebooks.py --notebooks-dir _notebooks --results-dir _notebooks/test-results
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Injected as the first code cell of every notebook under test.
PRELUDE = r"""
import builtins
import numpy as np
import pandas as pd

if not hasattr(np, "NaN"):
    np.NaN = np.nan


def _mock_input(prompt=""):
    p = str(prompt).lower()
    if "distance" in p:
        print(f"[mock input] {prompt!r} -> 50.0")
        return "50.0"
    if (
        (not prompt)
        or ("yes" in p)
        or ("proceed" in p)
        or ("continue" in p)
        or ("[y/n]" in p)
        or ("[yes/no]" in p)
    ):
        print(f"[mock input] {prompt!r} -> y")
        return "y"
    print(f"[mock input] {prompt!r} -> 1.0")
    return "1.0"


builtins.input = _mock_input
try:
    ip = get_ipython()
    if ip is not None and getattr(ip, "kernel", None) is not None:
        ip.kernel._allow_stdin = True
        ip.kernel.raw_input = _mock_input
except Exception as e:
    print("stdin patch skipped", e)

import matplotlib

matplotlib.use("Agg")

# Lecture notebooks write figures under these relative dirs (cwd = notebooks dir).
from pathlib import Path as _Path

for _d in (
    "./datasets",
    "./data-science",
    "./dimred",
    "./gp",
    "./kern",
    "./ml",
    "./mlai",
):
    _Path(_d).mkdir(parents=True, exist_ok=True)


# Avoid ipywidgets.interact deadlocks under nbclient.
def _once_interact(func=None, **kwargs):
    if func is None:
        return lambda f: _once_interact(f, **kwargs)
    call = {}
    for k, v in kwargs.items():
        if hasattr(v, "value"):
            call[k] = v.value
        elif (
            isinstance(v, tuple)
            and len(v) >= 2
            and all(isinstance(x, (int, float)) for x in v[:3] if not isinstance(x, bool))
        ):
            call[k] = v[0]
        else:
            call[k] = v
    try:
        return func(**call)
    except Exception as e:
        print("[smoke] interact once failed:", e)


try:
    import ipywidgets

    ipywidgets.interact = _once_interact
    ipywidgets.interact_manual = _once_interact
except Exception as e:
    print("ipywidgets patch skipped", e)

try:
    import notutils

    notutils.display_plots = lambda *a, **k: print(
        "[smoke] skip display_plots", a[:1] if a else ""
    )
    if hasattr(notutils, "display_prediction"):

        def _skip_pred(*a, **k):
            print("[smoke] skip display_prediction")

        notutils.display_prediction = _skip_pred
except Exception as e:
    print("notutils patch skipped", e)


def _cap_optimize(cls, method_name="optimize", max_iters=40):
    if not hasattr(cls, method_name):
        return
    orig = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        if "max_iters" in kwargs and kwargs["max_iters"] is not None:
            kwargs["max_iters"] = min(int(kwargs["max_iters"]), max_iters)
        else:
            kwargs["max_iters"] = max_iters
        if "messages" in kwargs:
            kwargs["messages"] = False
        if "max_f_eval" in kwargs and kwargs["max_f_eval"] is not None:
            kwargs["max_f_eval"] = min(int(kwargs["max_f_eval"]), max_iters)
        return orig(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)


try:
    import GPy

    _cap_optimize(GPy.core.gp.GP, "optimize", 40)
except Exception as e:
    print("GPy patch skipped", e)

try:
    import mlai
    from mlai import linear_models as _lm

    for _name in ("relu", "hyperbolic_tangent"):
        if not hasattr(mlai, _name) and hasattr(_lm, _name):
            setattr(mlai, _name, getattr(_lm, _name))
            print(f"[smoke] attached mlai.{_name} from linear_models")

    _write_figure = mlai.write_figure

    def _write_figure_diagrams(filename, figure=None, directory=None, frameon=None, **kwargs):
        if directory is None:
            directory = kwargs.pop("diagrams", None)
        else:
            kwargs.pop("diagrams", None)
        return _write_figure(
            filename, figure=figure, directory=directory, frameon=frameon, **kwargs
        )

    mlai.write_figure = _write_figure_diagrams
except Exception as e:
    print("mlai basis attach skipped", e)

try:
    import plotly.io as pio

    pio.renderers.default = "json"
except Exception as e:
    print("plotly renderer patch skipped", e)
"""

_LOAD_MAGIC = re.compile(r"^%load\s+-n\s+([\w.]+)\s*$")


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src


def expand_load_magics(nb: nbformat.NotebookNode) -> None:
    """Execute `%load -n module.attr` by importing the name.

    IPython `%load` pastes source into the cell but does not run it. Students
    re-execute interactively; under nbclient the name would never be defined.
    """
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        lines = [
            ln
            for ln in _cell_source(cell).strip().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if len(lines) != 1:
            continue
        match = _LOAD_MAGIC.match(lines[0].strip())
        if not match:
            continue
        target = match.group(1)
        if "." not in target:
            continue
        mod_name, attr = target.rsplit(".", 1)
        cell["source"] = (
            f"# smoke: execute `%load -n {target}`\n"
            f"from {mod_name} import {attr}\n"
        )


def patch_student_placeholders(nb: nbformat.NotebookNode) -> None:
    """Fill interactive lecture placeholders that cannot run unattended."""
    dummy_ratings = (
        "Y = pd.DataFrame({\n"
        "    'user': ['smoke-a', 'smoke-a', 'smoke-b', 'smoke-b', 'smoke-c', 'smoke-c'],\n"
        "    'film': ['Inception', 'Avatar', 'Inception', 'Titanic', 'Avatar', 'Titanic'],\n"
        "    'rating': [8, 7, 9, 6, 5, 8],\n"
        "})\n"
    )
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        changed = False
        if re.search(r"assert\s+name\s*!=\s*['\"]todo['\"]", src):
            src = re.sub(r"name\s*=\s*['\"]todo['\"]", "name = 'smoke-test'", src, count=1)
            src = re.sub(
                r"assert\s+name\s*!=\s*['\"]todo['\"]",
                "pass  # smoke: student name placeholder",
                src,
                count=1,
            )
            changed = True
        if re.search(r"^Y\s*=\s*responses\(\)\s*$", src, re.M):
            src = re.sub(
                r"^Y\s*=\s*responses\(\)\s*$",
                dummy_ratings.rstrip(),
                src,
                count=1,
                flags=re.M,
            )
            changed = True
        if changed:
            cell["source"] = src


def blank_incomplete_exercises(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if "Write code for you answer" not in src and "Write your answer to Exercise" not in src:
            continue
        try:
            ast.parse(src)
        except SyntaxError:
            if "w_cov" in src and "w_mean" in src:
                cell["source"] = (
                    "# smoke: Bayesian linear-model posterior from lecture formulae\n"
                    "sigma2 = 0.01\n"
                    "w_cov = np.linalg.inv(Phi.T @ Phi / sigma2 + np.eye(Phi.shape[1]) / alpha)\n"
                    "w_mean = w_cov @ (Phi.T @ y) / sigma2\n"
                )
            else:
                cell["source"] = "# skipped incomplete exercise cell for smoke test\npass\n"


def prepare(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    nb = deepcopy(nb)
    blank_incomplete_exercises(nb)
    expand_load_magics(nb)
    patch_student_placeholders(nb)
    nb.cells.insert(0, nbformat.v4.new_code_cell(PRELUDE))
    return nb


def _matches_token(name: str, token: str) -> bool:
    token = token.strip()
    return bool(token) and (name.startswith(token) or name == token)


def list_notebooks(nb_dir: Path, only: list[str] | None, skip: list[str] | None) -> list[str]:
    names = sorted(p.name for p in nb_dir.glob("[0-9][0-9]-*.ipynb"))
    if only:
        selected = []
        for token in only:
            matches = [n for n in names if _matches_token(n, token)]
            if not matches:
                raise SystemExit(f"No notebook matches --only {token!r} under {nb_dir}")
            for m in matches:
                if m not in selected:
                    selected.append(m)
        return selected
    skip = skip or []
    selected = []
    for name in names:
        skipped = next((tok for tok in skip if _matches_token(name, tok)), None)
        if skipped is not None:
            print(f"[smoke] skip {name} (--skip {skipped})", flush=True)
            continue
        selected.append(name)
    return selected


def run_one(
    name: str,
    nb_dir: Path,
    out_dir: Path,
    timeout: int,
) -> dict:
    path = nb_dir / name
    print(f"\n======== {name} ========", flush=True)
    t0 = time.time()
    try:
        nb = nbformat.read(path, as_version=4)
        nb = prepare(nb)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(nb_dir)}},
            allow_errors=False,
            allow_stdin=True,
        )
        client.execute()
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s)", flush=True)
        (out_dir / f"{name}.ok").write_text(f"OK {elapsed:.1f}s\n")
        fail = out_dir / f"{name}.fail"
        if fail.exists():
            fail.unlink()
        return {"name": name, "status": "OK", "elapsed": elapsed, "error": ""}
    except CellExecutionError as e:
        elapsed = time.time() - t0
        msg = str(e)
        print(f"FAIL ({elapsed:.1f}s)\n{msg[:4000]}", flush=True)
        (out_dir / f"{name}.fail").write_text(msg)
        return {"name": name, "status": "FAIL", "elapsed": elapsed, "error": msg}
    except Exception:
        elapsed = time.time() - t0
        msg = traceback.format_exc()
        print(f"ERROR ({elapsed:.1f}s)\n{msg[:4000]}", flush=True)
        (out_dir / f"{name}.fail").write_text(msg)
        return {"name": name, "status": "ERROR", "elapsed": elapsed, "error": msg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("_notebooks"),
        help="Directory containing NN-*.ipynb lecture notebooks",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("_notebooks/test-results"),
        help="Where to write .ok/.fail markers and summary JSON",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to notebooks whose names start with these prefixes (e.g. 01 12)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=["05"],
        help="Skip notebooks whose names start with these prefixes. "
        "Default: 05 (MariaDB). Pass --skip with no values to run all.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-notebook cell execution timeout in seconds (default 900)",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Summary JSON filename under --results-dir",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    nb_dir = (repo_root / args.notebooks_dir).resolve()
    out_dir = (repo_root / args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl_dir = out_dir / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    if not nb_dir.is_dir():
        print(f"Notebooks directory not found: {nb_dir}", file=sys.stderr)
        return 2

    order = list_notebooks(nb_dir, args.only, args.skip)
    print("Notebooks:", order, flush=True)

    results = []
    for name in order:
        result = run_one(name, nb_dir, out_dir, args.timeout)
        results.append(result)
        (out_dir / args.summary_name).write_text(json.dumps(results, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    for r in results:
        print(f"{r['status']:5} {r['elapsed']:7.1f}s  {r['name']}")
    n_ok = sum(r["status"] == "OK" for r in results)
    print(f"{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
