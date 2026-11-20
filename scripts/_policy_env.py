"""Find the Python environment a policy needs, and switch to it (used by train.py and policy_server.py).

Some policies cannot share an environment with the rest of the repo (GR00T needs torch <= 2.8 for flash-attn, the
default ``uv sync`` environment uses torch 2.10). Each of them gets its own repo-local environment:

    .venv         default environment (``uv sync``): diffusion, act, ...
    .venv-groot   GR00T (``make install-groot``)

Scripts call :func:`ensure_env` with the policy type and are transparently re-run with that environment's Python, so
nobody has to remember which environment to activate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".venv"
# Policies that need their own environment, and the make target that creates it.
POLICY_ENVS: dict[str, tuple[Path, str]] = {
    "groot": (ROOT / ".venv-groot", "make install-groot"),
}


def env_for(policy: str) -> Path:
    return POLICY_ENVS[policy][0] if policy in POLICY_ENVS else DEFAULT_ENV


def env_executable(policy: str, name: str) -> str | None:
    """Path of ``name`` (e.g. ``lerobot-train``) inside the policy's environment, or None if it is not there."""
    path = env_for(policy) / "bin" / name
    return str(path) if path.exists() else None


def ensure_env(policy: str) -> None:
    """Re-run the current script with the policy's environment, unless it is already running in it.

    If the repo-local environment does not exist, nothing happens: the caller is assumed to have activated a suitable
    environment by hand (for example inside a container, or a custom setup). A policy with its own environment
    additionally gets a hint on how to create it.
    """
    target = env_for(policy)
    python = target / "bin" / "python"
    if not python.exists():
        if policy in POLICY_ENVS:
            print(
                f"[env] '{target.name}' not found. Create the {policy} environment with `{POLICY_ENVS[policy][1]}` "
                "(continuing with the current Python environment).",
                file=sys.stderr,
                flush=True,
            )
        return
    if Path(sys.prefix).resolve() == target.resolve():
        return
    print(f"[env] switching to {target}", flush=True)
    os.execv(str(python), [str(python), *sys.argv])
