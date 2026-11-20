"""Tests for the per-policy environment selection used by scripts/train.py and scripts/policy_server.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _policy_env as pe  # noqa: E402


def _make_env(root: Path) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python").write_text("")
    (root / "bin" / "lerobot-train").write_text("")
    return root


def test_policies_map_to_their_environment():
    assert pe.env_for("diffusion") == pe.DEFAULT_ENV
    assert pe.env_for("act") == pe.DEFAULT_ENV
    assert pe.env_for("groot").name == ".venv-groot"


def test_env_executable_found_and_missing(tmp_path, monkeypatch):
    env = _make_env(tmp_path / "env")
    monkeypatch.setitem(pe.POLICY_ENVS, "groot", (env, "make install-groot"))
    assert pe.env_executable("groot", "lerobot-train") == str(env / "bin" / "lerobot-train")
    assert pe.env_executable("groot", "does-not-exist") is None


def test_ensure_env_reruns_the_script_in_the_policy_environment(tmp_path, monkeypatch):
    env = _make_env(tmp_path / "env")
    monkeypatch.setitem(pe.POLICY_ENVS, "groot", (env, "make install-groot"))
    calls = []
    monkeypatch.setattr(pe.os, "execv", lambda path, argv: calls.append((path, argv)))
    pe.ensure_env("groot")
    assert len(calls) == 1
    assert calls[0][0] == str(env / "bin" / "python")
    assert calls[0][1][0] == str(env / "bin" / "python")


def test_ensure_env_does_nothing_when_already_in_the_environment(tmp_path, monkeypatch):
    env = _make_env(tmp_path / "env")
    monkeypatch.setitem(pe.POLICY_ENVS, "groot", (env, "make install-groot"))
    monkeypatch.setattr(pe.sys, "prefix", str(env))
    calls = []
    monkeypatch.setattr(pe.os, "execv", lambda path, argv: calls.append(path))
    pe.ensure_env("groot")
    assert calls == []


def test_ensure_env_hints_how_to_create_a_missing_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(pe.POLICY_ENVS, "groot", (tmp_path / "missing", "make install-groot"))
    calls = []
    monkeypatch.setattr(pe.os, "execv", lambda path, argv: calls.append(path))
    pe.ensure_env("groot")
    assert calls == []
    assert "make install-groot" in capsys.readouterr().err
