"""Common training entry point for LeRobot policies (diffusion, GR00T, ...).

Runs on the HOST; Isaac Lab is not involved. The right environment is selected automatically (``.venv`` from
``uv sync``, or ``.venv-groot`` from ``make install-groot`` for GR00T). It builds one
``lerobot-train`` command from a few common options plus per-policy defaults, so that switching the policy is a matter
of changing ``--policy``:

    python scripts/train.py --policy diffusion --dataset ${HF_USER}/<repo_id> --name cup_dp
    python scripts/train.py --policy groot     --dataset ${HF_USER}/<repo_id> --name cup_groot

Any argument this script does not know is forwarded to ``lerobot-train`` unchanged, e.g.
``--policy.tune_llm=true`` or ``--dataset.root=data/<repo_id>``. Use ``--dry-run`` to only print the command.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys

from _policy_env import env_executable, ensure_env

# Per-policy defaults. They only fill in options the user did not pass. Values marked "untested" have not been run
# on the reference machine yet.
POLICY_PRESETS: dict[str, dict] = {
    # Values used for the verified diffusion runs.
    "diffusion": {"steps": 20000, "save_freq": 5000, "log_freq": 100, "batch_size": 16, "extra": []},
    # GR00T N1.5 (lerobot 0.4.4, needs `lerobot[groot]` incl. flash-attn). The batch size is the one verified on an
    # L40S 46GB (50 steps, ~2.9 step/s); a larger batch has not been tried. Saving a checkpoint takes ~2 minutes.
    "groot": {"steps": 10000, "save_freq": 2000, "log_freq": 50, "batch_size": 4, "extra": []},
}
FALLBACK_PRESET = {"steps": 20000, "save_freq": 5000, "log_freq": 100, "batch_size": 16, "extra": []}


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    preset = POLICY_PRESETS.get(args.policy, FALLBACK_PRESET)
    steps = args.steps if args.steps is not None else preset["steps"]
    save_freq = args.save_freq if args.save_freq is not None else preset["save_freq"]
    log_freq = args.log_freq if args.log_freq is not None else preset["log_freq"]
    batch_size = args.batch_size if args.batch_size is not None else preset["batch_size"]
    name = args.name or f"{args.policy}_{args.dataset.split('/')[-1]}"

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset}",
        f"--policy.type={args.policy}",
        f"--output_dir={args.output_dir or f'checkpoints/{name}'}",
        f"--job_name={name}",
        f"--policy.device={args.device}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
        f"--policy.push_to_hub={'true' if args.push_to_hub else 'false'}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        f"--batch_size={batch_size}",
    ]
    if args.num_workers is not None:
        cmd.append(f"--num_workers={args.num_workers}")
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    cmd += preset["extra"]
    cmd += passthrough
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--policy", required=True, help="LeRobot policy type, e.g. diffusion, groot, act.")
    parser.add_argument("--dataset", required=True, help="Dataset repo id, e.g. ${HF_USER}/<repo_id>.")
    parser.add_argument("--name", default=None, help="Run name (default: <policy>_<dataset name>).")
    parser.add_argument("--output-dir", default=None, help="Default: checkpoints/<name> (must not exist yet).")
    parser.add_argument("--dataset-root", default=None, help="Use a local copy of the dataset instead of the Hub.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the policy to the Hub after training.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the lerobot-train command.")
    args, passthrough = parser.parse_known_args()
    if not args.dry_run:
        ensure_env(args.policy)

    cmd = build_command(args, passthrough)
    cmd[0] = env_executable(args.policy, "lerobot-train") or shutil.which("lerobot-train") or cmd[0]
    print("[train] " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    if not os.path.exists(cmd[0]) and shutil.which(cmd[0]) is None:
        print("[train] lerobot-train not found. Run `uv sync` (default policies) or `make install-groot` (groot).", file=sys.stderr)
        return 1
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
