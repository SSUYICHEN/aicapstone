"""Serve a trained LeRobot policy to the simulator over TCP (runs on the HOST, not in the Isaac Lab container).

Policies such as GR00T need packages (flash-attn, ``transformers<5``) that cannot be installed next to Isaac Lab 3.0.
Run the policy here, in the environment it was trained in, and let ``scripts/rollout.py`` in the simulator container
query it:

    # host (switches to the environment the policy needs automatically)
    python scripts/policy_server.py --policy-type groot \\
        --checkpoint checkpoints/<name>/checkpoints/last/pretrained_model

    # container (--net=host, so 127.0.0.1 reaches the host)
    python scripts/rollout.py --task=eval/cup_stacking_eval.py --policy_server=127.0.0.1:5555 ...

There is no authentication: keep the default ``--host 127.0.0.1`` unless the network is trusted.
"""

from __future__ import annotations

import argparse
import errno
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "simulator" / "src"))

from _policy_env import ensure_env  # noqa: E402  (standard library only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-type", required=True, help="LeRobot policy type, e.g. diffusion, groot.")
    parser.add_argument("--checkpoint", required=True, help="Path to <run>/checkpoints/<step>/pretrained_model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ensure_env(args.policy_type)  # re-runs this script in the policy's environment (.venv / .venv-groot)

    # Imported only now: they need numpy / lerobot, which the Python that started this script may not have.
    from simulator.policy_service.runner import LeRobotPolicyRunner
    from simulator.policy_service.server import serve

    runner = LeRobotPolicyRunner(args.policy_type, args.checkpoint, args.device)
    try:
        serve(
            runner,
            host=args.host,
            port=args.port,
            on_listening=lambda port: print(f"[policy-server] listening on {args.host}:{port}", flush=True),
        )
    except KeyboardInterrupt:
        print("[policy-server] stopped", flush=True)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            print(
                f"[policy-server] port {args.port} is already in use. Stop the other process or choose another port"
                " with --port (use the same port for rollout.py --policy_server=HOST:PORT).",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
