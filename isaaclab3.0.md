# Isaac Sim 6 / Isaac Lab 3.0 port (`isaaclab-3.0` branch)

This branch ports the simulator side of the repo from **Isaac Sim 5.1 / Isaac Lab 2.x** to
**Isaac Sim 6.2 / Isaac Lab 3.0**, so the pipeline can run with **NVIDIA driver 595**.

The rule of the port: **nothing changes except API / syntax usage.** Task logic, state machines, config values and the
training setup are untouched. `main`, the original `Dockerfile` and `make launch-isaaclab` are not modified, so the old
and the new setup can be used side by side.

Per-API details of the port: [docs/isaaclab3_migration.md](docs/isaaclab3_migration.md).

## 1. What was changed

**Environment**

- New `Dockerfile.isaac6` and `make build-isaaclab6` / `make launch-isaaclab6` (image `leisaac-isaaclab:6.0`):
  Python 3.12, PyTorch 2.11, Isaac Sim 6.2, Isaac Lab from the `isaaclab==3.0.0` PyPI wheel (no submodule, no
  `isaaclab.sh`, which Isaac Lab 3.0 removed).
- `packages/simulator/pyproject.toml`: Python >= 3.12, `lerobot[dataset,diffusion]==0.6.1` (0.4.x needs `torch<2.8`,
  Isaac Lab 3.0 needs `torch==2.11`), and `leisaac` from a fork branch (below).

**LeIsaac** ([`SSUYICHEN/leisaac`, branch `isaaclab-3.0`](https://github.com/SSUYICHEN/leisaac/tree/isaaclab-3.0);
upstream has no Isaac Lab 3.0 support). Only what this repo uses was ported: removed the `TerminationManager` monkey
patch (fixed upstream in Isaac Lab 3.0), replaced `isaacsim.core.utils.prims`, `.data.*` reads via `.torch`,
`slice` `env_ids` in the recorder managers, an `image()` shim (removed from `isaaclab.envs.mdp`), the
`SimulationCfg.physx` / `render` -> `physics` / renderer settings, and per-task import tolerance.

**This repo**

- **Quaternion order `(w, x, y, z)` -> `(x, y, z, w)`**, which Isaac Lab 3.0 uses everywhere: all `rot=` literals in
  the task and eval configs, `object_poses_loader` (now returns xyzw), the yaw / shortest-quaternion helpers in the
  state machines. A wrong order does not raise an error, it silently gives wrong orientations.
- Data access: asset data properties are `ProxyArray` (read with `.torch`); the Jacobian comes from
  `robot.data.body_link_jacobian_w` instead of `root_physx_view`; `write_root_pose_to_sim` -> `..._index`.
- Franka: `FRANKA_PANDA_LEGACY_CFG`, which is the Isaac Lab 2.x config kept by 3.0 (same USD, same gains); the new
  `FRANKA_PANDA_CFG` has different actuator groups.
- `sim.physx` / `sim.render` (removed in 3.0) -> `sim.physics = PhysxCfg(...)` and the cameras' `renderer_cfg`;
  dome-light randomization uses `find_matching_prims`; `ActionTermCfg` is re-exported from the template `mdp`.
- `--enable_cameras`: Isaac Lab 3.0 no longer defines it, but Kit still needs it to load the RTX renderer when
  `AppLauncher` is used. `generate.py`, `teleop.py` and `rollout.py` now enable cameras by default and still accept
  the flag.
- Optional, off by default: `rollout.py --save_video_dir <dir>` saves each episode's `front` / `wrist` camera as mp4.
- New, additive: `scripts/train.py` (one training entry for diffusion and GR00T), `make install-groot` (pinned GR00T
  environment), and `scripts/policy_server.py` + `rollout.py --policy_server` (run a policy on the host and query it
  from the simulator container). Existing commands are unchanged.

**Not changed:** `main`, `Dockerfile`, `packages/umi`, the host training environment (lerobot 0.4.4), task / state
machine logic.

## 2. How to reproduce

Reference machine: Ubuntu 22.04, NVIDIA L40S 46GB, driver 595.112.07, 2 cores / 32GB RAM.

**Requirements**

- Linux x86_64 and an NVIDIA GPU **with RT cores** (RTX 4080 16GB or better, L40S, ...; A100 / H100 are not
  supported), driver 595, 100GB+ disk, 32GB+ RAM, internet access (assets are downloaded on first run).
- `git`, Docker, `nvidia-container-toolkit`
  (check: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`).
- For host-side training: `uv` and `ffmpeg` (`sudo apt-get install -y ffmpeg`).

**Get the branch, build, launch**

```bash
git clone -b isaaclab-3.0 https://github.com/SSUYICHEN/aicapstone.git && cd aicapstone
make build-isaaclab6       # can take tens of minutes
make launch-isaaclab6      # opens a shell inside the container
python -c "import isaacsim, isaaclab, leisaac; print(isaaclab.__version__)"    # 3.0.0
```

- The first start compiles RTX shaders (5-10+ minutes). The container runs with
  `--rm`, so the cache is lost on exit: stay in the same container while testing.
- `UserWarning: Reading .torch on a wp.quatf-typed ProxyArray` is intentional (`WARN_ON_TORCH_QUATF_ACCESS=1` is set to
  catch leftover wxyz assumptions; `unset` it to silence). `xhost` / `GLFW` / display warnings can be ignored on
  headless machines.

**Stage 1, generate data (container)**

```bash
python scripts/datagen/generate.py --task HCIS-CupStacking-SingleArm-v0 --num_envs 1 --device cuda \
    --record --use_lerobot_recorder --lerobot_dataset_repo_id ${HF_USER}/<repo_id> --num_demos 20
```

`--record` is required; only successful episodes are exported.
The dataset is written inside the container (`~/.cache/huggingface/lerobot/...`) and lost when it exits, so copy it
to `/workspace/aicapstone/data/` first. Then upload it and tag the dataset repo with its `codebase_version`
(`v3.0`, see `meta/info.json`); `lerobot-train` needs the tag to download a dataset from the Hub:

```bash
hf upload ${HF_USER}/<repo_id> data/<repo_id> --repo-type dataset
python -c "from huggingface_hub import HfApi; HfApi().create_tag('${HF_USER}/<repo_id>', tag='v3.0', repo_type='dataset')"
```

**Stage 2, train (host, not Docker; Isaac Lab is not involved)**

`scripts/train.py` is one entry point for every policy; `--policy` selects the policy and its defaults. It runs the
policy in the right environment automatically (`.venv` or `.venv-groot`). Install the environment of the policy you
want (both can coexist):

| Policy | Environment | Lerobot |
|---|---|---|
| `diffusion` | `uv sync` (`.venv`) | 0.4.4 |
| `groot` (GR00T N1.5) | `make install-groot` (`.venv-groot`) | 0.4.4, flash-attn, pinned in `envs/groot/constraints.txt` |

GR00T cannot share an environment with Isaac Lab or the default one (it needs `torch<=2.8` for flash-attn and
`transformers<5`), hence the separate, pinned environment. `make install-groot` needs `uv` and `ffmpeg`; flash-attn is installed from a prebuilt wheel, so no CUDA toolkit is
required.

```bash
python scripts/train.py --policy diffusion --dataset ${HF_USER}/<repo_id> --name cup_dp
python scripts/train.py --policy groot     --dataset ${HF_USER}/<repo_id> --name cup_groot
```

Defaults per policy (override with `--steps`, `--save-freq`, `--log-freq`, `--batch-size`, `--num-workers`):
diffusion 20000 steps / batch 16, groot 10000 steps / batch 4 (L40S 46GB). Unknown arguments are passed to
`lerobot-train` (e.g. `--policy.tune_llm=true`); `--dry-run` prints the command. Checkpoints go to
`checkpoints/<name>` (must not exist yet), which the container can see.

**Stage 3, rollout (container)**

A diffusion policy runs directly in the container:

```bash
python scripts/rollout.py --task=eval/cup_stacking_eval.py --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=checkpoints/<name>/checkpoints/last/pretrained_model \
    --policy_action_horizon=1 --device=cuda --eval_rounds=10 --episode_length_s=20 \
    --save_video_dir data/rollout_videos
```

The container uses lerobot 0.6.1, host training uses 0.4.4; a 0.4.4-trained diffusion policy loads and runs in the
container.

**GR00T, or any policy that cannot be installed next to Isaac Lab, runs through a policy server.** The policy runs on
the host in its own environment; the simulator in the container queries it over TCP (`--net=host`, so `127.0.0.1`
reaches the host). Start the server on the host, then run the rollout in the container with `--policy_server`:

```bash
# host
python scripts/policy_server.py --policy-type groot \
    --checkpoint checkpoints/<name>/checkpoints/last/pretrained_model          # add --port N if 5555 is in use
```

```bash
# container
python scripts/rollout.py --task=eval/cup_stacking_eval.py --policy_server=127.0.0.1:5555 \
    --policy_action_horizon=1 --device=cuda --eval_rounds=10 --episode_length_s=20 --save_video_dir data/rollout_videos
```

`--policy-type diffusion` works the same way. There is no authentication; the server binds to `127.0.0.1` by default.
Policies that need `transformers` (`pi0`, `smolvla`, ...) cannot be installed in the container (Isaac Lab 3.0 pins
`transformers==5.10.4`, lerobot needs `<5.6`).

## 3. Documentation

Apart from the differences described above, the workflow is unchanged, so the existing documents still apply.

| For | See |
|---|---|
| Every API change of the port, file by file | [docs/isaaclab3_migration.md](docs/isaaclab3_migration.md) |
| Environment of this branch | `Dockerfile.isaac6`, `Makefile` (`build-isaaclab6`, `launch-isaaclab6`), `packages/simulator/pyproject.toml` |
| LeIsaac changes | [`SSUYICHEN/leisaac`, branch `isaaclab-3.0`](https://github.com/SSUYICHEN/leisaac/tree/isaaclab-3.0) |
| End-to-end workflow | [README.md](README.md), [docs/getting_started.md](docs/getting_started.md) |
| Data generation | [docs/synthetic_data_generation.md](docs/synthetic_data_generation.md) |
| Training | [docs/lerobot_training.md](docs/lerobot_training.md) |
| Rollout / evaluation | [docs/lerobot_rollout.md](docs/lerobot_rollout.md) |
| Keyboard teleoperation | [docs/keyboard_teleoperation.md](docs/keyboard_teleoperation.md) |
