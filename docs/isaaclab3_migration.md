# Isaac Sim 6 / Isaac Lab 3.0 migration (branch `isaaclab-3.0`)

This branch ports the simulator side of the repo from Isaac Sim 5.1 / Isaac Lab 2.x to
**Isaac Sim 6.2 / Isaac Lab 3.0**. The `main` branch and the original `Dockerfile` are unchanged.

> **Status: statically checked against the Isaac Lab `v3.0.0` source only. Not yet built or run on a GPU machine.**
> Treat everything below marked *unverified* as a checklist for the first GPU run.

## Target environment

| | `main` (old) | `isaaclab-3.0` |
|---|---|---|
| Isaac Sim | 5.1.0 | 6.2.0.0 |
| Isaac Lab | 2.x (git submodule) | 3.0.0 (`pip install isaaclab==3.0.0`, no submodule needed) |
| Python | 3.11 | 3.12 |
| PyTorch | 2.7.0 | 2.11.0 (torchvision 0.26.0) |
| numpy | 1.26.0 (pinned) | 2.x |
| lerobot (simulator) | 0.4.2 | 0.6.1 |
| NVIDIA driver | 580 | 595 (version Isaac Sim 6 is tested with) |

## Build and run

```bash
make build-isaaclab6      # builds leisaac-isaaclab:6.0 from Dockerfile.isaac6
make launch-isaaclab6     # starts the container (separate from `make launch-isaaclab`)
```

`WARN_ON_TORCH_QUATF_ACCESS=1` is set in the container: Isaac Lab then warns whenever a quaternion tensor is read,
which helps to find leftover `(w, x, y, z)` assumptions.

## What changed

### Quaternion convention: `(w, x, y, z)` -> `(x, y, z, w)`
Isaac Lab 3.0 uses `(x, y, z, w)` everywhere, including every `rot=` config field. A wrong order does **not** raise an
error, it silently produces wrong orientations.

- All hard-coded `rot=(...)` / `.rot = (...)` literals in `tasks/*`, `eval/*` and `DININGTABLE_WORLD_ROT` were reordered
  (pure reordering, same rotation).
- `object_poses_loader.py` now returns `(pos_xyz, quat_xyzw)`; `generate.py` / `teleop.py` write that directly to the sim.
- State machines: `_shortest_quat` reads the scalar part from the last component; `_yaw_from_quat_wxyz` became
  `_yaw_from_quat_xyzw`.
- Quaternions that only flow through `isaaclab.utils.math` functions need no change (they are consistent in 3.0).

### Data access
- Asset/sensor data properties return `ProxyArray`; reads were changed to `.data.<name>.torch`.
- `root_physx_view.get_jacobians()` (state machines, `franka_keyboard.py`) was replaced by
  `robot.data.body_link_jacobian_w.torch[...]`, the same property Isaac Lab 3.0's own IK action uses.
- `write_root_pose_to_sim` -> `write_root_pose_to_sim_index(root_pose=...)`.

### Launching / rendering
- `AppLauncher` still exists in Isaac Lab 3.0 as a deprecated adapter, so the scripts' launch code is unchanged.
- `env_cfg.sim.render.*` no longer exists. `teleop.py --quality` now sets anti-aliasing through the cameras'
  `renderer_cfg`; the old `rendering_mode = "quality"` has no direct equivalent and is not applied.

### Dependencies
- `packages/simulator/pyproject.toml` installs `leisaac` from the fork branch
  `SSUYICHEN/leisaac@isaaclab-3.0` (upstream has no Isaac Lab 3.0 support and no plans for it at the time of writing).
- `lerobot==0.6.1`: 0.4.2 requires `torch<2.8` and 0.5.x requires `torch<2.11`, both incompatible with Isaac Lab 3.0's
  `torch==2.11.0`. 0.7.0 removed `lerobot.async_inference.helpers`, which `scripts/rollout.py` imports.

## Unverified / to check on the first GPU run

1. `docker build` of `Dockerfile.isaac6` and `pip` dependency resolution (Isaac Lab pins `transformers==5.10.4`;
   lerobot extras that need `transformers<5.6`, e.g. smolvla/pi, will conflict. The diffusion policy should not).
2. `--enable_cameras` / `--headless` flags: **verified on a GPU machine (driver 595.112.07, L40S)**: Isaac Lab 3.0's launcher
   no longer defines them, so `generate.py --enable_cameras` failed with `unrecognized arguments`. Cameras are now enabled
   automatically when the scene has camera sensors. The three scripts accept `--enable_cameras` as a deprecated no-op so
   existing commands and docs keep working; you can simply omit it. `--headless` is gone as well (no window is opened
   without a visualizer).
3. `scripts/rollout.py` uses `omni.kit.viewport.utility` / `omni.ui` (Kit modules); confirm they exist in Isaac Sim 6.
4. `lerobot` 0.6.1 symbols used by `rollout.py` (`policies.factory`, `policies.utils`, `async_inference.helpers`,
   `utils.constants`) and the dataset format written by `LeRobotDataset`; host-side training must read what the
   simulator container writes.
5. The default physics backend: make sure tasks run on **PhysX** (Franka config already selects `Physics: physx`).
6. Compare old vs new: same seed / same `object_poses.json` -> object start poses, end-effector poses, success rate.
7. Known pre-existing issue: `tests/test_object_poses_loader.py` imports `load_object_poses`, which does not exist
   (the loader exposes `load_episode_poses`); the test was already failing on `main`.
8. `packages/simulator/uv.lock` still reflects the old dependency set.
