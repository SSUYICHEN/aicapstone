from pathlib import Path

from isaaclab.assets import ArticulationCfg
# Isaac Lab 3.0 replaced FRANKA_PANDA_CFG by a new flat asset with different actuator groups (panda_arm / panda_hand)
# and USD-driven gains. FRANKA_PANDA_LEGACY_CFG is the Isaac Lab 2.x configuration (same USD, panda_shoulder /
# panda_forearm / panda_hand actuators and gains), which the task configs were tuned against.
from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_CFG as FRANKA_PANDA_CFG


from leisaac.utils.constant import ASSETS_ROOT

FRANKA_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "franka.usd"

# FRANKA_PANDA_CFG = FRANKA_PANDA_CFG.replace(
#     spawn=FRANKA_PANDA_CFG.spawn.replace(
#         usd_path=str(FRANKA_ASSET_PATH),
#     )
# )
