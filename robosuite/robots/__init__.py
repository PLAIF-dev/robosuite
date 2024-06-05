from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from .bimanual import Bimanual
from .manipulator import Manipulator
from .single_arm import SingleArm

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Baxter": Bimanual,
    "ALOHA": Bimanual,
    "PlaifDualRB3": Bimanual,
    "IIWA": SingleArm,
    "Jaco": SingleArm,
    "Kinova3": SingleArm,
    "Panda": SingleArm,
    "Sawyer": SingleArm,
    "UR5e": SingleArm,
    "RB3": SingleArm,
}

BIMANUAL_ROBOTS = {k.lower() for k, v in ROBOT_CLASS_MAPPING.items() if v == Bimanual}
