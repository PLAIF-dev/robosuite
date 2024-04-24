"""
Gripper with two fingers for ALOHA Robots.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ALOHAGripperBase(GripperModel):
    """
    Gripper with long two-fingered parallel jaw.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/aloha_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0084, 0.0084])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["l_finger_col", "l_finger_g0_col", "l_finger_g1_col", "l_finger_g2_col"],
            "right_finger": ["r_finger_col", "r_finger_g0_col", "r_finger_g1_col", "r_finger_g2_col"],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }


class ALOHAGripper(ALOHAGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + np.array([1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
