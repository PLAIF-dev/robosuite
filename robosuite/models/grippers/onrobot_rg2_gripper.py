"""
Gripper with two fingers of OnRobot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnRobotRG2GripperBase(GripperModel):
    """
    Gripper with two fingers

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/onrobot_gripper_rg2.xml"), idn=idn)
        

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.1])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["rg2_l_finger_col"],
            "right_finger": ["rg2_r_finger_col"],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }


class OnRobotRG2Gripper(OnRobotRG2GripperBase):
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
            self.current_action + np.array([1.0,]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.002

    @property
    def dof(self):
        return 1
