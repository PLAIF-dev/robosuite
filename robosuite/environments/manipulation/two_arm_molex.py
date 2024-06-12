import collections
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Union, Iterable

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MolexCableSolidObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    UniformRandomSampler,
    SequentialCompositeSampler,
)


class TwoArmMolex(TwoArmEnv):
    """
    This class corresponds to the molex cable manipulation for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="bimanual-plaif-molex",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.6, 0.8),
        table_friction=(1.0, 0.3, 0.2),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=["robot0_stand_belly_cam", "robot0_stand_head_cam"],
        camera_heights=[1080, 720],
        camera_widths=[1920, 1280],
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        # grasp_target_pose: (float or 2-tuple, float or 2-tuple)
        #   relative grasp position on the cable,
        #   and grasp angle deviation (radian) from perpendicular angle.
        grasp_target_pose: Optional[
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]]
        ] = (0.5, 0.0),
        # grasp_tolerance: (float, float) tolerance for the grasp target pose. (ratio, rad)
        grasp_tolerance: Optional[Tuple[float, float]] = (
            0.1,
            0.17,
        ),
        # cable_target_pos: (float, float, float) cable target xyz position
        # fmt: off
        cable_target_pos: Optional[Tuple[float, float, float,]] = (0.0, 0.1, 0.9,),
        # fmt: on
        cable_pos_tolerance: float = 0.02,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # target grasp pose and cable position
        self.grasp_target_pose = grasp_target_pose
        self.grasp_tolerance = grasp_tolerance
        self.cable_target_pos = np.array(cable_target_pos)
        self.cable_pos_tolerance = cable_pos_tolerance  # meters

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="NullMount",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.0 is provided if the cable is lifted to a
              certain position and is perpendicular to the gripper.

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._cable_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if "bimanual" in self.env_configuration:

            if self.env_configuration == "bimanual-plaif-molex":
                self.robots[0].robot_model.set_base_xpos([-0.32, 0, 0.8])
            else:
                xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](
                        self.table_full_size[1]
                    )
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            elif (
                self.env_configuration == "single-arm-parallel"
            ):  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](
                        self.table_full_size[0]
                    )
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.molex_cable = MolexCableSolidObject(name="molex_cable")

        # Create placement initializer
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(
                name="ObjectSampler"
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="MolexCableSampler",
                    x_range=[0.0, 0.1],
                    y_range=[-0.05, 0.05],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    z_offset=0.01,
                    reference_pos=self.table_offset,
                    rotation_axis="euler",
                    rotation=(0, 0, (-np.pi / 2, np.pi / 2)),
                )
            )

        self.placement_initializer.reset()
        self.placement_initializer.add_objects_to_sampler(
            "MolexCableSampler", self.molex_cable
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[
                self.molex_cable,
            ],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.molex_cable_body_id = self.sim.model.body_name2id(
            self.molex_cable.root_body
        )

        self.table_top_id = self.sim.model.site_name2id("table_top")

        self.cable_upside_id = self.sim.model.site_name2id(
            self.molex_cable.important_sites["upside"]
        )
        self.cable_downside_id = self.sim.model.site_name2id(
            self.molex_cable.important_sites["downside"]
        )
        self.cable_conn0_id = self.sim.model.site_name2id(
            self.molex_cable.important_sites["connector_0"]
        )
        self.cable_conn1_id = self.sim.model.site_name2id(
            self.molex_cable.important_sites["connector_1"]
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            if "bimanual" in self.env_configuration:
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of object

            @sensor(modality=modality)
            def cable_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.molex_cable_body_id])

            @sensor(modality=modality)
            def cable_quat(obs_cache):
                return T.convert_quat(
                    self.sim.data.body_xquat[self.molex_cable_body_id], to="xyzw"
                )

            @sensor(modality=modality)
            def gripper0_to_cable(obs_cache):
                return (
                    obs_cache["cable_xpos"] - obs_cache[f"{pf0}eef_pos"]
                    if "cable_xpos" in obs_cache and f"{pf0}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper1_to_cable(obs_cache):
                return (
                    obs_cache["cable_xpos"] - obs_cache[f"{pf1}eef_pos"]
                    if "cable_xpos" in obs_cache and f"{pf1}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cable_pos, cable_quat, gripper0_to_cable, gripper1_to_cable]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    # robosuite injects free joint to the xml object, so we use the last joint as the object pose.
                    obj.joints[-1],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        self.current_grasp_target_pose = np.zeros(2)

        if isinstance(self.grasp_target_pose[0], collections.abc.Iterable):
            grasp_point_ratio = np.random.uniform(
                min(self.grasp_target_pose[0]), max(self.grasp_target_pose[0])
            )
        else:
            grasp_point_ratio = self.grasp_target_pose[0]

        self.current_grasp_target_pose[0] = np.clip(grasp_point_ratio, 0, 1)

        if isinstance(self.grasp_target_pose[1], collections.abc.Iterable):
            self.current_grasp_target_pose[1] = np.random.uniform(
                min(self.grasp_target_pose[1]), max(self.grasp_target_pose[1])
            )
        else:
            self.current_grasp_target_pose[1] = self.grasp_target_pose[1]

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to each handle
        if vis_settings["grippers"]:
            cable = (
                self.molex_cable.important_sites["upside"] / 2
                + self.molex_cable.important_sites["downside"] / 2
            )
            gripper = (
                self.robots[0].gripper[self.robots[0].arms[0]]
                if "bimanual" in self.env_configuration
                else self.robots[0].gripper
            )
            self._visualize_gripper_to_target(
                gripper=gripper, target=cable, target_type="site"
            )

    def _check_success(self):
        """
        Check if molex cable is successfully grasped and moved to target pose.

        Returns:
            bool: True if molex cable is grasped and moved to target pose
        """
        (g0, g1) = (
            (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
            if "bimanual" in self.env_configuration
            else (self.robots[0].gripper, self.robots[1].gripper)
        )

        grasp_point_err = 1000000
        grasp_angle_err = 1000000

        # Check if the cable is successfully grasped
        if self._check_grasp(g0, "molex_cable_cable_col"):
            dist_g0_c0 = np.linalg.norm(self.cable_conn0_id - self._eef0_xpos, 2)
            dist_g0_c1 = np.linalg.norm(self.cable_conn1_id - self._eef1_xpos, 2)

            grasp_point = 1 - abs(dist_g0_c0 - dist_g0_c1) / (dist_g0_c0 + dist_g0_c1)

            grasp_point_err = np.abs(
                (self.current_grasp_target_pose[0] - (grasp_point / 2) + 0.5) % 0.5
            )

            eef_vec = T.quat2mat(self._eef0_xquat)[:, 2]
            cable_vec = T.quat2mat(self._cable_quat)[:, 2]

            # how to tell the direction of the cable?
            grasp_angle = np.arccos(np.dot(eef_vec, cable_vec)) - np.pi
            grasp_angle_err = abs(self.current_grasp_target_pose[1] - grasp_angle)

        return all(
            (
                grasp_point_err < self.grasp_tolerance[0],
                grasp_angle_err < self.grasp_tolerance[1],
                np.linalg.norm(self.cable_target_pos - self._cable_xpos, 2) < 0.02,
            )
        )

    @property
    def _cable_xpos(self):
        """
        Grab the position of the plug body.

        Returns:
            np.array: (x,y,z) position of the plug body
        """
        return self.sim.data.body_xpos[self.molex_cable_body_id]

    @property
    def _cable_quat(self):
        """
        Grab the orientation of the plug body.

        Returns:
            np.array: (x,y,z,w) quaternion of the plug body
        """
        return T.convert_quat(
            self.sim.data.body_xquat[self.molex_cable_body_id], to="xyzw"
        )

    @property
    def _gripper0_to_cable(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._cable_xpos - self._eef0_xpos

    @property
    def _gripper1_to_cable(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._cable_xpos - self._eef1_xpos

    @property
    def _cable_conn0_xpos(self):
        return self.sim.data.site_xpos[self.cable_conn0_id]

    @property
    def _cable_conn0_xquat(self):
        return self.sim.data.site_xquat[self.cable_conn0_id]

    @property
    def _cable_conn1_xpos(self):
        return self.sim.data.site_xpos[self.cable_conn1_id]

    @property
    def _cable_conn1_xquat(self):
        return self.sim.data.site_xquat[self.cable_conn1_id]

    @property
    def _cable_upside(self):
        return self.sim.data.site_xpos[self.cable_upside_id]

    @property
    def _cable_downside(self):
        return self.sim.data.site_xpos[self.cable_downside_id]

    @property
    def _grasp_point_xpos(self):
        return (
            self._cable_conn0_xpos * self.current_grasp_target_pose[0]
            + (1 - self.current_grasp_target_pose[0]) * self._cable_conn1_xpos
        )
