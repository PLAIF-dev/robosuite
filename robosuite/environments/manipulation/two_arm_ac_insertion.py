from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import ACPlugObject, ACSocketObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.mjcf_utils import xml_remove_name_attr_under_composite


class TwoArmACInsertion(TwoArmEnv):
    """
    This class corresponds to the 220V AC plug to socket insertion task for two robot arms.

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
        env_configuration="single-arm-plaif-ac",
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
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        # plaif additional parameters
        pre_grasped=False,
        pre_aligned=False,
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

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._plug_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (
                (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
                if self.env_configuration == "bimanual"
                else (self.robots[0].gripper, self.robots[1].gripper)
            )

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[1])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            elif  self.env_configuration == "single-arm-parallel":  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
            elif self.env_configuration == "single-arm-plaif-ac":
                self.robots[0].robot_model.set_base_xpos([0.13, -0.53, 0.8])
                self.robots[0].robot_model.set_base_ori([0, 0, np.pi / 2])
                self.robots[1].robot_model.set_base_xpos([0.13, 0.53, 0.8])
                self.robots[1].robot_model.set_base_ori([0, 0, -np.pi / 2])

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.socket = ACSocketObject(name="socket")
        self.plug = ACPlugObject(name="plug")

        # Create placement initializer
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="PlugSampler",
                    x_range=[0.004999, 0.005],
                    y_range=[0.32999, 0.33],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    z_offset=0.01,
                    reference_pos=self.table_offset,
                    #rotation=(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6),
                    rotation=np.pi / 2,
                )
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="SocketSampler",
                    x_range=[-0.14, -0.13999],
                    y_range=[-0.335, -0.33499],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    #rotation=(np.pi / 3, np.pi / 2.2),
                    rotation=np.pi / 2,
                )
            )
            
        self.placement_initializer.reset()
        self.placement_initializer.add_objects_to_sampler("PlugSampler", self.plug)
        self.placement_initializer.add_objects_to_sampler("SocketSampler", self.socket)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.socket, self.plug],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.plug_body_id = self.sim.model.body_name2id(self.plug.root_body)
        self.socket_body_id = self.sim.model.body_name2id(self.socket.root_body)

        self.table_top_id = self.sim.model.site_name2id("table_top")

        self.plug_left_rod_base_id = self.sim.model.site_name2id(self.plug.important_sites["left_rod_base"])
        self.plug_left_rod_end_id = self.sim.model.site_name2id(self.plug.important_sites["left_rod_end"])
        self.plug_right_rod_base_id = self.sim.model.site_name2id(self.plug.important_sites["right_rod_base"])
        self.plug_right_rod_end_id = self.sim.model.site_name2id(self.plug.important_sites["right_rod_end"])

        self.socket_left_hole_enter_id = self.sim.model.site_name2id(self.socket.important_sites["left_hole_enter"])
        self.socket_left_hole_end_id = self.sim.model.site_name2id(self.socket.important_sites["left_hole_end"])
        self.socket_right_hole_enter_id = self.sim.model.site_name2id(self.socket.important_sites["right_hole_enter"])
        self.socket_right_hole_end_id = self.sim.model.site_name2id(self.socket.important_sites["right_hole_end"])


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
            if self.env_configuration == "bimanual":
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of object

            @sensor(modality=modality)
            def plug_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.plug_body_id])

            @sensor(modality=modality)
            def plug_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.plug_body_id], to="xyzw")
            
            @sensor(modality=modality)
            def socket_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.socket_body_id])
            
            @sensor(modality=modality)
            def socket_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.socket_body_id], to="xyzw")
            
            @sensor(modality=modality)
            def gripper0_to_socket(obs_cache):
                return (
                    obs_cache["socket_xpos"] - obs_cache[f"{pf0}eef_pos"]
                    if "socket_xpos" in obs_cache and f"{pf0}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper1_to_plug(obs_cache):
                return (
                    obs_cache["plug_xpos"] - obs_cache[f"{pf1}eef_pos"]
                    if "plug_xpos" in obs_cache and f"{pf1}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            #sensors = [pot_pos, pot_quat, handle0_xpos, handle1_xpos, gripper0_to_handle0, gripper1_to_handle1]
            sensors = [plug_pos, plug_quat, socket_pos, socket_quat, gripper0_to_socket, gripper1_to_plug]
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
                self.sim.data.set_joint_qpos(obj.joints[-1], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
            handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
            grippers = (
                [self.robots[0].gripper[arm] for arm in self.robots[0].arms]
                if self.env_configuration == "bimanual"
                else [robot.gripper for robot in self.robots]
            )
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return False
    

    @property
    def _plug_left_rod_base_xpos(self):
        """
        Grab the position of the left rod base of the plug.

        Returns:
            np.array: (x,y,z) position of left rod base
        """
        return self.sim.data.site_xpos[self.plug_left_rod_base_id]
    
    @property
    def _plug_left_rod_end_xpos(self):
        """
        Grab the position of the left rod end of the plug.

        Returns:
            np.array: (x,y,z) position of left rod end
        """
        return self.sim.data.site_xpos[self.plug_left_rod_end_id]
    
    @property
    def _plug_right_rod_base_xpos(self):
        """
        Grab the position of the right rod base of the plug.

        Returns:
            np.array: (x,y,z) position of right rod base
        """
        return self.sim.data.site_xpos[self.plug_right_rod_base_id]
    
    @property
    def _plug_right_rod_end_xpos(self):
        """
        Grab the position of the right rod end of the plug.

        Returns:
            np.array: (x,y,z) position of right rod end
        """
        return self.sim.data.site_xpos[self.plug_right_rod_end_id]

    @property
    def _socket_left_hole_enter_xpos(self):
        """
        Grab the position of the left hole enter of the socket.

        Returns:
            np.array: (x,y,z) position of left hole enter
        """
        return self.sim.data.site_xpos[self.socket_left_hole_enter_id]

    @property
    def _socket_left_hole_end_xpos(self):
        """
        Grab the position of the left hole end of the socket.

        Returns:
            np.array: (x,y,z) position of left hole end
        """
        return self.sim.data.site_xpos[self.socket_left_hole_end_id]
    
    @property
    def _socket_right_hole_enter_xpos(self):
        """
        Grab the position of the right hole enter of the socket.

        Returns:
            np.array: (x,y,z) position of right hole enter
        """
        return self.sim.data.site_xpos[self.socket_right_hole_enter_id]
    
    @property
    def _socket_right_hole_end_xpos(self):
        """
        Grab the position of the right hole end of the socket.

        Returns:
            np.array: (x,y,z) position of right hole end
        """
        return self.sim.data.site_xpos[self.socket_right_hole_end_id]

    @property
    def _plug_xpos(self):
        """
        Grab the position of the plug body.

        Returns:
            np.array: (x,y,z) position of the plug body
        """
        return self.sim.data.body_xpos[self.plug_body_id]
    
    @property
    def _plug_quat(self):
        """
        Grab the orientation of the plug body.

        Returns:
            np.array: (x,y,z,w) quaternion of the plug body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.plug_body_id], to="xyzw")
    
    @property
    def _socket_xpos(self):
        """
        Grab the position of the socket body.

        Returns:
            np.array: (x,y,z) position of the socket body
        """
        return self.sim.data.body_xpos[self.socket_body_id]
    
    @property
    def _socket_quat(self):
        """
        Grab the orientation of the socket body.

        Returns:
            np.array: (x,y,z,w) quaternion of the socket body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.socket_body_id], to="xyzw")

    @property
    def _gripper0_to_socket(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._socket_xpos - self._eef0_xpos

    @property
    def _gripper1_to_plug(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._plug_xpos - self._eef1_xpos
    
    @property
    def _plug_to_socket(self):
        """
        Calculate vector from the left gripper to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._plug_xpos - self._socket_xpos

    @property
    def _plug_socket_alignment(self):
        """
        Calculate the alignment of the plug and socket.
        """
        plug_quat = self._plug_quat
        socket_quat = self._socket_quat
        return np.dot(plug_quat, socket_quat)