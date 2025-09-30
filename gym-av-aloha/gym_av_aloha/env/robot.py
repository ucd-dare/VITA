import numpy as np


class RobotArm:
    def get_joint_positions(self):
        raise NotImplementedError

    def set_joint_positions(self, joint_positions):
        raise NotImplementedError

    def get_gripper_position(self):
        raise NotImplementedError

    def set_gripper_position(self, gripper_position):
        raise NotImplementedError

    def set_joint_ctrl(self, joint_ctrl):
        raise NotImplementedError

    def get_joint_ctrl(self):
        raise NotImplementedError

    def set_gripper_ctrl(self, gripper_ctrl):
        raise NotImplementedError

    def get_gripper_ctrl(self):
        raise NotImplementedError


class SimRobotArm(RobotArm):
    def __init__(
        self,
        physics,
        joints,
        actuators,
        eef_site,
        has_gripper=False,
        gripper_joints=None,
        gripper_actuator=None,
    ):
        self.physics = physics
        self.joints = joints
        self.actuators = actuators
        self.eef_site = eef_site
        self.has_gripper = has_gripper
        self.gripper_joints = gripper_joints
        self.gripper_actuator = gripper_actuator

        self.joint_range = self.physics.bind(self.joints).range
        self.actuator_range = self.physics.bind(self.actuators).ctrlrange

        if has_gripper:
            self.gripper_range = self.physics.bind(self.gripper_actuator).ctrlrange
            self.gripper_norm_fn = lambda x: (x - self.gripper_range[0]) / (self.gripper_range[1] - self.gripper_range[0])
            self.gripper_unnorm_fn = lambda x: x * (self.gripper_range[1] - self.gripper_range[0]) + self.gripper_range[0]

    def get_eef_position(self):
        return self.physics.bind(self.eef_site).xpos.copy()
    
    def get_eef_rotation(self):
        return self.physics.bind(self.eef_site).xmat.reshape(3, 3).copy()
    
    def get_eef_pose(self):
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = self.get_eef_rotation()
        eef_pose[:3, 3] = self.get_eef_position()
        return eef_pose

    def get_joint_positions(self):
        return self.physics.bind(self.joints).qpos.copy()

    def set_joint_positions(self, joint_positions):
        # clip the joint positions to the joint range
        joint_positions = np.clip(joint_positions, self.joint_range[:, 0], self.joint_range[:, 1])
        self.physics.bind(self.joints).qpos = joint_positions

    def get_gripper_position(self):
        if not self.has_gripper:
            raise ValueError("Gripper not available")

        return self.gripper_norm_fn(self.physics.bind(self.gripper_joints[0]).qpos)

    def set_gripper_position(self, gripper_position):
        if not self.has_gripper:
            raise ValueError("Gripper not available")

        # clip to be between 0 and 1
        gripper_position = np.clip(gripper_position, 0, 1)
        self.physics.bind(self.gripper_joints).qpos = self.gripper_unnorm_fn(gripper_position)

    def set_joint_ctrl(self, joint_ctrl):
        # clip the joint control to the joint range
        joint_ctrl = np.clip(joint_ctrl, self.actuator_range[:, 0], self.actuator_range[:, 1])
        self.physics.bind(self.actuators).ctrl = joint_ctrl

    def get_joint_ctrl(self):
        return self.physics.bind(self.actuators).ctrl.copy()

    def set_gripper_ctrl(self, gripper_ctrl):
        if not self.has_gripper:
            raise ValueError("Gripper not available")

        # clip to be between 0 and 1
        gripper_ctrl = np.clip(gripper_ctrl, 0, 1)
        self.physics.bind(self.gripper_actuator).ctrl = self.gripper_unnorm_fn(gripper_ctrl)

    def get_gripper_ctrl(self):
        if not self.has_gripper:
            raise ValueError("Gripper not available")

        return self.gripper_norm_fn(self.physics.bind(self.gripper_actuator).ctrl)
