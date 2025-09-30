import numpy as np
from numba import jit, prange
from gym_av_aloha.utils.transform_utils import wxyz_to_xyzw, quat2mat, angular_error, within_pose_threshold, exp2mat
import mujoco

@jit(nopython=True, fastmath=True, cache=True)
def run_grad_ik(
    q_start,
    target_pos,
    target_mat,
    # gradient descent params
    max_iterations,
    step_size,
    min_cost_delta,
    joint_limits,
    joint_p,
    # cost fn
    position_weight,
    rotation_weight,
    joint_center_weight,
    joint_centers,
    joint_displacement_weight,
    # solution fn
    position_threshold,
    rotation_threshold,
    # fk fn
    w0,
    v0,
    site0,
):
    def fk_fn(theta):
        M = np.eye(4)
        M[:,:] = site0
        for i in prange(len(theta)-1, -1, -1):
            T = exp2mat(w0[i], v0[i], theta[i])
            M = np.dot(T, M)
        return M
    
    def cost_fn(q, q_start, target_xpos, target_xmat): 
        current_pose = fk_fn(q)
        current_xpos = current_pose[:3, 3]
        current_xmat = current_pose[:3, :3]
        cost = 0.0
        # position cost
        cost += (position_weight * np.linalg.norm(target_xpos - current_xpos))**2
        # rotation cost
        cost += (rotation_weight * np.linalg.norm(angular_error(target_xmat, current_xmat)))**2
        # center joints cost
        cost += np.sum( (joint_center_weight * (q - joint_centers)) ** 2 )
        # minimal displacement cost
        cost += np.sum((joint_displacement_weight * (q - q_start))**2)
        return cost
    
    def solution_fn(q, q_start, target_xpos, target_xmat):
        current_pose = fk_fn(q)
        current_xpos = current_pose[:3, 3]
        current_xmat = current_pose[:3, :3]
        return within_pose_threshold(
            current_xpos, 
            current_xmat, 
            target_xpos, 
            target_xmat, 
            position_threshold, 
            rotation_threshold
        )

    init_cost = cost_fn(q_start, q_start, target_pos, target_mat)
    gradient = np.zeros(len(q_start))
    working = q_start.copy()
    local = q_start.copy()
    best = q_start.copy()
    local_cost = init_cost
    best_cost = init_cost
    
    previous_cost = 0.0
    for i in prange(max_iterations):
        count = len(local)

        for i in prange(count):
            working[i] = local[i] - step_size
            p1 = cost_fn(working, q_start, target_pos, target_mat)

            working[i] = local[i] + step_size
            p3 = cost_fn(working, q_start, target_pos, target_mat)

            working[i] = local[i]

            gradient[i] = p3 - p1

        sum_gradients = np.sum(np.abs(gradient)) + step_size
        f = step_size / sum_gradients
        gradient *= f

        working = local - gradient
        p1 = cost_fn(working, q_start, target_pos, target_mat)

        working = local + gradient
        p3 = cost_fn(working, q_start, target_pos, target_mat)
        p2 = 0.5 * (p1 + p3)

        cost_diff = 0.5 * (p3 - p1)
        joint_diff = p2 / cost_diff if np.isfinite(cost_diff) and cost_diff != 0.0 else 0.0

        working = local - gradient * joint_diff
        working = np.clip(working, joint_limits[:, 0], joint_limits[:, 1])

        local[:] = working
        local_cost = cost_fn(local, q_start, target_pos, target_mat)

        if local_cost < best_cost:
            best[:] = local
            best_cost = local_cost

        if solution_fn(local, q_start, target_pos, target_mat):
            break

        if abs(local_cost - previous_cost) <= min_cost_delta:
            break

        previous_cost = local_cost

    new_q = q_start + joint_p * (best - q_start)

    return new_q

class GradIKConfig:
    step_size = 0.0001
    min_cost_delta = 1.0e-12
    max_iterations = 50
    position_weight = 500.0
    rotation_weight = 100.0
    joint_center_weight = np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0])
    joint_displacement_weight = np.array(6*[50.0])
    position_threshold = 0.001
    rotation_threshold = 0.001
    joint_p = 0.9

class GradIK:
    def __init__(self, config: GradIKConfig, physics, joints, eef_site):
        self.config = config
        self.physics = physics
        self.joints = joints
        self.eef_site = eef_site

        self.num_joints = len(self.joints)
        self.joint_limits = self.physics.bind(self.joints).range.copy()
        self.joint_centers = 0.5 * (self.joint_limits[:, 0] + self.joint_limits[:, 1])
        self.half_ranges = 0.5 * (self.joint_limits[:, 1] - self.joint_limits[:, 0])

        physics.bind(joints).qpos = np.zeros(len(joints))
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
        self.w0 = physics.bind(joints).xaxis.copy()
        self.p0 = physics.bind(joints).xanchor.copy()
        self.v0 = -np.cross(self.w0, self.p0)
        self.site0 = np.eye(4)
        self.site0[:3, :3] = physics.bind(eef_site).xmat.reshape(3,3).copy()
        self.site0[:3, 3] = physics.bind(eef_site).xpos.copy()

    def run(self, q, target_pos, target_mat):        
        return run_grad_ik(
            q_start=q,
            target_pos=target_pos,
            target_mat=target_mat,
            max_iterations=self.config.max_iterations,
            step_size=self.config.step_size,
            min_cost_delta=self.config.min_cost_delta,
            joint_limits=self.joint_limits,
            joint_p=self.config.joint_p,
            # cost fn
            position_weight=self.config.position_weight,
            rotation_weight=self.config.rotation_weight,
            joint_center_weight=self.config.joint_center_weight,
            joint_centers=self.joint_centers,
            joint_displacement_weight=self.config.joint_displacement_weight,
            # solution fn
            position_threshold=self.config.position_threshold,
            rotation_threshold=self.config.rotation_threshold,
            # fk fn
            w0=self.w0,
            v0=self.v0,
            site0=self.site0,
        )

if __name__ == '__main__':
    from dm_control import mjcf
    from gym_av_aloha.env.sim_config import XML_DIR, LEFT_ACTUATOR_NAMES, LEFT_JOINT_NAMES, LEFT_EEF_SITE
    from gym_av_aloha.env.sim_env import AVAlohaEnv
    import mujoco.viewer
    import time
    import os
    from gym_av_aloha.utils.transform_utils import mat2quat, xyzw_to_wxyz

    MOCAP_NAME = "target"
    PHYSICS_DT=0.002
    DT = 0.04
    PHYSICS_ENV_STEP_RATIO = int(DT/PHYSICS_DT)
    DT = PHYSICS_DT * PHYSICS_ENV_STEP_RATIO

    xml_path = os.path.join(XML_DIR, f'manipulator_arm.xml')
    mjcf_root = mjcf.from_path(xml_path)  
    mjcf_root.option.timestep = PHYSICS_DT  
    
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) 

    left_joints = [mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES][:6]
    left_actuators = [mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES][:6]
    left_eef_site = mjcf_root.find('site', LEFT_EEF_SITE)
    mocap = mjcf_root.find('body', MOCAP_NAME)

    # set up controllers
    config = GradIKConfig()
    left_controller = GradIK(
        config=config,
        physics=physics,
        joints=left_joints,
        eef_site=left_eef_site,
    )

    physics.bind(left_joints).qpos = AVAlohaEnv.LEFT_POSE
    physics.bind(left_actuators).ctrl = AVAlohaEnv.LEFT_POSE
    physics.bind(mocap).mocap_pos = physics.bind(left_eef_site).xpos
    physics.bind(mocap).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(left_eef_site).xmat.reshape(3,3)))


    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        while viewer.is_running():
            step_start = time.time()
            joints = physics.bind(left_joints).qpos
            mocap_pos = physics.bind(mocap).mocap_pos
            mocap_rot = quat2mat(wxyz_to_xyzw(physics.bind(mocap).mocap_quat))
            start = time.time()
            physics.bind(left_actuators).ctrl = left_controller.run(joints, mocap_pos, mocap_rot)
            print("Time taken: ", time.time() - start)
            physics.step(nstep=PHYSICS_ENV_STEP_RATIO)
            viewer.sync()

            time_until_next_step = DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  