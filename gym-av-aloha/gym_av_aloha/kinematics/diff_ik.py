import numpy as np
from numba import jit, prange
from gym_av_aloha.utils.transform_utils import wxyz_to_xyzw, quat2mat, angular_error, adjoint, exp2mat
from gym_av_aloha.env.sim_env import AVAlohaEnv
from gym_av_aloha.env.sim_config import SIM_DT
import mujoco

@jit(nopython=True, fastmath=True, cache=True)
def run_diff_ik(
    old_q, 
    target_pos, 
    target_mat, 
    iterations,
    integration_dt,
    k_pos,
    k_ori,
    diag,
    eye,
    k_null,
    q0,
    max_angvel,
    joint_limits,
    joint_p,
    # jac and fk fn
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
    
    def jac_fn(theta):
        # screw axis at rest place
        S = np.hstack((w0, v0)) 
        J = np.zeros((6, len(theta)))
        Ts = np.eye(4)

        # compute each column of the Jacobian
        for i in prange(len(theta)):
            J[:, i] = adjoint(Ts) @ S[i,:]
            Ts = np.dot(Ts, exp2mat(w0[i], v0[i], theta[i]))

        # swap jacp and jacr
        J = J[np.array((3,4,5,0,1,2)),:]

        return J
    
    q = old_q.copy()

    for _ in range(iterations):
        current_pose = fk_fn(q)
        current_pos = current_pose[:3, 3]
        current_mat = current_pose[:3, :3]

        twist = np.zeros(6)
        dx = target_pos - current_pos
        twist[:3] = k_pos * dx / integration_dt
        dr = angular_error(target_mat, current_mat)
        twist[3:] = k_ori *dr / integration_dt

        # Jacobian.
        jac = jac_fn(q)

        # Damped least squares.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

        # Null space control.
        dq += (eye - np.linalg.pinv(jac) @ jac) @ (k_null * (q0 - q))

        # Limit joint velocity.
        dq = np.clip(dq, -max_angvel, max_angvel)

        # integrate
        q = q + dq * integration_dt

    # Limit joint position.
    q = np.clip(q, joint_limits[:,0], joint_limits[:,1])

    new_q = old_q + joint_p * (q - old_q)
    
    return new_q
    
class DiffIKConfig():
    k_pos=0.9
    k_ori=0.9
    damping=1.0e-1
    k_null=np.array([50.0, 10.0, 10.0, 50.0, 10.0, 10.0, 5.0])
    q0=np.array(AVAlohaEnv.MIDDLE_POSE)
    max_angvel=3.14
    integration_dt=SIM_DT
    iterations=10
    joint_p=0.9

class DiffIK():
    def __init__(
        self, 
        config: DiffIKConfig,
        physics,
        joints,
        eef_site,
    ):
        self.physics = physics
        self.joints = joints
        self.eef_site = eef_site
        self.config = config

        self.diag = np.ascontiguousarray(config.damping * np.eye(6))
        self.eye = np.ascontiguousarray(np.eye(len(self.joints)))
        self.joint_limits = physics.bind(self.joints).range.copy()

        physics.bind(joints).qpos = np.zeros(len(joints))
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
        self.w0 = physics.bind(joints).xaxis.copy()
        self.p0 = physics.bind(joints).xanchor.copy()
        self.v0 = -np.cross(self.w0, self.p0)
        self.site0 = np.eye(4)
        self.site0[:3, :3] = physics.bind(eef_site).xmat.reshape(3,3).copy()
        self.site0[:3, 3] = physics.bind(eef_site).xpos.copy()

    def run(self, q, target_pos, target_mat):
        return run_diff_ik(
            old_q=q,
            target_pos=target_pos,
            target_mat=target_mat,
            iterations=self.config.iterations,
            integration_dt=self.config.integration_dt,
            k_pos=self.config.k_pos,
            k_ori=self.config.k_ori,
            diag=self.diag,
            eye=self.eye,
            k_null=self.config.k_null,
            q0=self.config.q0,
            max_angvel=self.config.max_angvel,
            joint_limits=self.joint_limits,
            joint_p=self.config.joint_p,
            # jac and fk fn
            w0=self.w0,
            v0=self.v0,
            site0=self.site0,
        )

if __name__ == '__main__':
    from dm_control import mjcf
    from gym_av_aloha.env.sim_config import XML_DIR, MIDDLE_ACTUATOR_NAMES, MIDDLE_JOINT_NAMES, MIDDLE_EEF_SITE
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

    xml_path = os.path.join(XML_DIR, f'av_arm.xml')
    mjcf_root = mjcf.from_path(xml_path)  
    mjcf_root.option.timestep = PHYSICS_DT  
    
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) 

    middle_joints = [mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
    middle_actuators = [mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
    middle_eef_site = mjcf_root.find('site', MIDDLE_EEF_SITE)
    mocap = mjcf_root.find('body', MOCAP_NAME)

    # set up controllers
    config = DiffIKConfig()
    middle_controller = DiffIK(config, physics, middle_joints, middle_eef_site)

    physics.bind(middle_joints).qpos = AVAlohaEnv.MIDDLE_POSE
    physics.bind(middle_actuators).ctrl = AVAlohaEnv.MIDDLE_POSE
    physics.bind(mocap).mocap_pos = physics.bind(middle_eef_site).xpos
    physics.bind(mocap).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(middle_eef_site).xmat.reshape(3,3)))

    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mocap_pos = physics.bind(mocap).mocap_pos
            mocap_rot = quat2mat(wxyz_to_xyzw(physics.bind(mocap).mocap_quat))
            q = physics.bind(middle_joints).qpos
            start = time.time()
            physics.bind(middle_actuators).ctrl = middle_controller.run(q, mocap_pos, mocap_rot)
            print("Time taken: ", time.time() - start)
            physics.step(nstep=PHYSICS_ENV_STEP_RATIO)
            viewer.sync()

            time_until_next_step = DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  