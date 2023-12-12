from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.inspire_base import BaseRLEnv
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.real_world import lab

OBJECT_LIFT_LOWER_LIMIT = -0.03


class InspireRelocateRLEnv(RelocateEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="inspire_hand_free", constant_object_state=False,
                 rotation_reward_weight=0, object_category="YCB", object_name="tomato_soup_can", object_scale=1.0,
                 randomness_scale=1, friction=1, object_pose_noise=0.01, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_category, object_name, object_scale, randomness_scale, friction,
                         **renderer_kwargs)
        # Base class
        self.setup(robot_name)
        self.rotation_reward_weight = rotation_reward_weight

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name

        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        
        finger_contact_link_name = [
            "thtip", "righthand_thumb4", "righthand_thumb3",
            "fftip", "righthand_index2", "righthand_index1", 
            "mftip", "righthand_middle2", "righthand_middle1",
            "rftip", "righthand_ring2", "righthand_ring1",
            "lftip", "righthand_little2", "righthand_little1"
        ]
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]
        self.finger_contact_links = [self.robot.get_links()[robot_link_names.index(name)] for name in
                                     finger_contact_link_name]
        self.finger_contact_ids = np.array([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5])

        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])
        self.finger_reward_scale = np.ones(len(self.finger_tip_links)) * 0.01
        self.finger_reward_scale[0] = 0.04

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.object_in_tip = np.zeros([len(finger_tip_names), 3])
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0

        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # five tip, palm

        self.base_frame_pos = np.zeros(3)

        # real DOF
        self.real_dof = 12
        # ------------------------don't know----------------
        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

        self.object_pose_noise = object_pose_noise
        self.constant_object_state = constant_object_state
        # --------------------------------------------------

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        check_contact_links = self.finger_contact_links + [self.palm_link]
        contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
        self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = self.palm_pose.p - self.base_frame_pos
        self.object_in_tip = self.object_pose.p[None, :] - self.finger_tip_pos
        self.object_lift = self.object_pose.p[2] - self.object_height
        self.target_in_object = self.target_pose.p - self.object_pose.p
        self.target_in_object_angle[0] = np.arccos(
            np.clip(np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))

    # new
    # def get_oracle_state(self):
    #     object_pos = self.object_pose.p
    #     object_quat = self.object_pose.q
    #     object_pose_vec = np.concatenate([object_pos - self.base_frame_pos, object_quat])
    #     robot_qpos_vec = self.robot.get_qpos()
    #     return np.concatenate([
    #         robot_qpos_vec, self.palm_pos_in_base,  # dof + 3
    #         object_pose_vec, self.object_in_tip.flatten(), self.robot_object_contact,  # 7 + 12 + 5
    #         self.target_in_object, self.target_pose.q, self.target_in_object_angle  # 3 + 4 + 1
    #     ])
    
    # old
    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate(
            [robot_qpos_vec, object_pose_vec, palm_v, palm_w, object_in_palm, target_in_palm, target_in_object,
             self.target_pose.q, np.array([theta])])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    # def get_reward(self, action):
    #     object_pose = self.manipulated_object.get_pose()
    #     palm_pose = self.palm_link.get_pose()
    #     is_contact = self.check_contact(self.robot_collision_links, [self.manipulated_object])

    #     reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
    #     if is_contact:
    #         reward += 0.1
    #         lift = min(object_pose.p[2], self.target_pose.p[2]) - self.object_height
    #         lift = max(lift, 0)
    #         reward += 5 * lift
    #         if lift > 0.015:
    #             reward += 2
    #             obj_target_distance = min(np.linalg.norm(object_pose.p - self.target_pose.p), 0.5)
    #             reward += -1 * min(np.linalg.norm(palm_pose.p - self.target_pose.p), 0.5)
    #             reward += -3 * obj_target_distance  # make object go to target

    #             if obj_target_distance < 0.1:
    #                 reward += (0.1 - obj_target_distance) * 20
    #                 theta = np.arccos(
    #                     np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
    #                 reward += max((np.pi / 2 - theta) * self.rotation_reward_weight, 0)
    #                 if theta < np.pi / 4 and self.rotation_reward_weight >= 1e-6:
    #                     reward += (np.pi / 4 - theta) * 6 * self.rotation_reward_weight

    #     return reward

    def get_reward(self, action):
        finger_object_dist = np.linalg.norm(self.object_in_tip, axis=1, keepdims=False)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        reward = np.sum(1.0 / (0.06 + finger_object_dist) * self.finger_reward_scale)
        # at least one tip and palm or two tips are contacting obj. Thumb contact is required.
        is_contact = np.sum(self.robot_object_contact) >= 2

        if is_contact:
            reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 10 * lift
            if lift > 0.02:
                reward += 1
                target_obj_dist = np.linalg.norm(self.target_in_object)
                reward += 1.0 / (0.04 + target_obj_dist)

                if target_obj_dist < 0.1:
                    theta = self.target_in_object_angle[0]
                    reward += 4.0 / (0.4 + theta) * self.rotation_reward_weight

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * -0.01
        # controller_penalty = (self.cartesian_error ** 2) * -1e3
        # return (reward + action_penalty + controller_penalty) / 10
        return (reward + action_penalty) / 10

    




    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        # if not self.is_robot_free:
        #     qpos = np.zeros(self.robot.dof)
        #     xarm_qpos = self.robot_info.arm_init_qpos
        #     qpos[:self.arm_dof] = xarm_qpos
        #     self.robot.set_qpos(qpos)
        #     self.robot.set_drive_target(qpos)
        #     init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        #     init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        # else:
        # init_pose = sapien.Pose(np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        init_pose = sapien.Pose(np.array([-0.4, 0, 0.15]), transforms3d.euler.euler2quat(np.pi/2, 0, np.pi/4))

        self.robot.set_pose(init_pose)
        self.reset_internal()
        # q_initial = np.array([ 0, 0, 0, 0, 0, 0,
        #                         0, 0.396, 0, 0.396, 0, 0.396,
        #                         0, 0.396, 0.36, -0.48, 0.2393, -0.16])

        q_initial = np.array([ 0, 0, 0, 0, 0, 0,
                        0, 0.396, 0, 0.396, 0, 0.396,
                        0, 0.396, -1.24, -0.48, 0.2393, -0.16])


        self.robot.set_qpos(q_initial)
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return self.manipulated_object.pose.p[2] - self.object_height < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 250


def main_env():
    env = InspireRelocateRLEnv(use_gui=True, robot_name="inspire_hand_free",
                        object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    base_env.viewer.set_camera_xyz(x=-1, y=0, z=1)
    base_env.viewer.set_camera_rpy(r=0, p=-np.arctan2(4, 2), y=0)

    import time

    # for i in range(100):
    #     env.reset()
    #     env.render()
    #     time.sleep(1)
    # viewer.toggle_pause(True)
    for i in range(5000):
        # print(i)
        # print(robot_dof)
        action = np.zeros(robot_dof)
        # action[12] = -0.5
        # action[14] = -0.5
        # action[15] = -0.5
        action = np.array([0, 0, 0, 0, -0.01, 0.0,
                           1, 0, 1, 0, 1, 0,
                           1, 0, 1, 1, 0, 0])
        # assert 0
        # action[1] = 0.005
        # action[14] = 0.5
        # self.activate_joint_index = [6, 8, 10, 12, 14, 15]
        # self.coupled_joint_index = [7, 9, 11, 13, 16, 17]
        # action[1] =0.01
        # action[[6, 8, 10, 12]] = 0
        # action[14] = 0.1
        # action[15] = 0
        # action[12] = -1
        # assert 0
        # action[2] =0.01
        # action[1] = i * 0.02
        obs, reward, done, _ = env.step(action)
        # print(reward)
        # print('obs:')
        print(obs[:robot_dof])
        env.render()


    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
