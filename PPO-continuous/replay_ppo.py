import time
# import torch
# import sys
# import numpy as np
# import os
# from policy import TanhGaussianPolicy
# from env_wrappers import NormalizedBoxEnv
# import gym
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_util as ptu
# from hand_teleop.env.rl_env.inspire_relocate_env import InspireRelocateRLEnv
# from hand_teleop.env.rl_env.inspire_relocate_env_new_reward import InspireRelocateRLEnv as InspireRelocateRLEnvReward
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import gym
import argparse
# from normalization import Normalization, RewardScaling
# from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
# from ppo_continuous_pc import PPO_continuous_pc
from ppo_continuous_pc_vae import PPO_continuous_pc as PPO_continuous_pc
import os
from hand_teleop.env.rl_env.inspire_relocate_pc_env import InspireRelocateRLEnv

import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--log_dir", type=str, default='./Ex', help=" The log directory")
parser.add_argument("--seed", type=int, default=2333, help=" Seed")

parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=64,
                    help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")

parser.add_argument("--lr_ac", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--critic_coef", type=float, default=0.5, help="critic coef")

parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--cham_coef", type=float, default=1, help="Trick 5: policy entropy")
parser.add_argument("--state_coef", type=float, default=1, help="Trick 5: policy entropy")

parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
parser.add_argument("--use_visual_obs", type=float, default=True, help="Trick 10: tanh activation function")
parser.add_argument("--path", type=str, default='', help="Trick 10: tanh activation function")
parser.add_argument("--use_ori_obs", type=bool, default=False, help="Trick 10: tanh activation function")
parser.add_argument("--cls", type=bool, default=False, help="Trick 10: tanh activation function")
parser.add_argument("--class_num", type=int, default=9, help="Trick 10: tanh activation function")

args = parser.parse_args()

# -------parameters-------
render = True
directory = './2333pc_1/best'
# directory = './EX_new_reward/SAC_2333/final'

ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

# env = InspireRelocateRLEnv(
# rotation_reward_weight=0,
#                                robot_name="inspire_hand_free",
#                                object_name="mustard_bottle",
#                                 use_gui=False,
#                                 frame_skip=10,
#                                 use_visual_obs=True,
#                                 no_rgb=False)
#     use_gui=True, robot_name="inspire_hand_free",
#                            object_name="mustard_bottle", frame_skip=10,
#                            # use_visual_obs=False
# use_visual_obs=True,
#                                 no_rgb=False
#                            )
env = InspireRelocateRLEnv(rotation_reward_weight=0,
                           robot_name="inspire_hand_free",
                           object_name="mustard_bottle",
                           use_gui=True,
                           frame_skip=10,
                           use_visual_obs=True,
                           no_rgb=False)
args.action_dim = 12
from dexpoint.real_world import task_setting

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# args.state_dim = env.observation_space.shape[0]

# #print(type(env.observation_space) is gym.spaces.Dict,'kk222222222kk')

# #print(env.observation_space.spaces['state'].shape,'kk444444444kk')
# assert False
args.action_dim = env.real_dof  # 12
args.max_action = float(env.action_space.high[0])
# args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
args.max_episode_steps = 300000
# print("env={}".format(env_name))
# #print("state_dim={}".format(args.state_dim))
# print("action_dim={}".format(args.action_dim))
# print("max_action={}".format(args.max_action))
# print("max_episode_steps={}".format(args.max_episode_steps))

env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
# Specify observation modality
env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

# print(env.observation_space.keys(), 'kk333333333kk')
# for k,v in env.observation_space.items():
#     print(k,v.shape)
args.state_dim = env.observation_space['state'].shape[0] + env.observation_space['oracle_state'].shape[0] + \
                 64
print(env.observation_space['state'].shape[0], 'state_dim')
print(env.observation_space['oracle_state'].shape[0], 'oracle_state')
# breakpoint()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# args.state_dim = env.observation_space.shape[0]
# args.action_dim = env.action_space.shape[0]
print(args.state_dim, 'state_dim')
print(env.observation_space, 'observation')
print(type(env.observation_space))
# assert False
args.max_action = float(env.action_space.high[0])
# args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
args.max_episode_steps = 300000

# obs_dim = env.observation_space.low.size
# action_dim = env.action_space.low.size
action_dim = env.real_dof
args.action_dim = 12
args.action_dim = env.real_dof  # 12
M = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# policy = TanhGaussianPolicy(
#     obs_dim=obs_dim,
#     action_dim=action_dim,
#     hidden_sizes=[M, M],
# ).to(device)
camera_matrix = env.cameras['relocate'].get_intrinsic_matrix()
args.camera_matrix = camera_matrix

directory = os.path.join(os.getcwd(), '2333pc/best/')
agent = PPO_continuous_pc(args)
print(agent)
# breakpoint()
from torchsummary import summary

# summary(model=agent)
# print(os.path.join(directory, 'Ex2333/best/actor.pth'))
# assert False

# agent.actor.load_state_dict(torch.load(\
#     os.path.join(directory,'actor.pth')))
# agent.critic.load_state_dict(torch.load(\
#     os.path.join(directory, 'critic.pth')))
agent.load_state_dict(torch.load( \
    os.path.join(directory, 'AC_vae.pth')))
# agent.point_net.load_state_dict(torch.load(\
#     os.path.join(directory,'pointnet.pth')))
# policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth'), map_location ='cpu'))
# policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth')))
# env.seed(2333)
seed = 2333
env.seed(seed)
env.action_space.seed(seed)

# env_evaluate.seed(seed)
# env_evaluate.action_space.seed(seed)
# state = env.reset()

from sapien.utils import Viewer

base_env = env
viewer = Viewer(base_env.renderer)
viewer.set_scene(base_env.scene)
base_env.viewer = viewer
base_env.viewer.set_camera_xyz(x=-1, y=0, z=1)
base_env.viewer.set_camera_rpy(r=0, p=-np.arctan2(4, 2), y=0)

env.render()

step = 0
eps_reward = 0
t = 0
if args.use_reward_norm:  # Trick 3:reward normalization
    reward_norm = Normalization(shape=1)
elif args.use_reward_scaling:  # Trick 4:reward scaling
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
# if args.use_state_norm:
#     state = state_norm(state, update=False)  # During the evaluating,update=False
# if args.use_state_norm:
#     s = state_norm(s)
if args.use_reward_scaling:
    reward_scaling.reset()


def depth_to_point_cloud(depth_map, camera_matrix):
    # 获取深度图像的高度和宽度
    height, width = depth_map.shape

    # 创建像素坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coordinates = np.vstack((x.flatten(), y.flatten(), np.ones(width * height)))

    # 计算相机内参矩阵的逆矩阵
    inv_camera_matrix = np.linalg.inv(camera_matrix)

    # 将像素坐标转换为相机坐标
    camera_coordinates = np.dot(inv_camera_matrix, pixel_coordinates)

    # 将相机坐标与深度值相乘，得到点云坐标
    point_cloud = camera_coordinates * depth_map.flatten()

    return point_cloud


def evaluate_policy(args, env, agent, state_norm):
    times = 3

    evaluate_reward = 0
    pc_step = 0
    camera_matrix = env.cameras['relocate'].get_intrinsic_matrix()

    for _ in range(times):
        s = env.reset(seed=2333)
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            pc_step += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            a = action
            # print(a.shape,'a')
            a_apply = np.zeros(env.robot.dof)
            a_apply[env.total_activate_joint_index] = a

            if pc_step < 20:
                depth_map = s['relocate-depth'].squeeze(2)
                mask = (s['relocate-segmentation'][..., 0] == 6)
                depth_map = np.multiply(depth_map, mask)

                # 将深度图转换为点云
                point_cloud = depth_to_point_cloud(depth_map, camera_matrix)
                point_cloud_new = point_cloud

                point_cloud = point_cloud_new.transpose(1, 0)
                true_index=np.nonzero(np.any(point_cloud,axis=1))[0]
                point_cloud=point_cloud[true_index]

                fig = plt.figure(figsize=(10, 10))
                print(point_cloud.shape)

                sample = point_cloud
                ax1 = fig.add_subplot(141, projection="3d")
                ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                ax1.set_title("Original Point Cloud_view1")

                plt.savefig(f'./result/vis/{pc_step}original_plot.png')
            # plt.close()
            # s_, r, done, _ = env.step(action)
            s_, r, done, _ = env.step(a_apply)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            if render:
                env.render()
        evaluate_reward += episode_reward

    return evaluate_reward / times


# print(next(agent.critic.parameters()).data[0], '22')

evaluate_reward = evaluate_policy(args, env, agent, state_norm)
print(evaluate_reward, 'evaluate_reward')
# times = 3
# evaluate_reward = 0
#
# for _ in range(times):
#     state = env.reset()
#     terminal = False
#     if args.use_state_norm:
#         state = state_norm(state, update=False)  # During the evaluating,update=False
#     episode_reward = 0
#
#     while not terminal:
#         t += 1
#         # print(t)
#         state = np.array(state)
#         # a, a_logprob = agent.choose_action(state)  # Action and the corresponding log probability
#         # print(a.shape,'a')
#         a = agent.evaluate(state)  # We use the deterministic policy during the evaluating
#         if args.policy_dist == "Beta":
#             action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
#         else:
#             action = a
#         # a = policy.get_evaluate_action(state)
#         a_apply = np.zeros(env.robot.dof)
#         # print(env.robot.dof,'dof')
#         # print(env.action_space.shape[0],'terst')
#         a_apply[env.total_activate_joint_index] = a
#         # a_apply[env.total_activate_joint_index] = a.detach().cpu().numpy()
#
#         # next_state, r, terminal, _ = env.step(a)
#         # next_state, r, terminal, _ = env.step(action)
#         next_state, r, terminal, _ = env.step(a_apply)
#         if args.use_state_norm:
#             s_ = state_norm(next_state, update=False)
#
#         # time.sleep(1 / 10)
#         # if render:
#         #     env.render()
#         # print(r)
#         episode_reward += r
#         # print(terminal,'terminal')
#         step += 1
#         state = s_
#         # if terminal:
#         #     break
#     evaluate_reward+=episode_reward
#     print(episode_reward,'eps_reward')
# print(evaluate_reward/times,'evaluate_reward')
