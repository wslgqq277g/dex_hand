"""
This script is used to create the dataset for the point cloud based RL.
"""
import torch
import numpy as np
from scipy.linalg import expm
import argparse
import os
from hand_teleop.env.rl_env.inspire_relocate_pc_env import InspireRelocateRLEnv
from sapien.core import Pose
from dexpoint.real_world import task_setting
from model.VAE import depth_to_point_cloud
import os.path as osp
import matplotlib.pyplot as plt


def add_noise_to_transformation_matrix(num, noise_scale):
    """
    为4x4转移矩阵添加一个扰动。

    参数:
        transformation_matrix (numpy.ndarray): 原始的4x4转移矩阵。
        noise_scale (float): 扰动的尺度。

    返回:
        numpy.ndarray: 添加扰动后的4x4转移矩阵。
    """
    p_list = []
    for i in range(num):
        noise = np.random.normal(loc=0, scale=noise_scale, size=(4, 4))
        noise[3, :] = [0, 0, 0, 1]  # 保持转移矩阵的最后一行不变
        p_list.append(noise)
    return p_list


def add_rotation_noise_to_transformation_matrix(num, rotation_noise_scale):
    """
    为4x4转移矩阵添加旋转的扰动。

    参数:
        transformation_matrix (numpy.ndarray): 原始的4x4转移矩阵。
        rotation_noise_scale (float): 旋转扰动的尺度。

    返回:
        numpy.ndarray: 添加旋转扰动后的4x4转移矩阵。
    """
    perturb_list = []
    for i in range(num):
        rotation_noise = np.random.normal(loc=0, scale=rotation_noise_scale, size=(3,))
        rotation_matrix = expm(np.cross(np.eye(3), rotation_noise))
        perturb_list.append(rotation_matrix)
    return perturb_list


camera_config = task_setting.CAMERA_CONFIG["relocate"]
came_pose = camera_config["relocate"]['pose2']


def main(args, seed):
    object_name = args.object_name
    # camera_config = task_setting.CAMERA_CONFIG["relocate"]
    # came_pose = camera_config["relocate"]['pose2']

    num_train = 100
    num_val = 25
    total_num = num_train + num_val
    rotation_noise_scale = 0.01

    noise_scale = 0.01
    all_noise = add_noise_to_transformation_matrix(total_num * 100, noise_scale)
    r_noise = add_rotation_noise_to_transformation_matrix(total_num * 20, rotation_noise_scale)
    point_num = 0
    step = 0
    train_num = 0
    val_num = 0
    while point_num < total_num:
        env = InspireRelocateRLEnv(rotation_reward_weight=0,
                                   robot_name="inspire_hand_free",
                                   object_name=object_name,
                                   use_gui=False,
                                   frame_skip=10,
                                   use_visual_obs=True,
                                   no_rgb=False)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.action_dim = env.real_dof  # 12
        args.max_action = float(env.action_space.high[0])
        # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        args.max_episode_steps = 300000
        # print(type(came_pose))
        # assert False
        perturbed_matrix = came_pose + all_noise[step]
        perturbed_matrix[:3, :3] = np.dot(perturbed_matrix[:3, :3], r_noise[step])

        # print(perturbed_matrix,f'step{step}')
        camera_config["relocate"]['pose2'] = Pose.from_transformation_matrix(perturbed_matrix)

        camera_config["relocate"]['pose'] = camera_config["relocate"]['pose1'] * camera_config["relocate"]['pose2']

        env.setup_camera_from_config(camera_config)

        # Specify observation modality
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

        args.state_dim = env.observation_space['state'].shape[0] + env.observation_space['oracle_state'].shape[0] + \
                         256
        # breakpoint()
        save_dir = os.path.join(os.getcwd(), '{}'.format(seed) + 'pc')
        obs = env.reset()
        env.scene.update_render()
        depth_map = obs['relocate-depth'].squeeze(2)

        # 示例相机内参矩阵
        camera_matrix = env.cameras['relocate'].get_intrinsic_matrix()
        mask = (obs['relocate-segmentation'][..., 0] == 6)
        depth_map = np.multiply(depth_map, mask)
        # 将深度图转换为点云
        point_cloud = depth_to_point_cloud(depth_map, camera_matrix)
        point_cloud_new = point_cloud
        # # 进行聚类
        point_cloud = point_cloud_new.transpose(1, 0)
        true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
        point_cloud = point_cloud[true_index]
        # point_cloud = point_cloud[:4000, :]
        # breakpoint()
        if os.path.exists(f'./dataset/{object_name}') == False:
            os.makedirs(f'./dataset/{object_name}')
        if os.path.exists(f'./dataset/{object_name}/train') == False:
            os.makedirs(f'./dataset/{object_name}/train')
        if os.path.exists(f'./dataset/{object_name}/val') == False:
            os.makedirs(f'./dataset/{object_name}/val')
        num_point = point_cloud.shape[0]
        print(num_point)
        if num_point<1200:
            del env
            torch.cuda.empty_cache()
            continue
        step += 1
        point_list = np.linspace(0, num_point - 1, num_point, dtype=int)
        point_list = np.random.choice(point_list, 1200)
        point_cloud = point_cloud[point_list]
        point_num += 1
        if train_num < num_train:
            train_num += 1
            np.save(f'./dataset/{object_name}/train/{train_num}.npy', point_cloud)
            if train_num == 1:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection="3d")
                ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
                ax1.set_title("Original Point Cloud")
                if os.path.exists('./dataset_vis/') == False:
                    os.mkdir('./dataset_vis/')
                plt.savefig(f'./dataset_vis/{object_name}_train_pc.png')
                plt.close()
        else:
            val_num += 1
            np.save(f'./dataset/{object_name}/val/{val_num}.npy', point_cloud)
            if val_num == 1:
                fig = plt.figure()
                ax1 = fig.add_subplot(121, projection="3d")
                ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
                ax1.set_title("Original Point Cloud")
                plt.savefig(f'./dataset_vis/{object_name}_val_pc.png')
                plt.close()

        def are_matrices_equal(matrix1, matrix2):
            if matrix1.shape != matrix2.shape:
                return False

            return np.allclose(matrix1, matrix2)

        if train_num > 1:
            if are_matrices_equal(before_point_cloud, point_cloud):
                assert False, 'not accurate update'
        del env
        torch.cuda.empty_cache()
        before_point_cloud = point_cloud
        # _= env.cameras.pop('relocate')

    print('dataset create done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--log_dir", type=str, default='./Ex', help=" The log directory")
    parser.add_argument("--seed", type=int, default=2333, help=" Seed")

    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")

    # shareac
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

    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_visual_obs", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--path", type=str, default='', help="Trick 10: tanh activation function")
    parser.add_argument("--object_name", type=str, default='mustard_bottle', help="Trick 10: tanh activation function")

    root_dir = os.getcwd()
    ojbect_dir = osp.join(root_dir, 'assets', 'ycb', 'visual')
    object_list = os.listdir(ojbect_dir)
    object_list = [object[4:] for object in object_list]
    object_list = object_list[::-1]
    args = parser.parse_args()
    if args.object_name == '':
        assert False, 'please input object name'
    if len(object_list) != 0:
        for object in object_list:
            args.object_name = object
            main(args, seed=args.seed)
            print(f'{object}s dataset creation is finished.')
    # args.use_adv_norm=False
