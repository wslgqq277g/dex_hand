"""
PointNet for point cloud feature extraction
Cited from Dexart
"""
import random
import torch
import torch.nn as nn

import sys, itertools, numpy as np
# from chamferdist import ChamferDistance

from icecream import ic, install

# install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')


# ic.disable()

def chamfer_loss(point_set1, point_set2):
    """
    # point_set1: (B, N, D)
    # point_set2: (B, M, D)
    """
    # 计算点集1中每个点到点集2的最小距离
    point_set1_expand = point_set1.unsqueeze(2)  # (B, N, 1, D)
    point_set2_expand = point_set2.unsqueeze(1)  # (B, 1, M, D)
    dist1 = torch.sum((point_set1_expand - point_set2_expand) ** 2, dim=-1)  # (B, N, M)
    min_dist1, _ = torch.min(dist1, dim=2)  # (B, N)

    # 计算点集2中每个点到点集1的最小距离
    dist2 = torch.sum((point_set1_expand - point_set2_expand) ** 2, dim=-1)  # (B, N, M)
    min_dist2, _ = torch.min(dist2, dim=1)  # (B, M)

    # 计算Chamfer距离
    chamfer_dist = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)  # (B,)
    chamfer_loss = torch.mean(chamfer_dist)

    return chamfer_loss


class ChamLoss:
    def __call__(self, coarse, fine, gt, alpha):
        return chamfer_loss(coarse, gt) + alpha * chamfer_loss(fine, gt)


class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        # print(f'PointNet')

        in_channel = point_channel
        # mlp_out_dim = 256 ori
        mlp_out_dim = 64
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),

        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        # local_feats = x
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]

        return x


class EncoderDecoder(nn.Module):
    def __init__(self, key_w):
        super(EncoderDecoder, self).__init__()
        kwargs = key_w
        self.cls = kwargs.cls
        self.class_num = kwargs.class_num
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 84
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 1374
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        self.pointnet = PointNet()
        # self.mid1=1024 ori
        self.mid1 = 256
        # self.mid2=512 ori
        self.mid2 = 256
        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(64, self.mid1),
            # nn.BatchNorm1d(self.mid1),
            nn.ReLU(),
            nn.Linear(self.mid1, self.mid1),
            # nn.BatchNorm1d(self.mid1),
            nn.ReLU(),
            nn.Linear(self.mid1, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(64 + 2 + 3, self.mid2, 1),
            nn.BatchNorm1d(self.mid2),
            nn.ReLU(),
            nn.Conv1d(self.mid2, self.mid2, 1),
            nn.BatchNorm1d(self.mid2),
            nn.ReLU(),
            nn.Conv1d(self.mid2, 3, 1))
        if self.cls:
            self.cls_layer = nn.Sequential(
                nn.Conv1d(84, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU())
            self.fc = nn.Linear(64, self.class_num)

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
            # ic(tensor.shape, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, x):
        feature = self.pointnet(x)
        ic(feature.shape)
        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)
        ic(coarse.shape)
        grid = self.build_grid(x.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        ic(point_feat.shape)
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])
        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        if self.cls:
            cls_output = self.cls_layer(coarse)  # (B,64,3)
            cls_output = torch.max(cls_output, dim=2)[0]
            logit = self.fc(cls_output)
            return feature, coarse, fine, logit
        else:
            # return feature, fine, 0
            return feature, coarse, fine, 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--log_dir", type=str, default='./Ex', help=" The log directory")
    parser.add_argument("--seed", type=int, default=2333, help=" Seed")

    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    # parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")

    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")

    # shareac
    parser.add_argument("--lr_ac", type=float, default=3e-4, help="Learning rate of critic")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")

    # loss/state coefficient
    parser.add_argument("--critic_coef", type=float, default=0.5, help="critic coef")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--cham_coef", type=float, default=1, help="Trick 5: policy entropy")
    parser.add_argument("--state_coef", type=float, default=1, help="Trick 5: policy entropy")

    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_visual_obs", type=bool, default=True, help="Trick 10: tanh activation function")
    # parser.add_argument("--cls", type=bool, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--path", type=str, default='', help="Trick 10: tanh activation function")

    parser.add_argument("--use_ori_obs", type=bool, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--cls", type=bool, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--class_num", type=int, default=9, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    model = EncoderDecoder(key_w=args)
    input_pc = torch.rand(1, 1200, 3)
    x = model(input_pc)
