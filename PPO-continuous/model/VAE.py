import torch.nn as nn
from typing import List, Optional, Tuple
# from maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfoDex, getPointNetWithInstanceInfo, \
#     getSparseUnetWithInstanceInfo
import torch
from model.point_net import PointNet




class AutoencoderPN(nn.Module):

    def __init__(self,
            input_feature_dim,
            mid_feat_dim,
            out_dim,
            path=None
            ):

        super(AutoencoderPN, self).__init__()

        # ENCODER
        self.input_feature_dim = input_feature_dim
        self.backbone = PointNet(input_feature_dim,mid_feat_dim,out_dim)
        """
        Input:
            [b,n,3]
        Output:
            [b,out_dim]
        """

        self.decoder = nn.Sequential(
            nn.Conv1d(out_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 4000, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b,  num_points,3].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        # print(x.shape,'asdasd')
        encoding = self.backbone(x)  # shape [b, out_dim]

        data = torch.cat((encoding.unsqueeze(2), torch.ones((encoding.shape[0], encoding.shape[1], 2), device=x.device)), dim=2)

        restoration = self.decoder(data)  # shape [b, num_points ,3]
        # restoration = x.view(b, 3, 4000)
        return encoding, restoration

def chamfer_loss(point_set1, point_set2):
    # point_set1: (B, N, D)
    # point_set2: (B, M, D)

    # 计算点集1中每个点到点集2的最小距离
    point_set1_expand = point_set1.unsqueeze(2)  # (B, N, 1, D)
    point_set2_expand = point_set2.unsqueeze(1)  # (B, 1, M, D)
    dist1 = torch.sum((point_set1_expand - point_set2_expand)**2, dim=-1)  # (B, N, M)
    min_dist1, _ = torch.min(dist1, dim=2)  # (B, N)

    # 计算点集2中每个点到点集1的最小距离
    dist2 = torch.sum((point_set1_expand - point_set2_expand)**2, dim=-1)  # (B, N, M)
    min_dist2, _ = torch.min(dist2, dim=1)  # (B, M)

    # 计算Chamfer距离
    chamfer_dist = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)  # (B,)
    chamfer_loss = torch.mean(chamfer_dist)

    return chamfer_loss


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

