import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import sys
# sys.path.append('../..')
import torch
import time as t

sys.path.append(os.path.join(os.getcwd(), '..'))
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import glob
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from model.point_net import EncoderDecoder
from model.PN_dataset import MultiClassPointCloudDataset, ToTensor
import wandb
from tqdm import tqdm
import torch.nn as nn
import os.path as osp

# from icecream import ic
# ic.disable()
CE_loss = nn.CrossEntropyLoss()
time_stamp = t.strftime('%d%H%M%S', t.localtime())
if osp.exists(f'./model/{time_stamp}') is False:
    os.mkdir(f'./model/{time_stamp}')
    os.mkdir(f'./result/{time_stamp}')


# chamfer_loss=ChamLoss()

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


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


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, dim= 1, largest= True, sorted= True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, points):
        self.points = points

    def __getitem__(self, index):
        return np.load(self.points[index])

    def __len__(self):
        return len(self.points)


# def chamfer_loss(x, x_recon):
#     dist_matrix = pairwise_distances(x, x_recon)
#     min_dist = torch.min(dist_matrix, dim=1)[0]
#     chamfer_loss = torch.mean(min_dist)
#     return chamfer_loss

def load_point_cloud_data(object_name):
    train_list = glob.glob('./dataset/{}/train/*.npy'.format(object_name))
    val_list = glob.glob('./dataset/{}/val/*.npy'.format(object_name))
    return train_list, val_list


def train_vae(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    accTop1_avg = RunningAverage()
    loss_avg = RunningAverage()
    loss_cls_avg = RunningAverage()
    loss_recon_avg = RunningAverage()

    for (data, label) in tqdm(train_loader, desc='Loading'):
        # for i in range(len(data)):
        #     print(data[i][:2,:2],f'{i}')
        #     if i==5:
        #         break
        # assert False
        # print(data.shape)
        data = data.to(device).float()
        label = label.to(device)
        optimizer.zero_grad()

        encoding, recon_batch, logits = model(data)
        # loss_recon = chamfer_loss(encoding, data)
        # loss_recon += chamfer_loss(recon_batch, data)
        loss_recon = ChamLoss(encoding, recon_batch, data, 0.3)

        loss = loss_recon
        if model.cls:
            loss_cls = CE_loss(logits, label)
            loss += loss_cls
            loss_cls_avg.update(loss_cls.item())
            _, predicted = torch.max(logits, 1)  # 在第一维度上找到最大值的索引
            accuracy = (predicted == label).float().mean().item()
            accTop1_avg.update(accuracy)

        loss_recon_avg.update(loss_recon.item())
        loss_avg.update(loss.item())

        loss.backward()
        optimizer.step()

    if model.cls:
        wandb.log({"train_cls_acc": accTop1_avg.value()})
        print({"train_cls_acc": accTop1_avg.value()})
        wandb.log({"train_cls_loss": loss_cls_avg.value()})
        print({"train_cls_loss": loss_cls_avg.value()})

    wandb.log({"Train Loss": loss_avg.value()})
    wandb.log({"train_recon_loss": loss_recon_avg.value()})
    print("train_recon_loss", loss_recon_avg.value())
    print(f"Train Loss:{loss_avg.value()}")

    return train_loss / len(train_loader)


def test_vae(model, test_loader, optimizer, device, epoch):
    model.eval()
    test_loss = 0
    accTop1_avg = RunningAverage()
    loss_cls_avg = RunningAverage()
    loss_recon_avg = RunningAverage()
    loss_avg = RunningAverage()

    with torch.no_grad():
        for (data, label) in test_loader:
            data = data.to(device).float()
            label = label.to(device)

            encoding, recon_batch, logits = model(data)
            # loss_recon = chamfer_loss(encoding, data)
            # loss_recon += chamfer_loss(recon_batch, data)
            loss_recon = ChamLoss(encoding, recon_batch, data, 0.3)
            loss = loss_recon
            if model.cls:
                loss_cls = CE_loss(logits, label)
                loss += loss_cls
                loss_cls_avg.update(loss_cls.item())
                _, predicted = torch.max(logits, 1)  # 在第一维度上找到最大值的索引
                accuracy = (predicted == label).float().mean().item()
                accTop1_avg.update(accuracy)

            loss_recon_avg.update(loss_recon.item())

            test_loss += loss.item()

            loss_avg.update(loss.item())
        if model.cls:
            wandb.log({"test_acc": accTop1_avg.value()})
            print({"test_cls_acc": accTop1_avg.value()})
            wandb.log({"test_cls_loss": loss_cls_avg.value()})
            print({"test_cls_loss": loss_cls_avg.value()})

        wandb.log({"test_loss": loss_avg.value()})
        wandb.log({"test_recon_loss": loss_recon_avg.value()})
        print(f"test_recon_loss {loss_recon_avg.value()}")
        print(f"Test Loss: {test_loss:.4f}")

    return test_loss / len(test_loader)


def main(args):
    # 定义超参数
    from torchvision import transforms
    epochs = 200
    learning_rate = 0.1
    milestones = [80, 150]
    batch_size = args.batch_size

    # 加载数据
    # 假设你已经有一个点云数据集，存储在一个numpy数组中，每个样本是一个形状为(N, 3)的点云
    # 这里假设你的数据集已经被划分为训练集和测试集
    # train_points和test_points分别是训练集和测试集的点云数据
    test_dataset_path = osp.join(os.getcwd(), 'dataset', 'val')
    train_dataset_path = osp.join(os.getcwd(), 'dataset', 'train')

    transform = transforms.Compose([ToTensor()])
    train_dataset = MultiClassPointCloudDataset(root_dir=train_dataset_path, transform=transform)
    test_dataset = MultiClassPointCloudDataset(root_dir=test_dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 创建数据加载器

    # 创建模型和优化器
    model = EncoderDecoder(cls=args.cls, class_num=args.class_num)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 将模型和数据加载到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # for index,data in enumerate(train_loader):
    #     print(data.shape,'1')
    #     if index==5:
    #         break
    # for index,data in enumerate(test_loader):
    #     print(data.shape,'1')
    #     if index==5:
    #         break
    # for data in tqdm(train_loader):
    #     print(data.shape,'2')
    #     break
    # 训练和测试模型
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = train_vae(model, train_loader, optimizer, device)
        test_loss = test_vae(model, test_loader, optimizer, device, epoch)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        scheduler.step()  # 更新学习率
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'./model/{time_stamp}/pointnet_autoencoder{epoch}.pth')
            """
            Draw Visualization plots
            """
            if epoch == 0:
                fig = plt.figure(figsize=(10, 10))

                sample = torch.tensor(test_dataset[0][0], dtype=torch.float32).cpu().numpy()
                ax1 = fig.add_subplot(141, projection="3d")
                ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                ax1.view_init(elev=0, azim=0)  # elev表示仰角，azim表示方位角
                ax1.set_title("Original Point Cloud_view1")

                num_point = sample.shape[0]
                print(num_point)
                print(sample.shape)
                random_view = torch.randperm(num_point)
                sample = sample[random_view]
                print(sample.shape)

                ax2 = fig.add_subplot(142, projection="3d")
                ax2.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                ax2.view_init(elev=20, azim=30)  # elev表示仰角，azim表示方位角
                ax2.set_title("Original Point Cloud_view2")

                random_view = torch.randperm(num_point)
                sample = sample[random_view]
                print(sample.shape)

                ax3 = fig.add_subplot(143, projection="3d")
                ax3.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                ax3.view_init(elev=40, azim=60)  # elev表示仰角，azim表示方位角

                ax3.set_title("Original Point Cloud_view3")

                random_view = torch.randperm(num_point)
                sample = sample[random_view]

                ax4 = fig.add_subplot(144, projection="3d")
                ax4.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
                ax4.view_init(elev=60, azim=90)  # elev表示仰角，azim表示方位角
                ax4.set_title("Original Point Cloud_view4")
                plt.tight_layout()

                plt.savefig(f'./result/{time_stamp}/original_plot.png')
                plt.close()
            with torch.no_grad():
                # sample = test_dataset[0].to(device).unsqueeze(0)

                sample = torch.tensor(test_dataset[0][0], dtype=torch.float32).unsqueeze(0).cuda()
                encoding, restoration, _ = model(sample)
                # print(restoration.shape,'asdasd')
                # assert False
                restoration = restoration.squeeze(0).cpu().numpy()

                fig = plt.figure(figsize=(10, 10))

                ax1 = fig.add_subplot(141, projection="3d")
                ax1.scatter(restoration[:, 0], restoration[:, 1], restoration[:, 2])
                ax1.set_title("Recon Point Cloud_view1")

                num_point = restoration.shape[0]
                random_view = torch.randperm(num_point)
                restoration = restoration[random_view]

                ax2 = fig.add_subplot(142, projection="3d")
                ax2.scatter(restoration[:, 0], restoration[:, 1], restoration[:, 2])
                ax2.view_init(elev=20, azim=30)  # elev表示仰角，azim表示方位角

                ax2.set_title("Recon Point Cloud_view2")

                random_view = torch.randperm(num_point)
                restoration = restoration[random_view]

                ax3 = fig.add_subplot(143, projection="3d")
                ax3.scatter(restoration[:, 0], restoration[:, 1], restoration[:, 2])
                ax3.view_init(elev=40, azim=60)  # elev表示仰角，azim表示方位角

                ax3.set_title("Recon Point Cloud_view3")

                random_view = torch.randperm(num_point)
                restoration = restoration[random_view]

                ax4 = fig.add_subplot(144, projection="3d")
                ax4.scatter(restoration[:, 0], restoration[:, 1], restoration[:, 2])
                ax4.view_init(elev=60, azim=90)  # elev表示仰角，azim表示方位角

                ax4.set_title("Recon Point Cloud_view4")
                plt.tight_layout()

                plt.savefig(f'./result/{time_stamp}/output{epoch}.png')
                plt.close()

                # plt.show()

    wandb.finish()
    # 绘制训练和测试损失曲线
    # plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    # plt.plot(range(1, epochs+1), test_losses, label="Test Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # 可视化重构的点云示例


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--log_dir", type=str, default='./Ex', help=" The log directory")
    parser.add_argument("--seed", type=int, default=2333, help=" Seed")

    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    # parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
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
    parser.add_argument("--batch_size", type=int, default=4, help="Trick 10: tanh activation function")
    parser.add_argument("--object_name", type=str, default='mustard_bottle', help="Trick 10: tanh activation function")
    parser.add_argument("--class_num", type=int, default=9)
    parser.add_argument("--cls", type=bool, default=False)

    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="pre-PN",

        # track hyperparameters and run metadata
        config=args
    )
    main(args)

# start a new wandb run to track this script

# simulate training
