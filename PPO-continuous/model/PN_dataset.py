import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class MultiClassPointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_labels = os.listdir(root_dir)
        self.class_to_idx = {class_label: idx for idx, class_label in enumerate(self.class_labels)}
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        file_list = []
        for class_label in self.class_labels:
            class_dir = os.path.join(self.root_dir, class_label)
            class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
            file_list.extend(class_files)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_list[idx]
        class_label = os.path.basename(os.path.dirname(file_name))

        pointcloud = np.load(file_name)

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud, self.class_to_idx[class_label]


class ToTensor(object):
    def __call__(self, pointcloud):
        # Convert open3d point cloud to torch tensor
        points = torch.tensor(pointcloud, dtype=torch.float32)
        return points


# 设置数据集路径


if __name__ == '__main__':
    import os.path as osp
    dataset_path = osp.join(osp.dirname(os.getcwd()), 'dataset', 'val')

    #    创建数据集实例
    transform = transforms.Compose([ToTensor()])
    pointcloud_dataset = MultiClassPointCloudDataset(root_dir=dataset_path, transform=transform)

    # 创建数据加载器
    batch_size = 32
    pointcloud_dataloader = DataLoader(pointcloud_dataset, batch_size=batch_size, shuffle=True)

    # 示例用法
    for batch, labels in pointcloud_dataloader:
        # 这里可以添加你的训练代码，batch 包含了一个批次的点云数据，labels 包含了对应的类别标签
        print("Batch shape:", batch.shape)
        print("Labels:", labels)
