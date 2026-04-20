import numpy as np
import torch
from torch.utils.data import Dataset

from typing import List


class C2VDataset(Dataset):
    """Loading cell line single modality target vector"""
    def __init__(self, cell_tgt_file: str, valid_node_file: str, task='regression'):
        tgt = np.load(cell_tgt_file)
        nodes = np.load(valid_node_file) # eg.[    0     1     5 ... 17143 17148 17159]
        
        self.task = task
        if task == 'regression':
            self.tgt = torch.from_numpy(tgt).float()
        elif task == 'classification':
            # 对于分类任务，标签应该是整数类型
            # 如果原始数据是连续值，需要转换为类别
            # 这里假设数据已经是类别标签，如果不是，需要进行离散化
            if tgt.dtype in [np.float32, np.float64]:
                # 示例：将连续值转换为二分类（0/1）
                # 可以根据实际需求修改阈值或分类方式
                print("Warning: Converting continuous targets to binary classification (threshold=0.5)")
                tgt = (tgt > 0.5).astype(np.int64)
            self.tgt = torch.from_numpy(tgt).long()
        
        self.node_indices = torch.from_numpy(nodes)

    def __len__(self):
        return self.tgt.shape[0]

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long), self.tgt[item]


class C2VSymDataset(Dataset):
    """Loading multiple target vectors"""
    def __init__(self, target_files: List[str], node_files: List[str]):
        self.targets = []
        self.nodes = []
        for t_f, n_f in zip(target_files, node_files):
            t = np.load(t_f)
            t = torch.from_numpy(t).float()
            n = np.load(n_f)
            n = torch.from_numpy(n)
            self.targets.append(t)
            self.nodes.append(n)

    def __len__(self):
        return self.targets[0].shape[0]

    def __getitem__(self, item):
        ret = [target[item] for target in self.targets]
        ret.insert(0, torch.tensor(item, dtype=torch.long))
        return tuple(ret)
