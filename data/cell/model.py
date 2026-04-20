import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# 尝试导入 GINConv，如果失败则使用自定义实现
try:
    from dgl.nn.pytorch.conv import GINConv
except ImportError:
    try:
        from dgl.nn import GINConv
    except ImportError:
        # 自定义 GINConv 实现
        class GINConv(nn.Module):
            def __init__(self, apply_func, aggregator_type='sum'):
                super(GINConv, self).__init__()
                self.apply_func = apply_func
                self.aggregator_type = aggregator_type
                self.eps = nn.Parameter(torch.FloatTensor([0]))
            
            def forward(self, graph, feat):
                with graph.local_scope():
                    aggregate_fn = {
                        'sum': dgl.function.sum,
                        'mean': dgl.function.mean,
                        'max': dgl.function.max
                    }[self.aggregator_type]
                    
                    graph.ndata['h'] = feat
                    graph.update_all(dgl.function.copy_u('h', 'm'), aggregate_fn('m', 'neigh'))
                    rst = (1 + self.eps) * feat + graph.ndata['neigh']
                    rst = self.apply_func(rst)
                    return rst


class GINEncoder(nn.Module):
    """Use multi-layer GINConv to extract graph structure node features"""
    def __init__(self, in_features: int, out_features: int, activation=F.relu, k: int = 2):
        super(GINEncoder, self).__init__()
        assert k >= 2
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.conv = nn.ModuleList()
        # First layer
        mlp_first = nn.Sequential(
            nn.Linear(in_features, 2 * out_features),
            nn.BatchNorm1d(2 * out_features),
            nn.ReLU(),
            nn.Linear(2 * out_features, 2 * out_features)
        )
        self.conv.append(GINConv(mlp_first, aggregator_type='sum'))
        
        # Middle layers
        for _ in range(1, k - 1):
            mlp_mid = nn.Sequential(
                nn.Linear(2 * out_features, 2 * out_features),
                nn.BatchNorm1d(2 * out_features),
                nn.ReLU(),
                nn.Linear(2 * out_features, 2 * out_features)
            )
            self.conv.append(GINConv(mlp_mid, aggregator_type='sum'))
        
        # Last layer
        mlp_last = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.conv.append(GINConv(mlp_last, aggregator_type='sum'))

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        """Input graph and node features, return encoded node embedding."""
        for conv in self.conv:
            x = self.activation(conv(g, x))
        return x


class Cell2Vec(nn.Module):
    """Cell lineage state modeling using graph neural network based on PPI graph"""
    def __init__(self, encoder, n_cell, n_dim, task='regression', num_classes=None):
        super(Cell2Vec, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_features, n_dim),
            nn.Dropout()
        )
        self.task = task
        
        # 如果是分类任务，添加分类头
        if task == 'classification':
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.num_classes = num_classes
            # 分类头：将输出映射到类别数
            self.classifier = nn.Sequential(
                nn.Linear(n_dim, n_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(n_dim // 2, num_classes)
            )
        
        # 实现公式（2）中的f(Z)操作，将 encoder 输出的特征维度投影到 n_dim，然后是一个 nn.Dropout 层，帮助防止模型过拟合。    
    
    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(g, x) # ppi
        encoded = encoded.index_select(0, x_indices)    # 从 encoded 中选择与基因索引 x_indices 对应的节点特征。这样可以确保只选择特定的节点特征。
        proj = self.projector(encoded).permute(1, 0)    # permute(1, 0) 是为了改变张量的维度顺序，使其与细胞嵌入相乘时的维度匹配，PPI ⽹络中获取的基因特征矩阵
        emb = self.embeddings(c_indices)    # c_j 细胞系的状态特征
        out = torch.mm(emb, proj)   # 实现公式（2）o^t_j = f(Z) · c_j,
        # 基因隐藏状态和细胞系状态的交互，产生了基因的显性状态向量，表示在当前细胞系中的基因表现
        
        # 如果是分类任务，通过分类头
        if self.task == 'classification':
            # 对每个节点的输出进行分类
            # out shape: (batch_size, num_nodes)
            # 我们需要对整个细胞系进行分类，所以聚合节点特征
            cell_features = emb  # 使用细胞嵌入作为特征
            out = self.classifier(cell_features)  # (batch_size, num_classes)
        
        return out


class RandomW(nn.Module):

    def __init__(self, n_node, n_node_dim, n_cell, n_dim):
        super(RandomW, self).__init__()
        self.encoder = nn.Embedding(n_node, n_node_dim)
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(n_node_dim, n_dim),
            nn.Dropout()
        )

    def forward(self, x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out
