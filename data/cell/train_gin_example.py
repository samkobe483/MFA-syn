# 使用 GIN 编码器的训练示例
# 只需要修改 train.py 中的编码器初始化部分

# 原来的 GAT 编码器初始化：
"""
num_heads = 8 
encoder = GATEncoder(
    in_features = node_features.shape[1], 
    out_features = hidden_dim, 
    num_heads = num_heads, 
    activation = F.relu, 
    k = num_layers
).to(device)
"""

# 替换为 GIN 编码器初始化：
"""
from model import GINEncoder

encoder = GINEncoder(
    in_features = node_features.shape[1], 
    out_features = hidden_dim, 
    activation = F.relu, 
    k = num_layers  # 默认 k=2，即 2 层 GIN
).to(device)
"""

# 其他代码保持不变
# model = Cell2Vec(encoder, len(c2v_dataset), emb_dim).to(device)
# ...
