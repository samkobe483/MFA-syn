import os
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader as GeometricDataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

con_loss_T = 0.01


# ============================================================================
# 新增模块1：SE-Block（Squeeze-and-Excitation）用于药物指纹特征过滤
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block：通道注意力机制

    对 4096 维的 Morgan 指纹进行特征过滤，学习每个 Bit 的重要性权重。
    先压缩（Squeeze）到低维，再激励（Excitation）回原维度，最后重新加权输入。
    """

    def __init__(self, input_dim, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.input_dim = input_dim
        reduced_dim = max(input_dim // reduction_ratio, 64)

        # Squeeze: 全局池化 + 压缩
        self.squeeze = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(inplace=True)
        )

        # Excitation: 扩展回原维度 + Sigmoid 生成权重
        self.excitation = nn.Sequential(
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, input_dim] 药物指纹特征
        return: [B, input_dim] 重新加权后的特征
        """
        # Squeeze: 学习通道重要性
        squeeze_out = self.squeeze(x)  # [B, reduced_dim]

        # Excitation: 生成每个通道的权重
        excitation_out = self.excitation(squeeze_out)  # [B, input_dim]

        # Scale: 用权重重新加权原始输入
        return x * excitation_out  # [B, input_dim]


# ============================================================================
# 新增模块2：双线性交互池化（Bilinear Pooling）
# ============================================================================
class BilinearInteractionPooling(nn.Module):
    """双线性交互池化：通过 Hadamard Product 捕获药物-细胞协同效应

    将药物特征 D 和细胞特征 C 进行逐元素相乘：F_inter = D ⊙ C
    这强制模型学习"协同效应"：如果药物在某位点为1，且细胞在对应位点也活跃，
    乘积就会放大这个信号，比简单拼接更能模拟药物-靶点相互作用。
    """

    def __init__(self, drug_dim, cell_dim, output_dim, dropout=0.2):
        super(BilinearInteractionPooling, self).__init__()

        # 将药物和细胞特征投影到相同维度
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        # 交互特征的后处理
        self.interaction_fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 门控机制：控制交互特征的贡献
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.Sigmoid()
        )

        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, drug_feat, cell_feat):
        """
        drug_feat: [B, drug_dim]
        cell_feat: [B, cell_dim]
        return: [B, output_dim] 融合后的交互特征
        """
        # 投影到相同维度
        drug_proj = self.drug_proj(drug_feat)  # [B, output_dim]
        cell_proj = self.cell_proj(cell_feat)  # [B, output_dim]

        # Hadamard Product（逐元素相乘）捕获协同效应
        interaction = drug_proj * cell_proj  # [B, output_dim]
        interaction = self.interaction_fc(interaction)  # [B, output_dim]

        # 拼接：药物特征 + 细胞特征 + 交互特征
        concat_feat = torch.cat([drug_proj, cell_proj, interaction], dim=1)  # [B, output_dim * 3]

        # 门控机制
        gate_weight = self.gate(concat_feat)  # [B, output_dim]

        # 加权交互特征
        gated_interaction = interaction * gate_weight  # [B, output_dim]

        # 最终融合
        final_concat = torch.cat([drug_proj, cell_proj, gated_interaction], dim=1)  # [B, output_dim * 3]
        output = self.output_layer(final_concat)  # [B, output_dim]

        return output, interaction  # 返回融合特征和原始交互特征


# ============================================================================
# 新增模块3：回归引导的分类器（Label Refinement）
# ============================================================================
class RegressionGuidedClassifier(nn.Module):
    """回归引导的分类器：利用回归值的序关系辅助分类

    不直接预测 0/1，而是先预测回归值（如 IC50），然后通过可学习的
    偏置 b 和缩放因子 s，用 Sigmoid(s * (pred - b)) 映射到分类概率。
    这让分类任务继承回归任务的"序关系"，建立更稳健的决策边界。
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(RegressionGuidedClassifier, self).__init__()

        # 共享特征提取层
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 回归头：预测连续值
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

        # 可学习的分类阈值参数
        self.threshold_bias = nn.Parameter(torch.tensor(0.0))  # 偏置 b
        self.threshold_scale = nn.Parameter(torch.tensor(1.0))  # 缩放因子 s

        # 独立的分类头（作为辅助）
        self.cls_head = nn.Linear(hidden_dim // 2, 2)

    def forward(self, feat):
        """
        feat: [B, input_dim]
        return: reg_out [B, 1], cls_out [B, 2], reg_guided_prob [B, 1]
        """
        # 共享特征提取
        shared_feat = self.shared_fc(feat)  # [B, hidden_dim // 2]

        # 回归预测
        reg_out = self.reg_head(shared_feat)  # [B, 1]

        # 回归引导的分类概率：Sigmoid(s * (pred - b))
        reg_guided_prob = torch.sigmoid(
            self.threshold_scale * (reg_out - self.threshold_bias)
        )  # [B, 1]

        # 独立分类预测
        cls_out = self.cls_head(shared_feat)  # [B, 2]

        return reg_out, cls_out, reg_guided_prob



# ============================================================================
# 细胞系特征处理器（双分支编码器）
# ============================================================================
class CellProcessor(nn.Module):
    """细胞系双分支编码器

    分支1: GIN embedding (954 dim) -> 512 dim
    分支2: Genomic features (768 dim) -> 512 dim
    融合后输出 512 dim
    """

    def __init__(self, cell1_dim=954, cell2_dim=768, output_dim=512, dropout=0.2):
        super(CellProcessor, self).__init__()

        # 分支1: GIN embedding 处理
        self.cell1_encoder = nn.Sequential(
            nn.Linear(cell1_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        # 分支2: Genomic features 处理
        self.cell2_encoder = nn.Sequential(
            nn.Linear(cell2_dim, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(640, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, cell1, cell2):
        """
        cell1: [B, 954] GIN embedding
        cell2: [B, 768] Genomic features
        return: [B, 512] 融合后的细胞特征
        """
        # 分支编码
        cell1_feat = self.cell1_encoder(cell1)  # [B, 512]
        cell2_feat = self.cell2_encoder(cell2)  # [B, 512]

        # 拼接
        concat_feat = torch.cat([cell1_feat, cell2_feat], dim=1)  # [B, 1024]

        # 门控融合
        gate_weight = self.gate(concat_feat)  # [B, 512]

        # 加权融合
        fused = self.fusion(concat_feat)  # [B, 512]
        output = fused * gate_weight + cell1_feat * (1 - gate_weight)

        return output


# ============================================================================
# 药物编码器（Transformer 风格，无位置编码）
# ============================================================================
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（无位置编码）"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        x: [B, seq_len, dim]
        return: [B, seq_len, dim]
        """
        B, seq_len, _ = x.shape

        Q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        K = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        V = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, S, S]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, H, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.dim)  # [B, S, dim]

        output = self.out_proj(attn_output)
        return output


class TransformerEncoderBlock(nn.Module):
    """Transformer 编码器块（无位置编码）"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, seq_len, dim]
        return: [B, seq_len, dim]
        """
        # Self-Attention + Residual
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # FFN + Residual
        x = x + self.mlp(self.norm2(x))
        return x


class DrugEncoder(nn.Module):
    """药物编码器：使用 SE-Block + Transformer（无位置编码）

    将 4096 维特征切分为 4 个 1024 维的 chunks，作为序列输入 Transformer。
    不使用位置编码，因为指纹的切片顺序没有时序关系。

    输入: 4096 dim (Morgan 1024 + Jaccard PCA 1024 + Target 1024 + Pathway 1024)
    输出: 512 dim
    """

    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=512, num_chunks=4,
                 num_heads=8, num_layers=2, dropout=0.2):
        super(DrugEncoder, self).__init__()

        self.num_chunks = num_chunks
        self.chunk_dim = input_dim // num_chunks  # 1024

        # SE-Block 用于特征过滤
        self.se_block = SEBlock(input_dim, reduction_ratio=16)

        # 将每个 chunk 投影到 hidden_dim
        self.chunk_proj = nn.Linear(self.chunk_dim, hidden_dim)
        self.chunk_norm = nn.LayerNorm(hidden_dim)

        # 注意：不使用位置编码！指纹切片没有时序关系
        # self.pos_embedding = ... (已移除)

        # Transformer 编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 输出层：聚合所有 chunks
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

    def forward(self, x):
        """
        x: [B, 4096] 药物特征
        return: [B, 512] 编码后的药物特征
        """
        B = x.size(0)

        # SE-Block 特征过滤
        x = self.se_block(x)  # [B, 4096]

        # 切分为 chunks: [B, 4096] -> [B, 4, 1024]
        chunks = x.view(B, self.num_chunks, self.chunk_dim)  # [B, 4, 1024]

        # 投影到 hidden_dim: [B, 4, 1024] -> [B, 4, 512]
        chunks = self.chunk_proj(chunks)  # [B, 4, 512]
        chunks = self.chunk_norm(chunks)

        # 不添加位置编码（Set Transformer 风格）

        # Transformer 编码
        for layer in self.transformer_layers:
            chunks = layer(chunks)  # [B, 4, 512]

        # 聚合：使用 mean pooling
        pooled, _ = chunks.max(dim=1)  # [B, 512]
        pooled = self.output_norm(pooled)

        # 输出
        output = self.output_fc(pooled)  # [B, 512]

        return output






# ============================================================================
# 对比学习 Projection Head
# ============================================================================
class ProjectionHead(nn.Module):
    """对比学习投影头：将特征投影到超球面空间"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [B, input_dim]
        return: [B, output_dim] L2 归一化后的投影特征
        """
        proj = self.net(x)
        return F.normalize(proj, dim=1)


# ============================================================================
# 主模型：MFASyn (Multi-Feature Attention Synergy)
# ============================================================================
class Model(nn.Module):
    """MFASyn 主模型 (优化版)

    优化点：
    1. 三线性药物交互：引入差值和乘积项，增强协同效应捕获。
    2. 全局使用 GELU 激活函数：提升回归任务的梯度平滑度。
    3. 维度对齐：适配增强后的交互特征。
    """

    def __init__(self, drug_dim=4096, cell1_dim=954, cell2_dim=768, hidden_dim=512, dropout=0.3):
        super(Model, self).__init__()

        # 药物编码器 (两个药物共享)
        self.drug_encoder = DrugEncoder(
            input_dim=drug_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_chunks=4,
            num_heads=8,
            num_layers=2,
            dropout=dropout
        )

        # 细胞编码器
        self.cell_encoder = CellProcessor(
            cell1_dim=cell1_dim,
            cell2_dim=cell2_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # 修改点：输入维度由 hidden_dim * 2 变为 * 4 (d1, d2, |d1-d2|, d1*d2)
        self.drug_pair_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),  # 回归任务推荐使用 GELU
            nn.Dropout(dropout)
        )

        # 双线性交互池化
        self.bilinear_pooling = BilinearInteractionPooling(
            drug_dim=hidden_dim,
            cell_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # 特征融合层 (去掉Cross-Attention后: drug_feat + cell_feat + bilinear_feat)
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)  # 最后一层轻微 dropout
        )

        # 回归引导分类器
        self.classifier = RegressionGuidedClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 对比学习投影头
        self.projection_head = ProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=256,
            output_dim=128
        )

    def forward(self, drug1, drug2, cell1, cell2):
        """
        drug1, drug2: [B, 4096]
        cell1: [B, 954]
        cell2: [B, 768]
        """
        # 1. 药物独立编码
        drug1_feat = self.drug_encoder(drug1)  # [B, 512]
        drug2_feat = self.drug_encoder(drug2)  # [B, 512]

        # 2. 增强药物对融合 (实现建议 3)
        # 显式捕捉药物间的相互作用信息
        diff_feat = torch.abs(drug1_feat - drug2_feat)  # 差异
        prod_feat = drug1_feat * drug2_feat  # 乘积

        # 拼接四个分支：[B, 512 * 4] = [B, 2048]
        drug_combined = torch.cat([drug1_feat, drug2_feat, diff_feat, prod_feat], dim=1)
        drug_feat = self.drug_pair_fusion(drug_combined)  # [B, 512]

        # 3. 细胞编码
        cell_feat = self.cell_encoder(cell1, cell2)  # [B, 512]

        # 4. 双线性交互池化 (药物与细胞的全局交互)
        bilinear_feat, interaction_feat = self.bilinear_pooling(drug_feat, cell_feat)  # [B, 512]

        # 5. 特征融合 (去掉Cross-Attention)
        # 拼接: 药物特征 + 细胞特征 + 双线性交互特征 [B, 512 * 3]
        concat_feat = torch.cat([drug_feat, cell_feat, bilinear_feat], dim=1)
        fused_feat = self.feature_fusion(concat_feat)  # [B, 512]

        # 6. 回归引导分类
        # 注意：reg_out 会在训练脚本中通过 y_std/y_mean 反归一化
        reg_out, cls_out, reg_guided_prob = self.classifier(fused_feat)

        # 7. 对比学习投影
        proj_feat = self.projection_head(fused_feat)

        return reg_out, cls_out, reg_guided_prob, proj_feat, interaction_feat


# ============================================================================
# 损失函数
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss：处理类别不平衡问题"""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([1.5, 0.8]).to(device)
        else:
            self.alpha = alpha  # [alpha_0, alpha_1] 类别权重
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, 2] logits
        targets: [B] 类别标签
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=torch.tensor(self.alpha, device=inputs.device) if self.alpha else None,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """监督对比学习损失"""

    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D] L2 归一化后的特征
        labels: [B] 类别标签
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )  # [B, B]

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 移除对角线（自身）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算正样本对的平均 log_prob
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos.mean()

        return loss


# ============================================================================
# 训练类：mfa
# ============================================================================
class mfa:
    """Multi-Feature Attention Synergy Model (优化版)"""

    def __init__(self, modeldir='Modelscl', foldnum=0, hiddim=8192, mmse=1000, task='classification', global_mean=0.0, global_std=1.0):
        self.modeldir = modeldir
        self.foldnum = foldnum
        self.hiddim = hiddim
        self.mmse = mmse
        self.task = task
        # 核心：使用传入的全局参数并转换为 Tensor 移动到 GPU
        self.y_mean = torch.tensor(0.0).to(device)
        self.y_std = torch.tensor(1.0).to(device)

        # 标签归一化参数 (建议 5)
        #self.y_mean = 0.0
        #self.y_std = 1.0

        # 模型初始化 - 药物特征维度 4096
        self.model = Model(
            drug_dim=4096,  # Morgan(1024) + Jaccard PCA(1024) + Target(1024) + Pathway(1024)
            cell1_dim=766,  # almanac cell features dimension
            cell2_dim=766,  # almanac cell_feat.npy dimension
            hidden_dim=512,
            dropout=0.4
        ).to(device)

        if self.task == 'regression':
            self.loss_weight_reg = 1.0
            self.loss_weight_con = 0.3
            # 归一化后数据分布在 [-3, 3] 左右，delta=1.0 更合理
            self.reg_loss_fn = nn.SmoothL1Loss(beta=1.0)
            #self.mse_loss = nn.MSELoss()
            self.supcon_loss = SupervisedContrastiveLoss(temperature=con_loss_T)
            self.best_pcc = -1.0
            self.best_mse = float('inf')
        else:
            self.loss_weight_reg = 1.0
            self.loss_weight_cls = 2.0
            self.loss_weight_con = 0.5
            self.focal_loss = FocalLoss(alpha=[0.4, 1.6], gamma=3.0)
            self.supcon_loss = SupervisedContrastiveLoss(temperature=con_loss_T)
            self.best_auc = 0.0
            self.best_aupr = 0.0

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        self.best_model = None
        self.best_epoch = 0
        self.patience = 15
        self.counter = 0

    def train(self, tr_dataset, test_dataset, epochs=100, batch_size=512):
        # --- 步骤 1: 计算标签统计量 (建议 5) ---
        if self.task == 'regression':
            all_y = torch.tensor(
                [data.y.item() for data in tr_dataset],
                dtype=torch.float32
            )
            self.y_mean = all_y.mean().to(device)
            self.y_std = all_y.std().to(device)
            print(f"Label Normalization Applied -> Mean: {self.y_mean:.4f}, Std: {self.y_std:.4f}")
        print(f"\n[Training Mode: {self.task}]")
        print(f"Fixed Global Normalization Base -> Mean: {self.y_mean.item():.4f}, Std: {self.y_std.item():.4f}")
        train_loader = GeometricDataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)

                # 正确 reshape 输入特征
                drug1 = batch.drug1.view(batch.num_graphs, -1)
                drug2 = batch.drug2.view(batch.num_graphs, -1)
                cell1 = batch.cell1.view(batch.num_graphs, -1)
                cell2 = batch.cell2.view(batch.num_graphs, -1)
                reg_labels = batch.y.view(-1, 1)

                # --- 步骤 2: 目标值归一化 ---
                if self.task == 'regression':
                    norm_target = (reg_labels - self.y_mean) / (self.y_std + 1e-8)
                else:
                    norm_target = reg_labels

                reg_out, cls_out, reg_guided_prob, proj_feat, _ = self.model(
                    drug1, drug2, cell1, cell2
                )

                if self.task == 'regression':
                    # 计算标准化空间的回归损失
                    reg_loss = self.reg_loss_fn(reg_out, norm_target)
                    # 对比学习使用离散化伪标签 (基于原始 reg_labels)
                    pseudo_labels = torch.zeros_like(reg_labels.view(-1), dtype=torch.long)
                    pseudo_labels[reg_labels.view(-1) < 0] = 0
                    pseudo_labels[(reg_labels.view(-1) >= 0) & (reg_labels.view(-1) < 15)] = 1
                    pseudo_labels[(reg_labels.view(-1) >= 15) & (reg_labels.view(-1) < 30)] = 2
                    pseudo_labels[(reg_labels.view(-1) >= 30) & (reg_labels.view(-1) < 50)] = 3
                    pseudo_labels[reg_labels.view(-1) >= 50] = 4
                    con_loss = self.supcon_loss(proj_feat, pseudo_labels)

                    loss = self.loss_weight_reg * reg_loss + self.loss_weight_con * con_loss
                else:
                    # 分类逻辑保持原有实现
                    labels = batch.type.view(-1)
                    cls_loss = self.focal_loss(cls_out, labels)
                    con_loss = self.supcon_loss(proj_feat, labels)
                    reg_guide_target = (reg_labels >= 30).float()
                    reg_loss = F.binary_cross_entropy(reg_guided_prob, reg_guide_target)
                    loss = self.loss_weight_reg * reg_loss + self.loss_weight_cls * cls_loss + self.loss_weight_con * con_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            #self.scheduler.step(mse)

            # --- 步骤 3: 评估 (含反归一化过程) ---
            if self.task == 'regression':
                mse, rmse, pcc, scc, ci = self.test_regression(test_loader)

                print(
                    f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f} | MSE: {mse:.4f} | PCC: {pcc:.4f} | SCC: {scc:.4f}")

                # 基于 MSE 的早停机制 (MSE 越小越好)
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_pcc = pcc
                    self.best_epoch = epoch + 1
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    self.save_model()
                    print(f"  -> New best model! MSE: {mse:.4f}")
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f"  -> No improvement for {self.counter}/{self.patience} epochs.")
                self.scheduler.step(mse)
                if self.counter >= self.patience:
                    print(f"==> Early stopping at epoch {epoch + 1}. Best MSE: {self.best_mse:.4f}")
                    break
            else:
                # 分类评估... (略)
                pass

        # 加载最佳模型
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
            print(f"\nLoaded best model from epoch {self.best_epoch} with MSE: {self.best_mse:.4f}, PCC: {self.best_pcc:.4f}")

    def test_regression(self, test_loader):
        """测试并反归一化"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                # 正确 reshape 输入特征
                drug1 = batch.drug1.view(batch.num_graphs, -1)
                drug2 = batch.drug2.view(batch.num_graphs, -1)
                cell1 = batch.cell1.view(batch.num_graphs, -1)
                cell2 = batch.cell2.view(batch.num_graphs, -1)

                reg_out, _, _, _, _ = self.model(drug1, drug2, cell1, cell2)

                # --- 步骤 4: 反归一化还原数值 ---
                pred_original = reg_out * self.y_std + self.y_mean

                all_preds.extend(pred_original.cpu().view(-1).numpy())
                all_labels.extend(batch.y.view(-1).cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        return mean_squared_error(all_labels, all_preds), np.sqrt(mean_squared_error(all_labels, all_preds)), \
            pearsonr(all_labels, all_preds)[0], spearmanr(all_labels, all_preds)[0], concordance_index(all_labels,
                                                                                                       all_preds)

    def predict_regression(self, test_loader):
        """预测并还原真实量级"""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                # 正确 reshape 输入特征
                drug1 = batch.drug1.view(batch.num_graphs, -1)
                drug2 = batch.drug2.view(batch.num_graphs, -1)
                cell1 = batch.cell1.view(batch.num_graphs, -1)
                cell2 = batch.cell2.view(batch.num_graphs, -1)

                reg_out, _, _, _, _ = self.model(drug1, drug2, cell1, cell2)
                pred_original = reg_out * self.y_std + self.y_mean
                all_preds.extend(pred_original.cpu().view(-1).numpy())
                all_labels.extend(batch.y.view(-1).cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    def save_model(self):
        """保存模型及归一化参数"""
        # 确保目录存在
        os.makedirs(self.modeldir, exist_ok=True)
        model_path = os.path.join(self.modeldir, f'model_fold{self.foldnum}.pt')
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'y_mean': self.y_mean.cpu() if torch.is_tensor(self.y_mean) else self.y_mean,
            'y_std': self.y_std.cpu() if torch.is_tensor(self.y_std) else self.y_std,
            'task': self.task,
            'best_pcc': self.best_pcc if hasattr(self, 'best_pcc') else 0.0,
            'best_mse': self.best_mse if hasattr(self, 'best_mse') else float('inf')
        }
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path} with Norm Params (mean={float(self.y_mean):.4f}, std={float(self.y_std):.4f})")

    def load_model(self, model_path):
        """加载模型及归一化参数"""
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.y_mean = checkpoint.get('y_mean', 0.0)
        self.y_std = checkpoint.get('y_std', 1.0)
        self.task = checkpoint.get('task', 'regression')
        print(f"Model loaded. Ready for {self.task}.")
