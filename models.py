import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
from dgllife.model.gnn import GCN
from torch.nn.utils.weight_norm import weight_norm
from DAGF import DAGFusion



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]


        self.drug_extractor = MolecularIGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor = ProteinInceptionCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = DAGFusion(h_dim=mlp_in_dim,h_out=ban_heads)

        self.mlp_classifier = MLP_KANDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = torch.sigmoid(self.mlp_classifier(f))

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class ResidualGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.gcn = GCN(in_feats=in_feats, hidden_feats=[out_feats], activation=[activation])
        self.norm = nn.BatchNorm1d(out_feats)  # 批归一化稳定训练
        # 若输入输出维度不同，通过线性变换匹配
        self.residual = nn.Linear(in_feats, out_feats) if in_feats != out_feats else nn.Identity()

    def forward(self, graph, feats):
        residual = self.residual(feats)
        out = self.gcn(graph, feats)  # GCN输出
        out = self.norm(out + residual)  # 残差融合+归一化
        return out

class ProteinInceptionCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes=[3,5, 7, 9], padding=True):
        super().__init__()
        self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0 if padding else None)

        # Inception 模块
        self.inception1 = Inception1D(embedding_dim, 128, kernel_sizes)
        self.inception2 = Inception1D(128, 128, kernel_sizes)
        self.inception3 = Inception1D(128, 128, kernel_sizes)
        self.inception4 = Inception1D(128, 128, kernel_sizes)
        # self.inception5 = Inception1D(128, 128, kernel_sizes)

        self.output_feats = 128 # num_filters[-1]
        # self.num_filters = num_filters[1]
        # self.num_filters1 = num_filters[2]
    def forward(self, v):
        # print("!!!",self.num_filters,self.num_filters1)
        v = self.embedding(v.long()).transpose(1, 2)  # [batch, embed_dim, seq_len]
        v = self.inception1(v)
        v = self.inception2(v)
        v = self.inception3(v)
        v = self.inception4(v)
        # v = self.inception5(v)

        #v = self.inception4(v)
        return v.transpose(1, 2)  # [batch, seq_len, output_feats]

class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5, 7, 9]):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4 for branch splitting."

        # 分支1: 1x1卷积
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, padding='same')

        # 分支2: 1x1 + 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=kernel_sizes[0], padding='same')
        )

        # 分支3: 1x1 + 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=kernel_sizes[1], padding='same')
        )

        # 分支4: 全局平均池化 + 1x1卷积（上采样在forward中动态设置）
        self.branch4_pool = nn.AdaptiveAvgPool1d(1)
        self.branch4_conv = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        # 动态上采样到输入长度 x.shape[-1]
        branch4 = self.branch4_pool(x)  # [batch, in_channels, 1]
        branch4 = self.branch4_conv(branch4)  # [batch, out_channels//4, 1]
        branch4 = F.interpolate(branch4, size=x.shape[-1], mode='linear', align_corners=False)  # 上采样

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return F.gelu(self.bn(out))#relu

class MolecularIGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=nn.GELU()):#ReLU
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)  # 处理 padding 特征

        # 构建残差GCN层
        self.gnn_layers = nn.ModuleList()
        prev_feats = dim_embedding
        for feat in hidden_feats:
            self.gnn_layers.append(ResidualGCNLayer(prev_feats, feat, activation))
            prev_feats = feat

        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)  # 初始特征映射

        # 逐层传递特征
        for layer in self.gnn_layers:
            node_feats = layer(batch_graph, node_feats)

        # 保持输出维度不变（batch_size, num_nodes, output_feats）
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class KANLayer(nn.Module):
    """可学习的非线性函数层（简化版 KAN 层）"""

    def __init__(self, input_dim, output_dim, num_basis=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 线性部分
        self.basis_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, num_basis),  # 样条基函数
                nn.Linear(num_basis, 1, bias=False)
            ) for _ in range(input_dim)
        ])

    def forward(self, x):
        # 线性变换
        linear_out = self.linear(x)

        # 非线性变换（对每个输入维度独立处理）
        nonlinear_out = torch.zeros_like(linear_out)
        for i in range(x.shape[1]):
            basis_input = x[:, i:i + 1]  # 取第 i 个特征
            nonlinear_out[:, i] = self.basis_functions[i](basis_input).squeeze()

        return linear_out + nonlinear_out  # 线性 + 非线性


class MLP_KANDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, use_kan=True):
        super().__init__()
        self.use_kan = use_kan

        self.fc1 = KANLayer(in_dim, hidden_dim)
        self.fc2 = KANLayer(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = self.fc4(x)
        return x



class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
