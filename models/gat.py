import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pointgat import sample_and_group


class GraphAttentionLayer(nn.Module):
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features
        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f: torch.Tensor, adj_mat: torch.Tensor):
        # Number of nodes
        batch_size = f.shape[0]
        n_nodes = f.shape[1]
        n_nodes_j = f.shape[2]

        g = self.linear(f).view(batch_size, n_nodes, self.n_heads, self.n_hidden)

        # Calculate attention score
        g_repeat = g.repeat(1, n_nodes, 1, 1)
        # `g_repeat_interleave` gets
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)
        # Now we concatenate to get
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # Reshape so that `g_concat[i, j]` is $\overrightarrow{g_i} \Vert \overrightarrow{g_j}$
        g_concat = g_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        
        # Calculate
        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)
        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=-2).squeeze(-2)
        


class TransitionDownBlock(nn.Module):
    def __init__(self, cfg, npoint, in_channel, out_channel, knn=True):
        super(TransitionDownBlock, self).__init__()
        self.npoint = npoint
        self.nneighbor = cfg.model.nneighbor
        self.transformer_dim = cfg.model.transformer_dim
        self.knn = knn

        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.transformer = TransformerBlock(in_channel, self.transformer_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(out_channel*2, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        # self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     #in_channels,就是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。
        # self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        # last_channel = out_channel
        self.pool1 = nn.AvgPool2d((self.nneighbor, 1), stride=1)
        self.pool2 = nn.MaxPool2d((self.nneighbor, 1), stride=1)

    def forward(self, xyz, festure):
        """
        Input:
            xyz: input points position data, [B, N, C]
            festure: input points data with feature, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        new_xyz, new_festure = sample_and_group(xyz, festure, npoint=self.npoint, nsample=self.nneighbor, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # # new_points: sampled points data, [B, npoint, nsample, C+D]
        # for i, layer in enumerate(self.fc1): 
        #     new_festure = layer(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(new_festure)
        repeat_festure = festure.repeat(1, self.nneighbor, 1).view(festure.shape[0], festure.shape[1], self.nneighbor, festure.shape[-1])
        g_concat = torch.cat([], dim=-1)

        new_festure = self.mlp1(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        new_festure = self.transformer(new_festure)  # [B, npoint, nsample, C+D]
        new_festure = self.layer1(new_festure, xyz)


        # for i, layer in enumerate(self.fc2): 
        #     new_festure = layer(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) if i == 1 else layer(new_festure)
        #  # [B, npoint, nsample, out_channel]

        new_festure = self.mlp2(new_festure.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     new_points =  self.relu(bn(conv(new_points)))

        # new_festure_avg = self.pool1(new_festure).squeeze(-2)  # [B, npoint, out_channel]
        new_festure = self.pool2(new_festure).squeeze(-2)  # [B, npoint, out_channel]
        # new_festure = torch.cat((new_festure_avg, new_festure_max), dim=-1)

        # new_festure = self.mlp3(new_festure.permute(0, 2, 1)).permute(0, 2, 1)

        return new_xyz, new_festure  

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model) # d_model=512, d_points = 32
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(d_model, d_points)
    # xyz: b x n x 3, features: b x n x f
    def forward(self, features):
        pre = features # 64
        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        kT = k.transpose(-1, -2)
        # attn = self.fc_gamma(torch.matmul(q,kT))
        attn = torch.matmul(q,kT)
        attn = self.softmax(attn/np.sqrt(k.size(-1)))  # b x n x k x f # TODO :哪个维度上比较好；测试-1，-2
        res = torch.matmul(attn, v)
        res = self.fc2(res) + pre
        return res

class Backbone(nn.Module):
    def __init__(self, cfg):
        super(Backbone, self).__init__()
        self.layer1 = GraphAttentionLayer(3, 128, 8, is_concat=True, dropout=0.5)
        # self.layer2 = GraphAttentionLayer(64, 128, 8, is_concat=True, dropout=0.5)
        # self.layer3 = GraphAttentionLayer(128, 256, 8, is_concat=True, dropout=0.5)
        self.layer4 = GraphAttentionLayer(128, 512, 8, is_concat=True, dropout=0.5)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        xyz = x[..., :3]  # torch.Size([16, 1024, 3])
        assert xyz.shape[-1] == 3
        points = self.layer1(xyz, xyz)
        points = self.activation(points)

        points = self.layer4(points, xyz)
        points = self.activation(points)
        points = self.dropout(points)
        return points

class Module(nn.Module):
    def __init__(self, cfg):
        super(Module,self).__init__()
        n_class = cfg.num_classes
        # npoints, nneighbor, d_points = cfg.num_point,  cfg.model.nneighbor,  cfg.input_dim
        # self.nblocks = nblocks
        self.backbone = Backbone(cfg)
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(512 , 256), # nblocks = 4

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_class)
        )
    def forward(self, x):
        # torch.Size([16, 1024, 6])
        points= self.backbone(x)  # torch.Size([16, 4, 512])
        res = self.aap(points.permute(0, 2, 1)).squeeze(-1)  # [16,512]
        res = self.fc2(res)  # [16,40]
        return res

if __name__ == '__main__':
    import sys
    sys.path.append("../") 
    from utils import read_yaml
    cfg = read_yaml('../configs/config.yaml')
    data = torch.rand(32, 1024, 3)
    print("===> testing pointMLP ...")
    model = Module(cfg).cuda()
    out = model(data.cuda())
    print(out.shape)