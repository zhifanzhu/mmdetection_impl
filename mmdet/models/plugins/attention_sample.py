import torch
import torch.nn as nn
import torch.nn.functional as F


class TriAtt(nn.Module):
    def __init__(self):
        super(TriAtt, self).__init__()
        self.feature_norm = nn.Softmax(dim=2)
        self.bilinear_norm = nn.Softmax(dim=2)

    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        h = x.size(2)
        w = x.size(3)
        f = x.reshape(n, c, -1)

        # *7 to obtain an appropriate scale for the input of softmax function.
        f_norm = self.feature_norm(f * 2)

        bilinear = f_norm.bmm(f.transpose(1, 2))  # torch.matmul can do...
        bilinear = self.bilinear_norm(bilinear)
        trilinear_atts = bilinear.bmm(f).view(n, c, h, w).detach()
        # Avg pooling here
        structure_att = torch.sum(trilinear_atts, dim=1, keepdim=True)
        return structure_att


class GridFC(nn.Module):
    def __init__(self, x_len, y_len, device='cuda'):
        super(GridFC, self).__init__()
        self.fc_x = nn.Linear(x_len, x_len)
        self.fc_y = nn.Linear(y_len, y_len)
        self.x_len = x_len
        self.y_len = y_len

        x_scale = 2.0 / x_len
        y_scale = 2.0 / y_len
        self.identity_x = torch.arange(x_len).view(
            1, x_len, 1).float().to(
            device).requires_grad_(False)
        self.identity_y = torch.arange(y_len).view(
            1, y_len, 1).float().to(
            device).requires_grad_(False)
        self.identity_x = self.identity_x * x_scale - 1.0
        self.identity_y = self.identity_y * y_scale - 1.0

    def init_weight(self):
        nn.init.xavier_normal_(self.fc_x.weight)
        nn.init.xavier_normal_(self.fc_y.weight)
        nn.init.constant_(self.fc_x.bias, 0)
        nn.init.constant_(self.fc_y.bias, 0)

    # alternative : do cumsum
    @staticmethod
    def map_axis_gen(att):
        # att.shape : [b, 1, h, w]
        map_sx, _ = torch.max(att, 3)
        map_sx = map_sx.unsqueeze(3)
        map_sy, _ = torch.max(att, 2)
        map_sy = map_sy.unsqueeze(2)
        sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
        sum_sy = torch.sum(map_sy, (1, 3), keepdim=True)
        map_sx = torch.div(map_sx, sum_sx)  # [1, 1, 32, 1]
        map_sy = torch.div(map_sy, sum_sy)  # [1, 1, 1, 32]
        return map_sx, map_sy

    def forward(self, att):
        """
        In reality, most of the time x_len == y_len, but we keep them
        seperate.
        """
        batch = att.size(0)
        x_len = self.x_len
        y_len = self.y_len
        att_x, att_y = self.map_axis_gen(att)

        index_x = self.fc_x(att_x.view(batch, -1)).view(batch, x_len, 1)
        index_y = self.fc_y(att_y.view(batch, -1)).view(batch, y_len, 1)
        # identity_x = torch.arange(x_len).unsqueeze(-1).unsqueeze(0)
        # identity_y = torch.arange(y_len).unsqueeze(-1).unsqueeze(0)
        index_x = index_x + self.identity_x  # h(x) = x + f(x) for easier learning?
        index_y = index_y + self.identity_y
        index_x = torch.clamp(index_x, min=-1.0, max=1.0)
        index_y = torch.clamp(index_y, min=-1.0, max=1.0)

        one_vector_x = torch.ones_like(index_x).cuda()
        one_vector_y = torch.ones_like(index_y).cuda()
        # generate meshgrid like
        grid_x = torch.matmul(one_vector_x, index_x.transpose(1, 2)).unsqueeze(-1)
        grid_y = torch.matmul(index_y, one_vector_y.transpose(1, 2)).unsqueeze(-1)
        grid = torch.cat([grid_x, grid_y], dim=3)
        return grid


class AttentionTransform(nn.Module):
    def __init__(self,
                 x_len,
                 y_len):
        super(AttentionTransform, self).__init__()
        self.tri_att = TriAtt()
        self.grid_func = GridFC(x_len=x_len, y_len=y_len)
        # self.init_weights()

    def init_weights(self):
        self.grid_func.init_weight()

    def forward(self, feat):
        att = self.tri_att(feat)
        grid = self.grid_func(att)
        out = F.grid_sample(feat, grid)
        return out
