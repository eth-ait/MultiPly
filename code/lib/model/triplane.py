import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .embedders import get_embedder
import math


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


EPS = 1e-3

class TriPlane(nn.Module):
    def __init__(self, number_person, features=64, resX=128, resY=128, resZ=128):
        super().__init__()
        assert resX == resY == resZ, "resX, resY, resZ must be the same"
        self.encoder = nn.Embedding(number_person, 3 * features * resX * resY)
        self.plane_xy_list = nn.ParameterList()
        self.plane_yz_list = nn.ParameterList()
        self.plane_xz_list = nn.ParameterList()
        for p in range(number_person):
            person_id_tensor = torch.from_numpy(np.array([p])).long()
            person_encoding = self.encoder(person_id_tensor.to(self.encoder.weight.device)).reshape(3, features, resX, resY)
            self.plane_xy_list.append(person_encoding[[0]])
            self.plane_yz_list.append(person_encoding[[1]])
            self.plane_xz_list.append(person_encoding[[2]])
        # self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        # self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        # self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features
        self.n_input_dims = 3
        # self.n_output_dims = 3 * features
        self.n_output_dims = features

    def forward(self, x, person_id, pose):
        # assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        # valid input [-1, 1]
        plane_xy = self.plane_xy_list[person_id]
        plane_xz = self.plane_xz_list[person_id]
        plane_yz = self.plane_yz_list[person_id]
        # x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        # padding_mode='zeros' ==> the value of the pixels outside the grid is considered as 0
        # feat_xy = F.grid_sample(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xz = F.grid_sample(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_yz = F.grid_sample(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xy = cu.grid_sample_2d(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_xz = cu.grid_sample_2d(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_yz = cu.grid_sample_2d(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        feat_xy = grid_sample(plane_xy, coords[..., [0, 1]],)[0, :, :, 0].transpose(0, 1)
        feat_xz = grid_sample(plane_xz, coords[..., [0, 2]],)[0, :, :, 0].transpose(0, 1)
        feat_yz = grid_sample(plane_yz, coords[..., [1, 2]],)[0, :, :, 0].transpose(0, 1)

        # feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = (feat_xy + feat_xz + feat_yz) / 3
        feat = feat.reshape(*shape[:-1], self.n_output_dims)
        return feat, 0


class TriPlaneMulti(nn.Module):
    def __init__(self, number_person, features=64, resX=128, resY=128, resZ=128, opt=None):
        super().__init__()
        self.opt = opt
        try:
            self.use_cuda_grid = opt.use_cuda_grid
            print("use_cuda_grid", self.use_cuda_grid)
        except:
            self.use_cuda_grid = False
            print("use_cuda_grid", self.use_cuda_grid)
        assert resX == resY == resZ, "resX, resY, resZ must be the same"
        embed_fn, input_ch = get_embedder(6, input_dims=3, mode='fourier')
        self.embed_fn = embed_fn
        # self.encoder = nn.Embedding(number_person, 3 * features * resX * resY)
        self.multi_encoder = nn.ParameterList()
        # L = [128, 64, 32, 16, 8, 4, 2]
        # L = [128, 64, 32, 16]
        L = list(self.opt.triplane_res)
        # self.triplane_shortcut = self.opt.triplane_shortcut
        self.L_number = len(L)
        for l in L:
            self.multi_encoder.append(nn.Embedding(number_person, 3 * features * l * l))

        self.all_xy = nn.ParameterList()
        self.all_yz = nn.ParameterList()
        self.all_xz = nn.ParameterList()
        self.all_persons_layer = nn.ModuleList()
        self.all_persons_implicit_layer = nn.ModuleList()
        self.last_layer = nn.ModuleList()
        for p in range(number_person):
            plane_xy_list = nn.ParameterList()
            plane_yz_list = nn.ParameterList()
            plane_xz_list = nn.ParameterList()
            person_id_tensor = torch.from_numpy(np.array([p])).long()
            for layer_id, l in enumerate(L):
                person_encoding = self.multi_encoder[layer_id](person_id_tensor.to(self.multi_encoder[layer_id].weight.device)).reshape(3, features, l, l)
                plane_xy_list.append(person_encoding[[0]])
                plane_yz_list.append(person_encoding[[1]])
                plane_xz_list.append(person_encoding[[2]])
            self.all_xy.append(plane_xy_list)
            self.all_yz.append(plane_yz_list)
            self.all_xz.append(plane_xz_list)
            adapt_layer = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(features * self.L_number * 3, 256)),
                nn.Softplus(beta=100),
                nn.utils.weight_norm(nn.Linear(256, 256)),
                nn.Softplus(beta=100),
                # nn.utils.weight_norm(nn.Linear(256, 64)), # used for multi_add version
            )
            self.all_persons_layer.append(adapt_layer)
            # used for multi_add version
            # implicit_layer = nn.Sequential(
            #     nn.utils.weight_norm(nn.Linear(64 + 69 + input_ch, 256)),
            #     nn.Softplus(beta=100),
            #     nn.utils.weight_norm(nn.Linear(256, 256)),
            #     nn.Softplus(beta=100),
            #     nn.utils.weight_norm(nn.Linear(256, 256)),
            #     nn.Softplus(beta=100),
            #     nn.utils.weight_norm(nn.Linear(256, 256)),
            # )
            # self.all_persons_implicit_layer.append(implicit_layer)
            last_layer = nn.Linear(256, 64 + 1)
            init_val = 1e-5
            torch.nn.init.constant_(last_layer.bias, 0.0)
            torch.nn.init.uniform_(last_layer.weight, -init_val, init_val)
            last_layer = nn.utils.weight_norm(last_layer)
            self.last_layer.append(last_layer)
        self.dim = features
        self.n_input_dims = 3
        # self.n_output_dims = 3 * features
        self.n_output_dims = features
        if self.use_cuda_grid:
            from .grid import cuda_gridsample as cu
            self.cu = cu

    def forward(self, x, person_id, pose):
        # assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        # valid input [-1, 1]

        # [N, input_ch]
        x_embed = self.embed_fn(x)

        plane_list_xy = self.all_xy[person_id]
        plane_list_xz = self.all_yz[person_id]
        plane_list_yz = self.all_xz[person_id]
        # x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        # padding_mode='zeros' ==> the value of the pixels outside the grid is considered as 0
        # feat_xy = F.grid_sample(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xz = F.grid_sample(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_yz = F.grid_sample(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xy = cu.grid_sample_2d(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_xz = cu.grid_sample_2d(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_yz = cu.grid_sample_2d(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        multi_feat_xy = []
        multi_feat_xz = []
        multi_feat_yz = []
        for plane_xy, plane_xz, plane_yz in zip(plane_list_xy, plane_list_xz, plane_list_yz):
            if not self.use_cuda_grid:
                feat_xy = grid_sample(plane_xy, coords[..., [0, 1]],)[0, :, :, 0].transpose(0, 1)
                feat_xz = grid_sample(plane_xz, coords[..., [0, 2]],)[0, :, :, 0].transpose(0, 1)
                feat_yz = grid_sample(plane_yz, coords[..., [1, 2]],)[0, :, :, 0].transpose(0, 1)
            else:
                feat_xy = self.cu.grid_sample_2d(plane_xy, coords[..., [0, 1]], align_corners=False, padding_mode="border")[0, :, :,
                          0].transpose(0, 1)
                feat_xz = self.cu.grid_sample_2d(plane_xz, coords[..., [0, 2]], align_corners=False, padding_mode="border")[0, :, :,
                          0].transpose(0, 1)
                feat_yz = self.cu.grid_sample_2d(plane_yz, coords[..., [1, 2]], align_corners=False, padding_mode="border")[0, :, :,
                          0].transpose(0, 1)
            multi_feat_xy.append(feat_xy)
            multi_feat_xz.append(feat_xz)
            multi_feat_yz.append(feat_yz)
        # [N, features * L_number]
        feat_xy = torch.cat(multi_feat_xy, dim=1)
        feat_xz = torch.cat(multi_feat_xz, dim=1)
        feat_yz = torch.cat(multi_feat_yz, dim=1)
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        adapt_layer = self.all_persons_layer[person_id]
        feat = adapt_layer(feat)
        # implicit_layer = self.all_persons_implicit_layer[person_id]
        # feat = torch.cat([feat, x_embed, pose], dim=1)
        # feat = implicit_layer(feat)
        last_layer = self.last_layer[person_id]
        feat_dsdf = last_layer(feat)
        feat = feat_dsdf[..., :-1]
        dsdf = feat_dsdf[..., -1]
        # feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        # feat = (feat_xy + feat_xz + feat_yz) / 3
        feat = feat.reshape(*shape[:-1], self.n_output_dims)
        dsdf = dsdf.reshape(*shape[:-1], 1)
        return feat, dsdf