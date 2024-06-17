import torch.nn as nn
import torch
import numpy as np
from .embedders import get_embedder
from .triplane import TriPlane, TriPlaneMulti

class ImplicitNet(nn.Module):
    def __init__(self, opt, betas=None):
        super().__init__()
        self.init_params(opt)
        dims = [opt.d_in] + list(
            opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt
        self.triplane = None
        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = opt.cond   
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 69
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = 32
        elif self.cond == 'smpl_id':
            self.cond_layer = [0]
            self.cond_dim = 69 + 64
        elif self.cond == 'smpl_tri':
            self.cond_layer = [0]
            self.cond_dim = 69 + 64
            # TODO add number_person in opt
            if self.multi_triplane:
                self.triplane = TriPlaneMulti(opt.number_person, features=64, opt=opt)
            else:
                self.triplane = TriPlane(opt.number_person, features=64)
            self.triplane_feature_list = [None for _ in range(opt.number_person)]
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)
        if self.offset_head:
            self.head_layer_list = nn.ModuleList()
            self.last_layer_list = nn.ModuleList()
            for i in range(opt.number_person):
                head_layer = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(256 + 1 + 69 + 64 + dims[0], 256)),
                    nn.Softplus(beta=100),
                    nn.utils.weight_norm(nn.Linear(256, 256)),
                    nn.Softplus(beta=100),
                    nn.utils.weight_norm(nn.Linear(256, 256)),
                    nn.Softplus(beta=100),
                    nn.utils.weight_norm(nn.Linear(256, 256)),
                    nn.Softplus(beta=100),
                )
                self.head_layer_list.append(head_layer)
                last_layer = nn.Linear(256, 256 + 1)
                init_val = 1e-6
                torch.nn.init.constant_(last_layer.bias, 0.0)
                torch.nn.init.uniform_(last_layer.weight, -init_val, init_val)
                last_layer = nn.utils.weight_norm(last_layer)
                self.last_layer_list.append(last_layer)
        if self.beta_encoding:
            self.betas = betas
            self.beta_layer_list = nn.ModuleList()
            for i in range(opt.number_person):
                beta_layer = nn.Linear(10, 256)
                init_val = 1e-5
                torch.nn.init.constant_(beta_layer.bias, 0.0)
                torch.nn.init.uniform_(beta_layer.weight, -init_val, init_val)
                beta_layer = nn.utils.weight_norm(beta_layer)
                self.beta_layer_list.append(beta_layer)


    def init_params(self, opt):
        self.multi_triplane = opt.get('multi_triplane', False)
        self.offset_head = opt.get('offset_head', False)
        self.beta_encoding = opt.get('beta_encoding', False)
        self.no_head_feature = opt.get('no_head_feature', False)

    
    def forward(self, input, cond, current_epoch=None, person_id=-1):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)
        dsdf = None
        if self.cond != 'none':
            if self.cond == 'smpl_tri':
                # need to normalize the triplane encoding
                # here I divide input by 2 to prevent the input away from the range of triplane [-1, 1]
                assert person_id != -1
                cond_pose = cond['smpl_id'][:, :69]
                num_batch, num_cond = cond_pose.shape
                input_cond_pose = cond_pose.unsqueeze(1).expand(num_batch, num_point, num_cond)
                input_cond_pose = input_cond_pose.reshape(num_batch * num_point, num_cond)
                tri_encoding, dsdf = self.triplane(input / 2, person_id, input_cond_pose)
                self.triplane_feature_list[person_id] = tri_encoding
                input_cond = torch.cat([input_cond_pose, tri_encoding], dim=1)
            else:
                num_batch, num_cond = cond[self.cond].shape

                input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

                input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if self.beta_encoding and l==0:
                assert person_id != -1
                beta_layer = self.beta_layer_list[person_id]
                beta = self.betas[person_id]
                beta = torch.from_numpy(beta).float().cuda()
                beta = beta.reshape(1,1,10).expand(num_batch, num_point, 10)
                beta = beta.reshape(num_batch * num_point, 10)
                beta = beta_layer(beta)
                x = x + beta

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = x.reshape(num_batch, num_point, -1)
        # since we have separate head, we do not need dsdf here
        # if dsdf is not None:
        #     sdf = x[:, :, 0:1]
        #     feat = x[:, :, 1:]
        #     sdf = sdf + dsdf.reshape(num_batch, num_point, 1)
        #     x = torch.cat([sdf, feat], dim=-1)

        if self.offset_head:
            assert person_id != -1
            head_layer = self.head_layer_list[person_id]
            last_layer = self.last_layer_list[person_id]
            input_head = torch.cat([x.reshape(num_batch * num_point, 1 + 256), input_cond, input], dim=1)
            feat_dsdf = head_layer(input_head)
            feat_dsdf = last_layer(feat_dsdf)
            feat_dsdf = feat_dsdf.reshape(num_batch, num_point, -1)
            sdf = x[:, :, 0:1]
            dsdf = feat_dsdf[:, :, 0:1]
            if self.no_head_feature:
                feat = x[:, :, 1:]
            else:
                feat = feat_dsdf[:, :, 1:]
            sdf = sdf + dsdf
            x = torch.cat([sdf, feat], dim=-1)
            x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt, triplane=None):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(
            opt.dims) + [opt.d_out]

        self.embedview_fn = None
        self.triplane = triplane
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        if self.mode == 'nerf_frame_encoding':
            dims[0] += 32
        if self.mode == 'pose_no_view':
            self.dim_cond_embed = 8
            self.cond_dim = 69
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        if self.mode == 'pose_id_no_view':
            self.dim_cond_embed = 8
            self.cond_dim = 69
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            self.lin_id = torch.nn.Linear(64, 8)
        if self.mode == 'pose_tri_no_view':
            self.dim_cond_embed = 8
            self.cond_dim = 69
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            self.lin_id = torch.nn.Linear(64, 8)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, frame_latent_code=None, id_latent_code=None, person_id=-1, tri_feat=None):
        if self.embedview_fn is not None:
            if self.mode == 'nerf_frame_encoding':
                view_dirs = self.embedview_fn(view_dirs)
            elif self.mode == 'pose_no_view':
                points = self.embedview_fn(points)
            elif self.mode == 'idr':
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf_frame_encoding':
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'pose_no_view':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        elif self.mode == 'pose_id_no_view':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            id_latent_code = id_latent_code.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            id_latent_code = self.lin_id(id_latent_code)
            rendering_input = torch.cat([points, normals, body_pose, id_latent_code, feature_vectors], dim=-1)
        elif self.mode == "pose_tri_no_view":
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            # TODO need to normalize
            # TODO here I divide input by 2 to prevent the input away from the range of triplane [-1, 1]
            assert person_id != -1
            # id_latent_code, dsdf = self.triplane(points/2, person_id, body_pose)
            id_latent_code = tri_feat
            body_pose = self.lin_pose(body_pose)
            id_latent_code = self.lin_id(id_latent_code)
            rendering_input = torch.cat([points, normals, body_pose, id_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x
