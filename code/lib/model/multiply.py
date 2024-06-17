from .networks import ImplicitNet, RenderingNet
from .density import LaplaceDensity, AbsDensity
from .ray_sampler import ErrorBoundSampler
from .deformer import SMPLDeformer
from .smpl import SMPLServer

from .sampler import PointInSpace, PointOnBones

from ..utils import rend_util

import numpy as np
import trimesh
import torch
import torch.nn as nn
from torch.autograd import grad
import hydra
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces
from pytorch3d import ops
from nerfacc import render_weight_from_density, pack_info, accumulate_along_rays
from .deformer import skinning
import json
class Multiply(nn.Module):
    def __init__(self, opt, betas_path):
        super().__init__()
        betas = np.load(betas_path)
        self.using_nerfacc = True
        self.init_params(opt)
        
        # human id encoder
        if self.use_person_encoder:
            assert len(betas.shape) == 2
            self.person_latent_encoder = nn.Embedding(betas.shape[0], 64)

        self.foreground_implicit_network_list = nn.ModuleList()
        self.foreground_rendering_network_list = nn.ModuleList()
        if len(betas.shape) == 2:
            if self.use_person_encoder:
                # we use shared network for all people
                implicit_model = ImplicitNet(opt.implicit_network, betas=betas)
                if opt.rendering_network.mode == 'pose_no_view':
                    print('use shared network only for shape network, separate for render network')
                    for i in range(betas.shape[0]):
                        self.foreground_implicit_network_list.append(implicit_model)
                        self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))
                else:
                    print('use shared network for all people')
                    render_model = RenderingNet(opt.rendering_network, triplane=implicit_model.triplane)
                    for i in range(betas.shape[0]):
                        self.foreground_implicit_network_list.append(implicit_model)
                        self.foreground_rendering_network_list.append(render_model)
            else:
                for i in range(betas.shape[0]):
                    self.foreground_implicit_network_list.append(ImplicitNet(opt.implicit_network))
                    self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))
        else:
            self.foreground_implicit_network_list.append(ImplicitNet(opt.implicit_network))
            self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))

        # Background networks
        self.with_bkgd = opt.with_bkgd
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(opt.num_training_frames, opt.dim_frame_encoding)
        self.sampler = PointInSpace()

        self.use_smpl_deformer = opt.use_smpl_deformer
        # self.gender = 'male'
        self.gender_list = np.load(betas_path[:-14] + "gender.npy")
        if self.use_smpl_deformer:
            self.deformer_list = torch.nn.ModuleList()
            # self.deformer_list = []
            if len(betas.shape) == 2:
                for i in range(betas.shape[0]):
                    deformer = SMPLDeformer(betas=betas[i], gender=self.gender_list[i])
                    self.deformer_list.append(deformer)
            else:
                deformer = SMPLDeformer(betas=betas, gender='male')
                self.deformer_list.append(deformer)


        # pre-defined bounding sphere
        self.sdf_bounding_sphere = 3.0
        
        # threshold for the out-surface points
        self.threshold = 0.05
        self.density = LaplaceDensity(**opt.density)
        self.bg_density = AbsDensity()

        self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler)
        self.smpl_server_list = torch.nn.ModuleList()
        if len(betas.shape) == 2:
            for i in range(betas.shape[0]):
                smpl_server = SMPLServer(gender=self.gender_list[i], betas=betas[i])
                self.smpl_server_list.append(smpl_server)
        else:
            smpl_server = SMPLServer(gender='male', betas=betas)
            self.smpl_server_list.append(smpl_server)

        if opt.smpl_init:
            if self.use_person_encoder:
                smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_256_id.pth' % 'male'))
            else:
                smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_256.pth' % 'male'))
            for implicit_network in self.foreground_implicit_network_list:
                implicit_network.load_state_dict(smpl_model_state["model_state_dict"], strict=False)
            if not self.use_smpl_deformer:
                self.deformer.load_state_dict(smpl_model_state["deformer_state_dict"])

        if self.smpl_surface_weight > 0:
            self.smpl_vertex_part = json.load(open(hydra.utils.to_absolute_path('./outputs/smpl_vert_segmentation.json')))

        self.mesh_v_cano_list = []
        self.mesh_f_cano_list = []
        self.mesh_face_vertices_list = []
        for smpl_server in self.smpl_server_list:
            self.mesh_v_cano_list.append(smpl_server.verts_c)
            self.mesh_f_cano_list.append(torch.tensor(smpl_server.smpl.faces.astype(np.int64), device=smpl_server.verts_c.device))
            self.mesh_face_vertices_list.append(index_vertices_by_faces(self.mesh_v_cano_list[-1], self.mesh_f_cano_list[-1]))


    def init_params(self, opt):
        self.smpl_surface_weight = opt.loss.get('smpl_surface_weight', 0)
        self.zero_pose_weight = opt.loss.get('zero_pose_weight', 0)
        self.use_person_encoder = opt.get('use_person_encoder', False)

    def get_deformed_mesh_fast_mode_multiple_person(self, verts, smpl_tfs, person_id):
        # verts = torch.tensor(verts).cuda().float()
        weights = self.deformer_list[person_id].query_weights(verts[0])
        verts_deformed = skinning(verts, weights, smpl_tfs)
        # verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed


    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts, person_id):
        if hasattr(self, "deformer_list"):
            x_c, outlier_mask = self.deformer_list[person_id].forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
            output = self.foreground_implicit_network_list[person_id](x_c, cond, person_id=person_id)[0]
            sdf = output[:, 0:1]
            if not self.training:
                sdf[outlier_mask] = 4. # set a large SDF value for outlier points
            ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
            if not self.with_bkgd:
                if self.sdf_bounding_sphere > 0.0:
                    sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                    sdf = torch.minimum(sdf, sphere_sdf)
            feature = output[:, 1:]

        return sdf, x_c, feature
    
    def check_off_in_surface_points_cano_mesh(self, x_cano, N_samples, person_id, threshold=0.05):

        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(x_cano.unsqueeze(0).contiguous(), self.mesh_face_vertices_list[person_id])

        distance = torch.sqrt(distance) # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(self.mesh_v_cano_list[person_id], self.mesh_f_cano_list[person_id], x_cano.unsqueeze(0)).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.).squeeze(1)
        return index_off_surface, index_in_surface

    def query_oc(self, x, cond, person_id):

        x = x.reshape(-1, 3)
        mnfld_pred = self.foreground_implicit_network_list[person_id](x, cond, person_id=person_id)[:,:,0].reshape(-1,1)
        return {'occ':mnfld_pred}
    def forward(self, input, id=-1, cond_zero_shit=False, canonical_pose=False):
        # Parse model input
        torch.set_grad_enabled(True)
        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        smpl_params = input['smpl_params']
        smpl_pose = input["smpl_pose"]
        scale = smpl_params[:,:, 0]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        # (B, P, 3)
        num_person = smpl_trans.shape[1]
        smpl_tfs_list = []
        smpl_output_list = []
        raymeshintersector_list = []
        oriented_box_mesh_list = []
        ray_box_intersector_list = []
        verts_deformed_list = []

        # check wheather the ray will intersect with certain SMPL
        for i in range(num_person):
            if canonical_pose:
                canonical_smpl_pose = torch.zeros_like(smpl_pose[:,i])
                canonical_smpl_trans = torch.zeros_like(smpl_trans[:,i])
                canonical_smpl_pose[0, 5] = np.pi/6
                canonical_smpl_pose[0, 8] = - np.pi / 6
                smpl_output = self.smpl_server_list[i](scale[:, i], canonical_smpl_trans, canonical_smpl_pose, smpl_shape[:, i])
            else:
                smpl_output = self.smpl_server_list[i](scale[:,i], smpl_trans[:,i], smpl_pose[:,i], smpl_shape[:,i])
            smpl_output_list.append(smpl_output)
            smpl_tfs_list.append(smpl_output['smpl_tfs'])
            # load into trimesh
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output['smpl_verts'][0].detach().cpu().numpy(),
                                        faces=self.smpl_server_list[i].faces, process=False)
            oriented_box = smpl_mesh.bounding_box_oriented.copy()
            # https://github.com/mikedh/trimesh/issues/1865 So in this case mesh.bounding_box_oriented.primitive.extents is the pre-transform extents (which I think you probably want) where mesh.bounding_box_oriented.extents is the AABB of the transformed box.
            extented_oriented_box = trimesh.primitives.Box(oriented_box.primitive.extents * 1.2, oriented_box.transform)
            oriented_box_mesh_list.append(extented_oriented_box.to_mesh())
            ray_box_intersector_list.append(trimesh.ray.ray_triangle.RayMeshIntersector(extented_oriented_box.to_mesh()))
            # comfirm that camera ray is aligned with scaled smpl



        # smpl_params = smpl_params.detach()
        # smpl_mesh = trimesh.Trimesh(vertices=smpl_output['smpl_verts'][0].detach().cpu().numpy(), faces=self.smpl_server.faces, process=False)
        # import pdb;pdb.set_trace()

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        fg_rgb_list = []
        normal_values_list = []
        sdf_output_list = []
        z_vals_list = []
        person_id_list = []
        z_max_list = []
        index_off_surface_list = []
        index_in_surface_list = []
        grad_theta_list = []
        interpenetration_loss = torch.zeros(1,device=smpl_pose.device)
        temporal_loss = torch.zeros(1, device=smpl_pose.device)
        smpl_surface_loss = torch.zeros(1, device=smpl_pose.device)
        zero_pose_loss = torch.zeros(1, device=smpl_pose.device)
        # if input['current_epoch'] < 500 and input['idx'] > 85 and input['idx'] < 125:
        if self.training and input['current_epoch'] > 250:
            temporal_loss = torch.mean(torch.square(input["smpl_pose_last"] - input["smpl_pose"]))
        if id == -1:
            person_list = range(num_person)
        else:
            person_list = [id]

        hitted_face_idx_list = []
        index_triangle_list = []
        index_ray_list = []
        locations_list = []
        index_ray_box_list = []
        for person_id in person_list:
            if self.using_nerfacc:
                index_triangle_box, index_ray_box, locations_box = ray_box_intersector_list[person_id].intersects_id(cam_loc.cpu(), ray_dirs.cpu(), multiple_hits=False, return_locations=True)
                # import pdb; pdb.set_trace()
                sorted_idx = np.argsort(index_ray_box)
                index_ray_box = index_ray_box[sorted_idx]
                index_triangle_box = index_triangle_box[sorted_idx]
                locations_box = locations_box[sorted_idx]
                if len(index_ray_box) == 0:
                    index_ray_box = np.array([0])
                index_ray_box = torch.from_numpy(index_ray_box).long().to(cam_loc.device)
                cam_loc_intersect = cam_loc[index_ray_box]
                ray_dirs_intersect = ray_dirs[index_ray_box]
                index_ray_box_list.append(index_ray_box)


            cond_pose = smpl_pose[:, person_id, 3:] / np.pi
            if self.training:
                if input['current_epoch'] < 20 or input['current_epoch'] % 20 == 0 or cond_zero_shit:
                    cond_pose = smpl_pose[:, person_id, 3:] * 0.

            if self.use_person_encoder:
                person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                person_encoding = self.person_latent_encoder(person_id_tensor)
                person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                cond = {'smpl_id': cond_pose_id}
            else:
                cond = {'smpl': cond_pose}

            if self.using_nerfacc:
                z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs_intersect, cam_loc_intersect, self, cond, smpl_tfs_list[person_id],
                                                        eval_mode=True,
                                                        smpl_verts=smpl_output_list[person_id]['smpl_verts'],
                                                        person_id=person_id)
                z_vals, z_vals_bg = z_vals
                z_max = z_vals[:, -1]
                z_vals = z_vals[:, :-1]
                N_samples = z_vals.shape[1]
                num_pixels = cam_loc_intersect.shape[0]
                points = cam_loc_intersect.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_intersect.unsqueeze(1)
                points_flat = points.reshape(-1, 3)
                dirs = ray_dirs_intersect.unsqueeze(1).repeat(1, N_samples, 1)
            else:
                z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, cond, smpl_tfs_list[person_id], eval_mode=True, smpl_verts=smpl_output_list[person_id]['smpl_verts'], person_id=person_id)
                z_vals, z_vals_bg = z_vals
                z_max = z_vals[:,-1]
                z_vals = z_vals[:,:-1]
                N_samples = z_vals.shape[1]
                points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
                points_flat = points.reshape(-1, 3)
                dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)


            sdf_output, canonical_points, feature_vectors = self.sdf_func_with_smpl_deformer(points_flat, cond, smpl_tfs_list[person_id], smpl_verts=smpl_output_list[person_id]['smpl_verts'], person_id=person_id)

            sdf_output = sdf_output.unsqueeze(1)

            if self.training:
                if input['current_epoch'] < 250:
                    index_off_surface_person, index_in_surface_person = self.check_off_in_surface_points_cano_mesh(canonical_points, N_samples, person_id=person_id, threshold=self.threshold)
                    index_in_surface_list.append(index_in_surface_person)
                    index_off_surface_list.append(index_off_surface_person)
                canonical_points = canonical_points.reshape(num_pixels, N_samples, 3)

                canonical_points = canonical_points.reshape(-1, 3)

                # sample canonical SMPL surface pnts for the eikonal loss
                smpl_verts_c = self.smpl_server_list[person_id].verts_c.repeat(batch_size, 1,1)
                number_eikonal_points = 512
                indices = torch.randperm(smpl_verts_c.shape[1])[:number_eikonal_points].cuda()
                verts_c = torch.index_select(smpl_verts_c, 1, indices)
                sample = self.sampler.get_points(verts_c, global_ratio=0.)

                sample.requires_grad_()
                local_pred = self.foreground_implicit_network_list[person_id](sample, cond, person_id=person_id)[..., 0:1]
                grad_theta_person = gradient(sample, local_pred)
                grad_theta_list.append(grad_theta_person)

                differentiable_points = canonical_points

                # sample point form deformed SMPL
                if self.smpl_surface_weight > 0:
                    number_surface_loss_points = num_pixels
                    assert smpl_output_list[person_id]['smpl_verts'].shape[1] == 6890
                    idx_weight = torch.ones(6890)
                    exclude_idx = self.smpl_vertex_part['head'] + self.smpl_vertex_part['rightHand'] + \
                                  self.smpl_vertex_part['leftHand'] + self.smpl_vertex_part['rightFoot'] + \
                                  self.smpl_vertex_part['leftFoot'] + self.smpl_vertex_part['leftHandIndex1'] + \
                                  self.smpl_vertex_part['rightHandIndex1']
                    idx_weight[exclude_idx] = 0
                    idx_weight = idx_weight.cuda()
                    idx = idx_weight.multinomial(num_samples=number_surface_loss_points, replacement=True)
                    # idx = torch.randperm(smpl_output_list[person_id]['smpl_verts'].shape[1])[:number_surface_loss_points].cuda()
                    sample_point = torch.index_select(smpl_output_list[person_id]['smpl_verts'], dim=1, index=idx)
                    x_c, outlier_mask = self.deformer_list[person_id].forward(sample_point.reshape(-1, 3),
                                                                               smpl_tfs_list[person_id],
                                                                               return_weights=False,
                                                                               inverse=True,
                                                                               smpl_verts=smpl_output_list[person_id][
                                                                                   'smpl_verts'])
                    output = self.foreground_implicit_network_list[person_id](x_c, cond, person_id=person_id)[0]
                    sdf = output[:, 0:1]
                    sdf = sdf.reshape(-1)
                    threshold_smpl_sdf = 0.02
                    if (sdf > threshold_smpl_sdf).any():
                        unplausible_sdf = sdf[sdf > threshold_smpl_sdf]
                        sdf_zero = torch.ones_like(unplausible_sdf, device=sdf.device) * threshold_smpl_sdf
                        smpl_surface_loss += torch.nn.functional.l1_loss(unplausible_sdf, sdf_zero, reduction='mean')

                sample_pixel = 2000
                if self.zero_pose_weight > 0:
                    for p, canonical_average_vertex in enumerate(self.mesh_v_cano_list):
                        idx = torch.randperm(canonical_average_vertex.shape[1])[:sample_pixel].cuda()
                        sample_point = torch.index_select(canonical_average_vertex, dim=1, index=idx)
                        output_pred = \
                        self.foreground_implicit_network_list[p](sample_point.reshape(-1, 3), cond, person_id=p)[0]
                        sdf_pred = output_pred[:, 0:1]

                        cond_zero_pose = smpl_pose[:, person_id, 3:] * 0.
                        if self.use_person_encoder:
                            # import pdb;pdb.set_trace()
                            person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                            person_encoding = self.person_latent_encoder(person_id_tensor)
                            person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                            cond_zero_pose_id = torch.cat([cond_zero_pose, person_encoding], dim=1)
                            cond_zero = {'smpl_id': cond_zero_pose_id}
                        else:
                            cond_zero = {'smpl': cond_zero_pose}

                        output_zero = \
                        self.foreground_implicit_network_list[p](sample_point.reshape(-1, 3), cond_zero, person_id=p)[0]
                        sdf_zero = output_zero[:, 0:1]
                        zero_pose_loss += torch.nn.functional.l1_loss(sdf_pred, sdf_zero, reduction='mean')
                        feature_zero = output_zero[:, 1:]
                        feature_pred = output_pred[:, 1:]
                        zero_pose_loss += torch.nn.functional.l1_loss(feature_pred, feature_zero, reduction='mean')

            else:
                differentiable_points = canonical_points.reshape(num_pixels, N_samples, 3).reshape(-1, 3)
                grad_theta = None

            sdf_output = sdf_output.reshape(num_pixels, N_samples, 1).reshape(-1, 1)
            sdf_output_list.append(sdf_output.reshape(num_pixels, N_samples))
            person_label = torch.ones_like(sdf_output.reshape(num_pixels, N_samples), device=sdf_output.device) * person_id
            z_vals = z_vals
            view = -dirs.reshape(-1, 3)

            if differentiable_points.shape[0] > 0:
                fg_rgb_flat, others = self.get_rbg_value(points_flat, differentiable_points, view,
                                                         cond, smpl_tfs_list[person_id], feature_vectors=feature_vectors, person_id=person_id, is_training=self.training)
                normal_values = others['normals']

            if 'image_id' in input.keys():
                frame_latent_code = self.frame_latent_encoder(input['image_id'])
            else:
                frame_latent_code = self.frame_latent_encoder(input['idx'])

            fg_rgb = fg_rgb_flat.reshape(-1, N_samples, 3)
            fg_rgb_list.append(fg_rgb)
            normal_values = normal_values.reshape(-1, N_samples, 3)
            normal_values_list.append(normal_values)
            z_max_list.append(z_max)
            z_vals_list.append(z_vals)
            person_id_list.append(person_label)

        fg_rgb_values_each_person_list = []
        t_list = []
        mean_hitted_vertex_list = []
        hitted_mask_idx = []
        
        if self.using_nerfacc:
            # reformat to fit nerfacc input
            index_ray_box_repeat = [index_ray_box_i.unsqueeze(1).repeat(1, N_samples).flatten() for index_ray_box_i in index_ray_box_list]
            index_ray_box_repeat_flatten = torch.cat(index_ray_box_repeat, dim=0).unsqueeze(-1)
            z_vals_max = [torch.cat([z_vals_i, z_max_i.unsqueeze(-1)], dim=1) for z_vals_i, z_max_i in zip(z_vals_list, z_max_list)]
            z_vals_start = [z_vals_i[:, :-1].flatten() for z_vals_i in z_vals_max]
            z_vals_end = [z_vals_i[:, 1:].flatten() for z_vals_i in z_vals_max]
            z_vals_start_flatten = torch.cat(z_vals_start, dim=0).unsqueeze(-1)
            z_vals_end_flatten = torch.cat(z_vals_end, dim=0).unsqueeze(-1)
            sdf_output_flatten = [sdf_output_i.flatten() for sdf_output_i in sdf_output_list]
            sdf_output_flatten = torch.cat(sdf_output_flatten, dim=0).unsqueeze(-1)
            fg_rgb_flatten = [fg_rgb_i.reshape(-1, 3) for fg_rgb_i in fg_rgb_list]
            fg_rgb_flatten = torch.cat(fg_rgb_flatten, dim=0)
            normal_values_flatten = [normal_values_i.reshape(-1, 3) for normal_values_i in normal_values_list]
            normal_values_flatten = torch.cat(normal_values_flatten, dim=0)
            person_id_flatten = [person_id_i.flatten() for person_id_i in person_id_list]
            person_id_flatten = torch.cat(person_id_flatten, dim=0).unsqueeze(-1)
            ray_zs_ze_sdf_rgb_normal_id = torch.cat([index_ray_box_repeat_flatten, z_vals_start_flatten, z_vals_end_flatten, sdf_output_flatten, fg_rgb_flatten, normal_values_flatten, person_id_flatten], dim=1)
            z_vals_sorted, sorted_index = torch.sort(ray_zs_ze_sdf_rgb_normal_id[:, 2], descending=False, dim=0)
            ray_zs_ze_sdf_rgb_normal_id = ray_zs_ze_sdf_rgb_normal_id[sorted_index]
            index_ray_sorted, ray_sorted_index = torch.sort(ray_zs_ze_sdf_rgb_normal_id[:, 0], descending=False, dim=0, stable=True)
            ray_zs_ze_sdf_rgb_normal_id = ray_zs_ze_sdf_rgb_normal_id[ray_sorted_index]
            ray_indices = ray_zs_ze_sdf_rgb_normal_id[:, 0].long()
            t_starts = ray_zs_ze_sdf_rgb_normal_id[:, 1]
            t_ends = ray_zs_ze_sdf_rgb_normal_id[:, 2]
            sigmas = self.density(ray_zs_ze_sdf_rgb_normal_id[:, 3])
            fg_rgb_flatten_sorted = ray_zs_ze_sdf_rgb_normal_id[:, 4:7]
            normal_values_flatten_sorted = ray_zs_ze_sdf_rgb_normal_id[:, 7:10]
            person_id_flatten_sorted = ray_zs_ze_sdf_rgb_normal_id[:, 10]
            n_rays = cam_loc.shape[0]
            weights, transmittance, alphas = render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=n_rays)
            packed_info = pack_info(ray_indices, n_rays=n_rays)
            packed_info_valid = packed_info[packed_info[:, 1] != 0]
            last_transmittance_index = packed_info_valid[1:, 0].long() - 1
            last_transmittance_index = torch.cat([last_transmittance_index,torch.tensor([weights.shape[0]-1],device=weights.device)], dim=0)
            bg_transmittance_valid = transmittance[last_transmittance_index]
            bg_transmittance_indices = ray_indices[last_transmittance_index]
            bg_transmittance = torch.ones((n_rays)).to(weights.device)
            bg_transmittance[bg_transmittance_indices] = bg_transmittance_valid
            # accumulate rgb
            acc_fg_rgb = accumulate_along_rays(weights, fg_rgb_flatten_sorted, ray_indices, n_rays=n_rays)
            fg_rgb_values = acc_fg_rgb
            # accumulate normal
            acc_normal = accumulate_along_rays(weights, normal_values_flatten_sorted, ray_indices, n_rays=n_rays)
            normal_values = acc_normal
            # accumulate weight
            acc_weight = accumulate_along_rays(weights, ray_indices=ray_indices, n_rays=n_rays).reshape(-1)
            acc_person_list = []
            for person_idx in person_list:
                person_mask = (person_id_flatten_sorted == person_idx)
                weight_person = weights[person_mask]
                ray_indices_person = ray_indices[person_mask]
                # (number_ray, 1)
                acc_weight_person = accumulate_along_rays(weight_person, ray_indices=ray_indices_person, n_rays=n_rays)
                acc_person_list.append(acc_weight_person.reshape(-1))
            acc_person = torch.stack(acc_person_list, dim=1)

            z_vals_inverse_sphere = self.ray_sampler.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, self)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1. / self.ray_sampler.scene_bounding_sphere)
            z_vals_bg = z_vals_inverse_sphere
        else:
            person_id_cat = torch.cat(person_id_list, dim=1)
            z_vals = torch.cat(z_vals_list, dim=1)
            sdf_output = torch.cat(sdf_output_list, dim=1)
            fg_rgb = torch.cat(fg_rgb_list, dim=1)
            normal_values = torch.cat(normal_values_list, dim=1)
            z_vals, sorted_index = torch.sort(z_vals,descending=False,dim=1)

            d1, d2 = sorted_index.shape
            d1_index = torch.arange(d1).unsqueeze(1).repeat((1, d2))
            sdf_output = sdf_output[d1_index, sorted_index]
            fg_rgb = fg_rgb[d1_index, sorted_index]
            normal_values = normal_values[d1_index, sorted_index]
            person_id_cat = person_id_cat[d1_index, sorted_index]
            weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf_output)
            fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 1)
            acc_person_list = []
            for person_id in person_list:
                number_pixel = weights.shape[0]
                weight_idx = (person_id_cat==person_id).reshape(-1)
                weight_person_i = weights.reshape(-1)[weight_idx]
                weight_person_i = weight_person_i.reshape(number_pixel, -1)
                acc_person_i = torch.sum(weight_person_i, dim=-1)
                acc_person_list.append(acc_person_i)
            acc_person = torch.stack(acc_person_list, dim=1)
            acc_weight = torch.sum(weights, -1)
            normal_values = torch.sum(weights.unsqueeze(-1) * normal_values, 1)
        
        # Background rendering
        if input['idx'] is not None:
            N_bg_samples = z_vals_bg.shape[1]
            z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
            bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

            bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
            bg_points_flat = bg_points.reshape(-1, 4)
            bg_dirs_flat = bg_dirs.reshape(-1, 3)
            bg_output = self.bg_implicit_network(bg_points_flat, {'frame': frame_latent_code})[0]
            bg_sdf = bg_output[:, :1]
            bg_feature_vectors = bg_output[:, 1:]
            
            bg_rendering_output = self.bg_rendering_network(None, None, bg_dirs_flat, None, bg_feature_vectors, frame_latent_code)
            if bg_rendering_output.shape[-1] == 4:
                bg_rgb_flat = bg_rendering_output[..., :-1]
                shadow_r = bg_rendering_output[..., -1]
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
                shadow_r = shadow_r.reshape(-1, N_bg_samples, 1)
                bg_rgb = (1 - shadow_r) * bg_rgb
            else:
                bg_rgb_flat = bg_rendering_output
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)
            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)
        else:
            bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)

        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values


        if self.training:
            if input['current_epoch'] < 250:
                if self.using_nerfacc:
                    index_off_surface = torch.tensor(np.ones((n_rays, len(person_list))), dtype=torch.bool, device=rgb_values.device)
                    index_in_surface = torch.tensor(np.zeros((n_rays, len(person_list))), dtype=torch.bool, device=rgb_values.device)
                    for p, (ray_index, index_off, index_in) in enumerate(zip(index_ray_box_list, index_off_surface_list, index_in_surface_list)):
                        index_off_surface[ray_index, p] = index_off
                        index_in_surface[ray_index, p] = index_in
                    index_off_surface = torch.all(index_off_surface, dim=1)
                    index_in_surface = torch.any(index_in_surface, dim=1)
                else:
                    index_off_surface = torch.all(torch.stack(index_off_surface_list, dim=0), dim=0)
                    index_in_surface = torch.any(torch.stack(index_in_surface_list, dim=0), dim=0)
            else:
                index_off_surface = None
                index_in_surface = None
            # import pdb; pdb.set_trace()
            grad_theta = torch.cat(grad_theta_list, dim=1)
            output = {
                'zero_pose_loss': zero_pose_loss,
                't_list': t_list,
                'fg_rgb_values_each_person_list' : fg_rgb_values_each_person_list,
                'cam_loc': cam_loc,
                'hitted_mask_idx': hitted_mask_idx,
                'mean_hitted_vertex_list': mean_hitted_vertex_list,
                'points': points,
                'rgb_values': rgb_values,
                'normal_values': normal_values,
                'index_outside': input['index_outside'],
                'index_off_surface': index_off_surface,
                'index_in_surface': index_in_surface,
                'acc_map': acc_weight,
                'grad_theta': grad_theta,
                'interpenetration_loss': interpenetration_loss,
                'temporal_loss': temporal_loss,
                'acc_person_list': acc_person, # Number pixel, Number person
                'smpl_surface_loss': smpl_surface_loss,
                'epoch': input['current_epoch'],
            }
            if 'sam_mask' in input.keys():
                output.update({"sam_mask": input['sam_mask'].squeeze()})
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(-1) * torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            output = {
                'acc_map': acc_weight,
                'acc_person_list': acc_person,
                'rgb_values': rgb_values,
                'fg_rgb_values': fg_output_rgb,
                'normal_values': normal_values,
            }
        return output

    def get_rbg_value(self, x, points, view_dirs, cond, tfs, feature_vectors, person_id, is_training=True):
        pnts_c = points
        others = {}

        _, gradients, feature_vectors = self.forward_gradient(x, pnts_c, cond, tfs, person_id, create_graph=is_training, retain_graph=is_training)
        # ensure the gradient is normalized
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        if self.use_person_encoder:
            tri_feat = self.foreground_implicit_network_list[person_id].triplane_feature_list[person_id]
            fg_rendering_output = self.foreground_rendering_network_list[person_id](pnts_c, normals, view_dirs,
                                                                                    cond['smpl_id'][:, :69],
                                                                                    feature_vectors, person_id=person_id, id_latent_code=cond['smpl_id'][:,69:], tri_feat=tri_feat)
        else:
            fg_rendering_output = self.foreground_rendering_network_list[person_id](pnts_c, normals, view_dirs, cond['smpl'],
                                                     feature_vectors, person_id=person_id)
        
        rgb_vals = fg_rendering_output[:, :3]
        others['normals'] = normals
        return rgb_vals, others

    def forward_gradient(self, x, pnts_c, cond, tfs, person_id, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer_list[person_id].forward_skinning(pnts_c.unsqueeze(0), None, tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            # import ipdb;ipdb.set_trace()
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.foreground_implicit_network_list[person_id](pnts_c, cond, person_id=person_id)[0]
        sdf = output[:, :1]

        if not self.with_bkgd:
            if self.sdf_bounding_sphere > 0.0:
                sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                sdf = torch.minimum(sdf, sphere_sdf)
        
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]

        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1), feature

    def volume_rendering(self, z_vals, z_max, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        # included also the dist from the sphere intersection
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights
    
    def depth2pts_outside(self, ray_o, ray_d, depth):

        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.sdf_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.sdf_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts

def gradient(inputs, outputs):

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :, -3:]
    return points_grad