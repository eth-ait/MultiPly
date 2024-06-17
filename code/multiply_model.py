import pytorch_lightning as pl
import torch.optim as optim
from lib.model.multiply import Multiply
from lib.model.body_model_params import BodyModelParams
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
from lib.utils.mesh import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import trimesh
from lib.model.deformer import skinning
from lib.utils import idr_utils
from lib.datasets import create_dataset
from tqdm import tqdm
from lib.model.render import Renderer
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from lib.model.sam_model import SAMServer
from torch import nn
import kaolin
from pytorch3d import ops
from lib.datasets.Hi4D import weighted_sampling
class MultiplyModel(pl.LightningModule):
    def __init__(self, opt, betas_path) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.init_params(opt)
        self.nerfacc = True
        self.model = Multiply(opt.model, betas_path)
        self.opt = opt
        self.num_training_frames = opt.model.num_training_frames
        self.start_frame = opt.dataset.train.start_frame
        self.end_frame = opt.dataset.train.end_frame
        self.training_indices = list(range(self.start_frame, self.end_frame))
        assert len(self.training_indices) == self.num_training_frames
        self.opt_smpl = True
        self.training_modules = ["model"]
        self.num_person = opt.dataset.train.num_person
        if self.opt_smpl:
            self.body_model_list = torch.nn.ModuleList()
            for i in range(self.num_person):
                body_model_params = BodyModelParams(opt.model.num_training_frames, model_type='smpl')
                self.body_model_list.append(body_model_params)
                self.load_body_model_params(i)
                optim_params = self.body_model_list[i].param_names
                for param_name in optim_params:
                    self.body_model_list[i].set_requires_grad(param_name, requires_grad=True)
            self.training_modules += ['body_model_list']

        self.loss = Loss(opt.model.loss)
        self.sam_server = SAMServer(opt.dataset.train)
        self.using_sam = opt.dataset.train.using_SAM
        self.pose_correction_epoch = opt.model.pose_correction_epoch
        self.sigmoid = nn.Sigmoid()
        self.l2_loss = nn.MSELoss(reduction='mean')


    def init_params(self, opt):
        # depth end determines whether apply depth & interpenetration loss during or at the end of epochs.
        self.depth_end = opt.model.get('depth_end', False)
        # only used when depth_end is False
        self.pose_start_epoch = opt.model.get('pose_start_epoch', 200)
        self.pose_end_epoch = opt.model.get('pose_end_epoch', 1000)
        self.pose_opt_interval = opt.model.get('pose_opt_interval', 10)
        self.pose_opt_epoch = opt.model.get('pose_opt_epoch', 1)
        # only used when depth_end is True
        self.depth_pose = opt.model.get('depth_pose', False)
        self.depth_epoch = opt.model.get('depth_epoch', [])
        self.depth_cond_zero = opt.model.get('depth_cond_zero', False)
        self.it_per_loop = opt.model.get('it_per_loop', 100)

        self.all_edge = opt.model.get('all_edge', False)
        if self.all_edge:
            assert opt.dataset.train.edge_sampling == self.all_edge

        
    def load_body_model_params(self, index):
        body_model_params = {param_name: [] for param_name in self.body_model_list[index].param_names}
        data_root = os.path.join('../data', self.opt.dataset.train.data_dir)
        data_root = hydra.utils.to_absolute_path(data_root)

        body_model_params['betas'] = torch.tensor(np.load(os.path.join(data_root, 'mean_shape.npy'))[None, index], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices][:, index, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices] [:, index, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(np.load(os.path.join(data_root, 'normalize_trans.npy'))[self.training_indices][:, index], dtype=torch.float32)

        for param_name in body_model_params.keys():
            self.body_model_list[index].init_parameters(param_name, body_model_params[param_name], requires_grad=False)

    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr':self.opt.model.learning_rate}]
        if self.opt_smpl:
            params.append({'params': self.body_model_list.parameters(), 'lr':self.opt.model.learning_rate*0.1})
        self.optimizer_joint = optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler_joint = optim.lr_scheduler.MultiStepLR(
            self.optimizer_joint, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)

        params_pose = [{'params': self.body_model_list.parameters(), 'lr':self.opt.model.learning_rate*0.1}]
        self.optimizer_pose = optim.Adam(params_pose, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler_pose = optim.lr_scheduler.MultiStepLR(
            self.optimizer_pose, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return [self.optimizer_joint, self.optimizer_pose], [self.scheduler_joint, self.scheduler_pose]


    def freeze_shape_model(self):
        for param in self.model.foreground_implicit_network_list.parameters():
            param.requires_grad = False
        for param in self.model.foreground_rendering_network_list.parameters():
            param.requires_grad = False
        for param in self.model.bg_implicit_network.parameters():
            param.requires_grad = False
        for param in self.model.bg_rendering_network.parameters():
            param.requires_grad = False
    

    def unfreeze_shape_model(self):
        for param in self.model.foreground_implicit_network_list.parameters():
            param.requires_grad = True
        for param in self.model.foreground_rendering_network_list.parameters():
            param.requires_grad = True
        for param in self.model.bg_implicit_network.parameters():
            param.requires_grad = True
        for param in self.model.bg_rendering_network.parameters():
            param.requires_grad = True

    
    def training_step(self, batch):
        inputs, targets = batch
        batch_idx = inputs["idx"]
        opt_joint, opt_pose = self.optimizers()
        sch_joint, sch_pose = self.lr_schedulers()

        pose_epoch = self.current_epoch % self.pose_opt_interval
        is_pose_depth_opt = 'sam_mask' in inputs.keys() and self.current_epoch >= self.pose_start_epoch and pose_epoch < self.pose_opt_epoch and self.current_epoch < self.pose_end_epoch and not self.depth_end
        is_delayed_pose_opt = False
        cur_opt = None
        opt_idx = 0
        if self.using_sam:
            is_certain = inputs["is_certain"].squeeze()
            is_delayed_pose_opt = self.current_epoch < self.pose_correction_epoch and not is_certain
            # optimze pose with inter-person pose loss within the loop
            if is_pose_depth_opt:
                cur_opt = opt_pose
                opt_idx = 1
                self.toggle_optimizer(opt_pose, opt_idx)
            # optimize pose for frames with unreliable pose
            elif is_delayed_pose_opt:
                cur_opt = opt_joint
                opt_idx = 0
                self.toggle_optimizer(opt_joint, opt_idx)
                self.freeze_shape_model()
            # jointly optimize pose and shape
            else:
                cur_opt = opt_joint
                opt_idx = 0
                self.toggle_optimizer(opt_joint, opt_idx)

        device = inputs["smpl_params"].device

        if self.opt_smpl:
            body_params_list = [self.body_model_list[i](batch_idx) for i in range(self.num_person)]
            inputs['smpl_trans'] = torch.stack([body_model_params['transl'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_shape'] = torch.stack([body_model_params['betas'] for body_model_params in body_params_list], dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list], dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            if batch_idx == 0:
                last_idx = batch_idx
            else:
                last_idx = batch_idx - 1
            body_params_list_last = [self.body_model_list[i](last_idx) for i in range(self.num_person)]
            global_orient_last = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list_last],
                                        dim=1)
            body_pose_last = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list_last], dim=1)
            inputs['smpl_pose_last'] = torch.cat((global_orient_last, body_pose_last), dim=2)
        else:
            inputs['smpl_pose'] = inputs["smpl_params"][..., 4:76]
            inputs['smpl_shape'] = inputs["smpl_params"][..., 76:]
            inputs['smpl_trans'] = inputs["smpl_params"][..., 1:4]

        inputs['current_epoch'] = self.current_epoch
        if (is_delayed_pose_opt or self.all_edge) and 'edge_uv' in inputs.keys():
            inputs['uv'] = inputs['edge_uv']
            targets['rgb'] = targets['edge_rgb']
            if 'edge_sam_mask' in inputs.keys():
                inputs['sam_mask'] = inputs['edge_sam_mask']
        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets)
        if is_pose_depth_opt:
            depth_order_loss_pyrender, loss_instance_silhouette, loss_interpenetration = self.get_depth_order_loss(inputs)
            loss_output.update({'loss_interpenetration': loss_interpenetration})
            loss_output.update({'depth_order_loss_pyrender': depth_order_loss_pyrender})
            loss_output.update({'loss_instance_silhouette': loss_instance_silhouette})
            loss_output["loss"] += loss_interpenetration
            loss_output["loss"] += depth_order_loss_pyrender
            loss_output["loss"] += loss_instance_silhouette
        else:
            loss_output.update({'depth_order_loss_pyrender': torch.zeros((1), device=device)})
            loss_output.update({'loss_instance_silhouette': torch.zeros((1), device=device)})
            loss_output.update({'loss_interpenetration': torch.zeros((1), device=device)})
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        if loss_output["loss"].isnan():
            print("Nan: overall loss")
            loss_output["loss"] = torch.zeros((1),device=loss_output["loss"].device, requires_grad=True)
        cur_opt.zero_grad()
        self.manual_backward(loss_output["loss"])
        cur_opt.step()

        sch_every_N_epoch = 1
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % sch_every_N_epoch == 0:
            sch_joint.step()
            sch_pose.step()

        self.untoggle_optimizer(opt_idx)
        self.unfreeze_shape_model()

        return loss_output["loss"]


    def opt_depth(self):
        # do not change the test setting to novel pose setting.
        testset = create_dataset(self.opt.dataset.test)
        for batch_ndx, batch in enumerate(tqdm(testset)):
            # generate instance mask for all test images
            inputs, targets, pixel_per_batch, total_pixels, idx = batch
            print("idx: ", inputs["idx"])
            inputs = {key: value.cuda() for key, value in inputs.items()}
            results = []
            batch_inputs = {"uv": inputs["uv"],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:, :, 4:76],
                            "smpl_shape": inputs["smpl_params"][:, :, 76:],
                            "smpl_trans": inputs["smpl_params"][:, :, 1:4],
                            "img_size": targets["img_size"],
                            "org_sam_mask": inputs["org_sam_mask"],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if self.depth_pose:
                opt_params = [{'params': self.body_model_list.parameters(), 'lr': self.opt.model.learning_rate}]
                optimizer_transl = optim.Adam(opt_params, lr=self.opt.model.learning_rate, eps=1e-8)
            else:
                opt_params = []
                for body_model_i in self.body_model_list:
                    opt_params.append({'params': body_model_i.transl.weight, 'lr': self.opt.model.learning_rate}) # here do not multiple 0.1
                optimizer_transl = optim.Adam(opt_params, lr=self.opt.model.learning_rate, eps=1e-8)


            body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
            batch_inputs['smpl_trans'] = torch.stack(
                [body_model_params['transl'] for body_model_params in body_params_list],
                dim=1)
            batch_inputs['smpl_shape'] = torch.stack(
                [body_model_params['betas'] for body_model_params in body_params_list],
                dim=1)
            global_orient = torch.stack(
                [body_model_params['global_orient'] for body_model_params in body_params_list],
                dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                    dim=1)
            batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            inputs = batch_inputs
            renderer = self.get_renderer(inputs)
            idx = inputs["idx"].item()
            smpl_params = inputs['smpl_params']
            smpl_pose = inputs["smpl_pose"]
            scale = smpl_params[:, :, 0]
            smpl_shape = inputs["smpl_shape"]
            smpl_trans = inputs["smpl_trans"]
            zbuf_list = []
            deformed_faces_list = []
            deformed_verts_color_list = []
            color_dict = [[255, 0.0, 0.0], [0.0, 255, 0.0], [0.0, 0.0, 255], [125, 125, 0.0], [0.0, 125, 125],
                          [125, 0.0, 125], [64, 0.0, 0.0], [0.0, 64, 0.0], [0.0, 0.0, 64], [32, 32, 0.0],
                          [0.0, 32, 32], [32, 0.0, 32]]
            vertex_color_list = []
            used_color_list = []
            canonical_vertex_list = []
            for person_idx, smpl_server in enumerate(self.model.smpl_server_list):
                smpl_output = smpl_server(scale[:, person_idx], smpl_trans[:, person_idx], smpl_pose[:, person_idx],
                                          smpl_shape[:, person_idx])
                smpl_tfs = smpl_output['smpl_tfs']
                cond_pose = smpl_pose[:, person_idx, 3:] / np.pi
                if self.depth_cond_zero:
                    cond_pose = smpl_pose[:, person_idx, 3:] * 0.
                if self.model.use_person_encoder:
                    person_id_tensor = torch.from_numpy(np.array([person_idx])).long().to(smpl_pose.device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}

                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_idx),
                                               smpl_server.verts_c[0], point_batch=10000, res_up=2)
                canonical_vertex = torch.tensor(mesh_canonical.vertices[None],
                                                device=self.model.mesh_v_cano_list[person_idx].device).float()
                canonical_vertex_list.append(canonical_vertex)
                canonical_face = torch.tensor(mesh_canonical.faces.astype(np.int64),
                                              device=self.model.mesh_v_cano_list[person_idx].device)
                deformed_verts = self.get_deformed_mesh_fast_mode_multiple_person_torch(canonical_vertex,
                                                                                        smpl_output['smpl_tfs'],
                                                                                        person_idx)
                deformed_verts = (1 / scale[:, person_idx].squeeze()) * deformed_verts
                deformed_faces_list.append(canonical_face.unsqueeze(0))

                verts_color = torch.tensor(color_dict[person_idx], device=deformed_verts.device).repeat(
                    deformed_verts.shape[1], 1).unsqueeze(0)
                verts_color = verts_color[..., :3] / 255.0
                vertex_color_list.append(verts_color)
                used_color_list.append(color_dict[person_idx])

            it_per_loop = self.it_per_loop
            loop = tqdm(range(it_per_loop))
            for it in loop:
                # start sampling point
                data = { "rgb": targets["org_img"].squeeze(0), "uv": targets["org_uv"].squeeze(0), "object_mask": targets["org_object_mask"].squeeze(0), "sam_mask": targets["org_sam_mask"].squeeze(0),}
                number_sample = 512
                img_size = [int(targets["img_size"][0].cpu().numpy()), int(targets["img_size"][1].cpu().numpy())]
                samples, index_outside = weighted_sampling(data, img_size, number_sample)

                optimizer_transl.zero_grad()

                body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                smpl_trans = torch.stack(
                    [body_model_params['transl'] for body_model_params in body_params_list],
                    dim=1)
                smpl_shape = torch.stack(
                    [body_model_params['betas'] for body_model_params in body_params_list],
                    dim=1)
                global_orient = torch.stack(
                    [body_model_params['global_orient'] for body_model_params in body_params_list],
                    dim=1)
                body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                        dim=1)
                smpl_pose = torch.cat((global_orient, body_pose), dim=2)

                idx = inputs["idx"].item()
                smpl_params = inputs['smpl_params']
                scale = smpl_params[:, :, 0]

                model_input = {
                    'smpl_trans': smpl_trans,
                    'smpl_shape': smpl_shape,
                    'smpl_pose': smpl_pose,
                    'smpl_pose_last': smpl_pose, # here I disable the temporal loss
                    "uv": torch.from_numpy(samples["uv"].astype(np.float32)[None]),
                    "idx": inputs['idx'],
                    'index_outside': torch.from_numpy(index_outside[None]),
                    "sam_mask": torch.from_numpy(samples['sam_mask'][None]),
                    "P": inputs["P"],
                    "C": inputs["C"],
                    "intrinsics": inputs['intrinsics'],
                    "pose": inputs['pose'],
                    "smpl_params": inputs['smpl_params'],
                    # "img_size": torch.from_numpy(np.array(inputs["img_size"])),

                }
                model_targets = {"rgb": torch.from_numpy(samples["rgb"].astype(np.float32)[None])}

                model_input = {key: value.cuda() for key, value in model_input.items()}
                model_targets = {key: value.cuda() for key, value in model_targets.items()}
                model_input['current_epoch'] = self.current_epoch

                if self.depth_cond_zero:
                    model_outputs = self.model(model_input, cond_zero_shit=True)
                else:
                    model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, model_targets)

                deformed_verts_list = []
                for person_idx, smpl_server in enumerate(self.model.smpl_server_list):
                    smpl_output = smpl_server(scale[:, person_idx], smpl_trans[:, person_idx], smpl_pose[:, person_idx],
                                              smpl_shape[:, person_idx])
                    deformed_verts = self.get_deformed_mesh_fast_mode_multiple_person_torch(canonical_vertex_list[person_idx].detach().clone(),
                                                                                            smpl_output['smpl_tfs'],
                                                                                            person_idx)
                    deformed_verts = (1 / scale[:, person_idx].squeeze()) * deformed_verts
                    deformed_verts_list.append(deformed_verts)

                renderer_depth_map = renderer.render_multiple_depth_map(deformed_verts_list, deformed_faces_list)
                interpenetration_loss = self.get_interpenetration_loss(deformed_verts_list, deformed_faces_list)

                reshape_depth_map_list = []
                for map_id, depth_map_i in enumerate(renderer_depth_map):
                    depth_map_i = depth_map_i[0, :, :, 0]
                    reshape_depth_map_list.append(depth_map_i)
                # get front depth map
                max_depth_map_list = []
                max_depth = 999
                for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                    # depth_map_processed = np.copy(depth_map_i)
                    depth_map_processed = depth_map_i.clone()
                    no_interaction = depth_map_processed < 0
                    depth_map_processed[no_interaction] = max_depth
                    max_depth_map_list.append(depth_map_processed)
                max_depth_map = torch.stack(max_depth_map_list, dim=-1)
                front_depth_map, _ = torch.min(max_depth_map, dim=-1)
                valid_mask = front_depth_map < max_depth
                sam_mask = inputs["org_sam_mask"]

                sam_mask = self.sigmoid(sam_mask)
                sam_mask = sam_mask.squeeze(0)

                valid_mask = torch.logical_and(valid_mask, sam_mask.sum(dim=-1) <= (1 + 1e-2))
                valid_mask = torch.logical_and(valid_mask, sam_mask.sum(dim=-1) >= (0.7))
                # (H, W, P) -> (H, W, 1)
                sam_mask_idx = torch.argmax(sam_mask, dim=-1)
                # sam_mask_idx = sam_mask_idx[valid_mask]
                # (H, W) -> (H, W, 1)
                sam_mask_idx = sam_mask_idx.unsqueeze(-1)
                # import pdb; pdb.set_trace()
                gt_depth_map = torch.gather(max_depth_map, dim=-1, index=sam_mask_idx)
                gt_depth_map = gt_depth_map.squeeze(-1)
                valid_mask = torch.logical_and(valid_mask, gt_depth_map < max_depth)
                if it == 0 or it == it_per_loop - 1:
                    front_numpy = front_depth_map.detach().cpu().numpy()
                    gt_numpy = gt_depth_map.detach().cpu().numpy()
                    vis_min_depth = 2.5
                    vis_max_depth = 5
                    front_numpy = np.clip(front_numpy, vis_min_depth, vis_max_depth)
                    gt_numpy = np.clip(gt_numpy, vis_min_depth, vis_max_depth)
                    # do the min max normalization manually
                    front_numpy = (front_numpy - vis_min_depth) / (vis_max_depth - vis_min_depth)
                    gt_numpy = (gt_numpy - vis_min_depth) / (vis_max_depth - vis_min_depth)
                    front_numpy = (front_numpy * 255).astype(np.uint8)
                    gt_numpy = (gt_numpy * 255).astype(np.uint8)

                    # Apply the reversed 'JET' colormap
                    front_numpy = cv2.applyColorMap(255 - front_numpy, cv2.COLORMAP_JET)
                    gt_numpy = cv2.applyColorMap(255 - gt_numpy, cv2.COLORMAP_JET)
                    os.makedirs(f"stage_depth_map_pyrender/{self.current_epoch:05d}/{it:05d}/front", exist_ok=True)
                    os.makedirs(f"stage_depth_map_pyrender/{self.current_epoch:05d}/{it:05d}/gt", exist_ok=True)
                    cv2.imwrite(os.path.join(f"stage_depth_map_pyrender/{self.current_epoch:05d}/{it:05d}/front",
                                             f'front_%04d.png' % idx), front_numpy)
                    cv2.imwrite(os.path.join(f"stage_depth_map_pyrender/{self.current_epoch:05d}/{it:05d}/gt",
                                             f'gt_%04d.png' % idx), gt_numpy)
                valid_mask = valid_mask.flatten()
                gt_depth_map = gt_depth_map.flatten()[valid_mask]
                front_depth_map = front_depth_map.flatten()[valid_mask]
                exclude_mask = ~(gt_depth_map == front_depth_map)
                depth_loss_milestone = 1000
                interpenetration_loss_weight = self.opt.model.loss.get('interpenetration_loss_weight', 0.0)
                interpenetration_loss = interpenetration_loss_weight * (1 - (min(depth_loss_milestone,
                                                                                 self.current_epoch) / depth_loss_milestone)) * interpenetration_loss
                loss_dict = dict()
                if exclude_mask.sum() == 0:
                    total_loss = interpenetration_loss + loss_output["loss"]
                    loss_dict['interpenetration_loss'] = interpenetration_loss
                    loss_dict['depth_order_loss'] = torch.zeros(1, device=interpenetration_loss.device, requires_grad=True)
                    loss_dict["render_loss"] = loss_output["loss"]
                else:
                    # import pdb; pdb.set_trace()
                    loss = torch.log(1 + torch.exp(gt_depth_map[exclude_mask] - front_depth_map[exclude_mask])).sum()
                    depth_order_weight = self.opt.model.loss.get('depth_order_weight', 0.005)
                    loss = depth_order_weight * (
                                1 - (min(depth_loss_milestone, self.current_epoch) / depth_loss_milestone)) * loss
                    total_loss = interpenetration_loss + loss + loss_output["loss"]
                    loss_dict['interpenetration_loss'] = interpenetration_loss
                    loss_dict['depth_order_loss'] = loss
                    loss_dict["render_loss"] = loss_output["loss"]

                total_loss.backward()
                optimizer_transl.step()

                l_str = 'Iter: %d' % it
                for k in loss_dict:
                    l_str += ', %s: %0.4f' % (k, loss_dict[k].mean().item())
                    loop.set_description(l_str)




    def training_epoch_end(self, outputs) -> None:
        # Canonical mesh update every 20 epochs
        if self.current_epoch != 0 and self.current_epoch % 20 == 0:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                cond_pose = torch.zeros(1, 69).float().cuda()
                if self.model.use_person_encoder:
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(cond_pose.device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(cond_pose.shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}
                try:
                    mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id), smpl_server.verts_c[0], point_batch=10000, res_up=2)
                    self.model.mesh_v_cano_list[person_id] = torch.tensor(mesh_canonical.vertices[None], device = self.model.mesh_v_cano_list[person_id].device).float()
                    self.model.mesh_f_cano_list[person_id] = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.mesh_v_cano_list[person_id].device)
                    self.model.mesh_face_vertices_list[person_id] = index_vertices_by_faces(self.model.mesh_v_cano_list[person_id], self.model.mesh_f_cano_list[person_id])
                except:
                    print("Canonical mesh generation failed, do not update it. mainly because the mesh (Surface level must be within volume data range).")
        if self.current_epoch % 50 == 0:
            torch.cuda.empty_cache()
            self.get_instance_mask()
            torch.cuda.empty_cache()
            self.get_sam_mask()
            torch.cuda.empty_cache()
        if (self.current_epoch in self.depth_epoch) and self.depth_end:
            torch.cuda.empty_cache()
            self.opt_depth()
            torch.cuda.empty_cache()
        return super().training_epoch_end(outputs)

    def get_interpenetration_loss(self, vertex_list, face_list):
        num_pixels = 5120
        interpenetration_loss = torch.zeros(1, device=vertex_list[0].device)
        # vertex_list shape (number point, 3)
        for person_id, vertex in enumerate(vertex_list):
            idx = torch.randperm(vertex.shape[1])[:num_pixels].cuda()
            sample_point = torch.index_select(vertex, dim=1, index=idx)
            # import pdb;pdb.set_trace()
            for partner_id, partner_vertex in enumerate(vertex_list):
                if partner_id == person_id:
                    continue
                sign = kaolin.ops.mesh.check_sign(vertex_list[partner_id], face_list[partner_id].squeeze(0),
                                                  sample_point).float()
                sign = sign.squeeze(0)
                # sign == -1 inside, sign > 1 outside
                sign = 1 - 2 * sign
                if (sign < 0).any():
                    penetrate_point = sample_point[:, sign < 0]
                    distance_batch, index_batch, neighbor_points = ops.knn_points(penetrate_point,
                                                                                  vertex_list[partner_id], K=1,
                                                                                  return_nn=True)
                    # (N, P1, K, D) for neighbor_points
                    neighbor_points = neighbor_points.mean(dim=2)
                    # filter outlier point
                    stable_point = (penetrate_point - neighbor_points).norm(dim=-1).reshape(-1) < 0.1
                    if stable_point.any():
                        penetrate_point = penetrate_point.reshape(-1, 3)[stable_point]
                        neighbor_points = neighbor_points.reshape(-1, 3)[stable_point]
                        interpenetration_loss = interpenetration_loss + torch.nn.functional.mse_loss(penetrate_point,
                                                                                                     neighbor_points, reduction='sum')
        return interpenetration_loss

    def get_renderer(self, inputs):
        img_size = inputs["img_size"]
        P = inputs["P"][0].cpu().numpy()
        P_norm = np.eye(4)
        P_norm[:, :] = P[:, :]
        assert inputs["smpl_params"][:, 0, 0] == inputs["smpl_params"][:, 1, 0]
        scale = inputs["smpl_params"][:, 0, 0]
        scale_eye = np.eye(4)
        scale_eye[0, 0] = scale
        scale_eye[1, 1] = scale
        scale_eye[2, 2] = scale
        P_norm = P_norm @ scale_eye
        out = cv2.decomposeProjectionMatrix(P_norm[:3, :])
        cam_intrinsics = out[0]
        render_R = out[1]
        cam_center = out[2]
        cam_center = (cam_center[:3] / cam_center[3])[:, 0]
        render_T = -render_R @ cam_center
        render_R = torch.tensor(render_R)[None].float()
        render_T = torch.tensor(render_T)[None].float()
        renderer = Renderer(img_size=[img_size[0].cpu().numpy(), img_size[1].cpu().numpy()],
                            cam_intrinsic=cam_intrinsics)
        renderer.set_camera(render_R, render_T)
        return renderer

    def get_depth_order_loss(self, inputs):
        # print("start get depth order loss")
        renderer = self.get_renderer(inputs)
        idx = inputs["idx"].item()
        img_size = inputs["img_size"]
        input_img = inputs["org_img"][0]
        smpl_params = inputs['smpl_params']
        smpl_pose = inputs["smpl_pose"]
        scale = smpl_params[:, :, 0]
        smpl_shape = inputs["smpl_shape"]
        smpl_trans = inputs["smpl_trans"]
        zbuf_list = []
        deformed_verts_list = []
        deformed_faces_list = []
        deformed_verts_color_list = []
        # number_persons = len(self.model.smpl_server_list)
        # color_dict = [[255, 0.0, 0.0] for _ in range(number_persons)]
        color_dict = [[255,0.0, 0.0], [0.0, 255, 0.0], [0.0, 0.0, 255], [125, 125, 0.0], [0.0, 125, 125], [125, 0.0, 125],[64, 0.0, 0.0], [0.0, 64, 0.0], [0.0, 0.0, 64], [32, 32, 0.0], [0.0, 32, 32], [32, 0.0, 32]]
        vertex_color_list = []
        used_color_list = []
        for person_idx, smpl_server in enumerate(self.model.smpl_server_list):
            smpl_output = smpl_server(scale[:, person_idx], smpl_trans[:, person_idx], smpl_pose[:, person_idx], smpl_shape[:, person_idx])
            # use deformed mesh as input, instead of SMPL
            # smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
            smpl_tfs = smpl_output['smpl_tfs']
            # cond = {'smpl': smpl_pose[:, 3:] / np.pi}
            cond_pose = smpl_pose[:, person_idx, 3:] / np.pi
            if self.model.use_person_encoder:
                # import pdb;pdb.set_trace()
                person_id_tensor = torch.from_numpy(np.array([person_idx])).long().to(smpl_pose.device)
                person_encoding = self.model.person_latent_encoder(person_id_tensor)
                person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                cond = {'smpl_id': cond_pose_id}
            else:
                cond = {'smpl': cond_pose}

            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_idx), smpl_server.verts_c[0], point_batch=10000, res_up=2)
            canonical_vertex = torch.tensor(mesh_canonical.vertices[None], device=self.model.mesh_v_cano_list[person_idx].device).float()
            canonical_face = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.mesh_v_cano_list[person_idx].device)
            deformed_verts = self.get_deformed_mesh_fast_mode_multiple_person_torch(canonical_vertex, smpl_output['smpl_tfs'], person_idx)
            # rescale it to the original scale
            deformed_verts = (1 / scale[:, person_idx].squeeze()) * deformed_verts
            deformed_verts_list.append(deformed_verts)
            deformed_faces_list.append(canonical_face.unsqueeze(0))

            verts_color = torch.tensor(color_dict[person_idx], device=deformed_verts.device).repeat(deformed_verts.shape[1], 1).unsqueeze(0)
            verts_color = verts_color[..., :3] / 255.0
            vertex_color_list.append(verts_color)
            used_color_list.append(color_dict[person_idx])


        # add background color
        used_color_list.append([0.0,0.0,0.0])
        # (P+1, 3)
        used_color_list = torch.tensor(used_color_list, device=deformed_verts.device)
        renderer_depth_map = renderer.render_multiple_depth_map(deformed_verts_list, deformed_faces_list)
        interpenetration_loss = self.get_interpenetration_loss(deformed_verts_list, deformed_faces_list)
        renderer_instance_map = renderer.softrender_multiple_meshes(deformed_verts_list, deformed_faces_list, vertex_color_list)
        renderer_instance_map = (255 * renderer_instance_map)
        renderer_instance_map = renderer_instance_map[0]
        reshape_depth_map_list = []
        for map_id, depth_map_i in enumerate(renderer_depth_map):
            depth_map_i = depth_map_i[0, :, :, 0]
            reshape_depth_map_list.append(depth_map_i)
        # get front depth map
        max_depth_map_list = []
        max_depth = 999
        for map_id, depth_map_i in enumerate(reshape_depth_map_list):
            depth_map_processed = depth_map_i.clone()
            no_interaction = depth_map_processed < 0
            depth_map_processed[no_interaction] = max_depth
            max_depth_map_list.append(depth_map_processed)
        max_depth_map = torch.stack(max_depth_map_list, dim=-1)
        front_depth_map, _ = torch.min(max_depth_map, dim=-1)
        valid_mask = front_depth_map < max_depth
        sam_mask = inputs["org_sam_mask"]

        sam_mask = self.sigmoid(sam_mask)
        sam_mask = sam_mask.squeeze(0)

        # determine background probability
        sam_background_mask = (1-sam_mask.sum(dim=-1, keepdims=True))
        sam_foreground_background = torch.cat([sam_mask, sam_background_mask], dim=-1)
        sam_foreground_background_idx = torch.argmax(sam_foreground_background, dim=-1)
        H = sam_foreground_background_idx.shape[0]
        W = sam_foreground_background_idx.shape[1]
        sam_foreground_background_idx = sam_foreground_background_idx.reshape(-1)
        gt_instance_map = used_color_list[sam_foreground_background_idx].reshape(H, W, 3)
        # visualize renderer_instance_map and gt_instance_map
        renderer_instance_map_RGB = renderer_instance_map[..., :3] * (renderer_instance_map[..., [3]] / 255.0)
        if self.current_epoch % 50 == 0:
            renderer_instance_numpy = renderer_instance_map_RGB.detach().cpu().numpy().astype(np.uint8)
            gt_instance_numpy = gt_instance_map.detach().cpu().numpy().astype(np.uint8)
            os.makedirs(f"stage_instance_mask_pyrender/{self.current_epoch:05d}/project", exist_ok=True)
            os.makedirs(f"stage_instance_mask_pyrender/{self.current_epoch:05d}/gt", exist_ok=True)
            cv2.imwrite(os.path.join(f"stage_instance_mask_pyrender/{self.current_epoch:05d}/project",
                                     f'%04d.png' % idx), renderer_instance_numpy[..., :3])
            cv2.imwrite(os.path.join(f"stage_instance_mask_pyrender/{self.current_epoch:05d}/gt",
                                     f'%04d.png' % idx), gt_instance_numpy[...,:3])
        # (H, W)
        # should not have more than one label in one pixel
        valid_mask = torch.logical_and(valid_mask, sam_mask.sum(dim=-1) <= (1 + 1e-2))
        valid_mask = torch.logical_and(valid_mask, sam_mask.sum(dim=-1) >= (0.7))
        # (H, W, P) -> (H, W, 1)
        sam_mask_idx = torch.argmax(sam_mask, dim=-1)
        # sam_mask_idx = sam_mask_idx[valid_mask]
        # (H, W) -> (H, W, 1)
        sam_mask_idx = sam_mask_idx.unsqueeze(-1)
        # import pdb; pdb.set_trace()
        gt_depth_map = torch.gather(max_depth_map, dim=-1, index=sam_mask_idx)
        gt_depth_map = gt_depth_map.squeeze(-1)
        valid_mask = torch.logical_and(valid_mask, gt_depth_map < max_depth)
        if self.current_epoch % 50 == 0:
            front_numpy = front_depth_map.detach().cpu().numpy()
            gt_numpy = gt_depth_map.detach().cpu().numpy()
            vis_min_depth = 2.5
            vis_max_depth = 5
            front_numpy = np.clip(front_numpy, vis_min_depth, vis_max_depth)
            gt_numpy = np.clip(gt_numpy, vis_min_depth, vis_max_depth)
            # do the min max normalization manually
            front_numpy = (front_numpy - vis_min_depth) / (vis_max_depth - vis_min_depth)
            gt_numpy = (gt_numpy - vis_min_depth) / (vis_max_depth - vis_min_depth)
            front_numpy = (front_numpy * 255).astype(np.uint8)
            gt_numpy = (gt_numpy * 255).astype(np.uint8)
            # Apply the reversed 'JET' colormap
            front_numpy = cv2.applyColorMap(255 - front_numpy, cv2.COLORMAP_JET)
            gt_numpy = cv2.applyColorMap(255 - gt_numpy, cv2.COLORMAP_JET)
            os.makedirs(f"stage_depth_map_pyrender/{self.current_epoch:05d}/front", exist_ok=True)
            os.makedirs(f"stage_depth_map_pyrender/{self.current_epoch:05d}/gt", exist_ok=True)
            cv2.imwrite(os.path.join(f"stage_depth_map_pyrender/{self.current_epoch:05d}/front",
                                     f'front_%04d.png' % idx), front_numpy)
            cv2.imwrite(os.path.join(f"stage_depth_map_pyrender/{self.current_epoch:05d}/gt",
                                     f'gt_%04d.png' % idx), gt_numpy)
        valid_mask = valid_mask.flatten()
        gt_depth_map = gt_depth_map.flatten()[valid_mask]
        front_depth_map = front_depth_map.flatten()[valid_mask]
        exclude_mask = ~(gt_depth_map == front_depth_map)
        depth_loss_milestone = 1000
        try:
            silhouette_weight = self.opt.model.loss.silhouette_weight
        except:
            silhouette_weight = 0.000
        loss_instance_silhouette = silhouette_weight * self.l2_loss(gt_instance_map, renderer_instance_map_RGB) * (1 - (min(depth_loss_milestone, self.current_epoch) / depth_loss_milestone))
        try:
            interpenetration_loss_weight = self.opt.model.loss.interpenetration_loss_weight
        except:
            interpenetration_loss_weight = 0.000
        interpenetration_loss = interpenetration_loss_weight * (1 - (min(depth_loss_milestone, self.current_epoch) / depth_loss_milestone)) * interpenetration_loss
        if exclude_mask.sum() == 0:
            return torch.tensor(0.0).cuda(), loss_instance_silhouette, interpenetration_loss
        # import pdb; pdb.set_trace()
        loss = torch.log(1 + torch.exp(gt_depth_map[exclude_mask] - front_depth_map[exclude_mask])).sum()
        try:
            depth_order_weight = self.opt.model.loss.depth_order_weight
        except:
            depth_order_weight = 0.005
        loss = depth_order_weight * (1 - (min(depth_loss_milestone, self.current_epoch) / depth_loss_milestone)) * loss
        return loss, loss_instance_silhouette, interpenetration_loss
        # gt_depth_map = max_depth_map[sam_mask_idx]
    def get_sam_mask(self):
        print("start get refined SAM mask")
        self.sam_server.get_sam_mask(self.current_epoch)
    def get_instance_mask(self):
        print("start get SMPL instance mask")
        self.model.eval()
        os.makedirs(f"stage_mask/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_rendering/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_fg_rendering/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_normal/{self.current_epoch:05d}/all", exist_ok=True)
        testset = create_dataset(self.opt.dataset.test)
        keypoint_list = [[] for _ in range(len(self.model.smpl_server_list))]
        all_person_smpl_mask_list =[]
        all_instance_mask_depth_list = []
        for batch_ndx, batch in enumerate(tqdm(testset)):
            # generate instance mask for all test images
            inputs, targets, pixel_per_batch, total_pixels, idx = batch
            inputs = {key: value.cuda() for key, value in inputs.items()}
            num_splits = (total_pixels + pixel_per_batch -
                          1) // pixel_per_batch
            results = []
            for i in range(num_splits):
                indices = list(range(i * pixel_per_batch,
                                     min((i + 1) * pixel_per_batch, total_pixels)))
                batch_inputs = {"uv": inputs["uv"][:, indices],
                                "P": inputs["P"],
                                "C": inputs["C"],
                                "intrinsics": inputs['intrinsics'],
                                "pose": inputs['pose'],
                                "smpl_params": inputs["smpl_params"],
                                "smpl_pose": inputs["smpl_params"][:, :, 4:76],
                                "smpl_shape": inputs["smpl_params"][:, :, 76:],
                                "smpl_trans": inputs["smpl_params"][:, :, 1:4],
                                "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

                if self.opt_smpl:
                    body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                    batch_inputs['smpl_trans'] = torch.stack(
                        [body_model_params['transl'] for body_model_params in body_params_list],
                        dim=1)
                    batch_inputs['smpl_shape'] = torch.stack(
                        [body_model_params['betas'] for body_model_params in body_params_list],
                        dim=1)
                    global_orient = torch.stack(
                        [body_model_params['global_orient'] for body_model_params in body_params_list],
                        dim=1)
                    body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                            dim=1)
                    batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

                batch_targets = {
                    "rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                    "img_size": targets["img_size"]}
                results.append({**batch_targets})

            img_size = targets["img_size"]
            P = batch_inputs["P"][0].cpu().numpy()
            P_norm = np.eye(4)
            P_norm[:, :] = P[:, :]
            assert batch_inputs["smpl_params"][:, 0, 0] == batch_inputs["smpl_params"][:, 1, 0]
            scale = batch_inputs["smpl_params"][:, 0, 0]
            scale_eye = np.eye(4)
            scale_eye[0, 0] = scale
            scale_eye[1, 1] = scale
            scale_eye[2, 2] = scale
            P_norm = P_norm @ scale_eye
            out = cv2.decomposeProjectionMatrix(P_norm[:3, :])
            cam_intrinsics = out[0]
            render_R = out[1]
            cam_center = out[2]
            cam_center = (cam_center[:3] / cam_center[3])[:, 0]
            render_T = -render_R @ cam_center
            render_R = torch.tensor(render_R)[None].float()
            render_T = torch.tensor(render_T)[None].float()
            renderer = Renderer(img_size=[img_size[0], img_size[1]],
                                cam_intrinsic=cam_intrinsics)

            img_size = targets["img_size"]
            renderer.set_camera(render_R, render_T)
            verts_list = []
            faces_list = []
            colors_list = []
            color_dict = [[255, 0.0, 0.0] for _ in range(self.num_person)]
            for person_idx, smpl_server in enumerate(self.model.smpl_server_list):
                smpl_outputs = smpl_server(batch_inputs["smpl_params"][:, person_idx, 0], batch_inputs['smpl_trans'][:, person_idx], batch_inputs['smpl_pose'][:, person_idx], batch_inputs['smpl_shape'][:, person_idx])
                verts_color = torch.tensor(color_dict[person_idx]).repeat(smpl_outputs["smpl_verts"].shape[1], 1)
                # use smpl mesh as mask prompt for the beginning
                if self.current_epoch <= 190:
                    smpl_mesh = trimesh.Trimesh((1/scale.squeeze().detach().cpu()) * smpl_outputs["smpl_verts"].squeeze().detach().cpu(), smpl_server.smpl.faces, process=False, vertex_colors=verts_color.cpu())
                    verts = torch.tensor(smpl_mesh.vertices).cuda().float()[None]
                    faces = torch.tensor(smpl_mesh.faces).cuda()[None]
                    colors = torch.tensor(smpl_mesh.visual.vertex_colors).float().cuda()[None,..., :3] / 255
                else:
                    # use deformed mesh as input, instead of SMPL
                    smpl_tfs = smpl_outputs['smpl_tfs']
                    cond_pose = batch_inputs["smpl_pose"][:, person_idx, 3:] / np.pi
                    if self.model.use_person_encoder:
                        person_id_tensor = torch.from_numpy(np.array([person_idx])).long().to(batch_inputs["smpl_pose"].device)
                        person_encoding = self.model.person_latent_encoder(person_id_tensor)
                        person_encoding = person_encoding.repeat(batch_inputs["smpl_pose"].shape[0], 1)
                        cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                        cond = {'smpl_id': cond_pose_id}
                    else:
                        cond = {'smpl': cond_pose}

                    mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_idx), smpl_server.verts_c[0],
                                                   point_batch=10000, res_up=2)
                    verts_deformed = self.get_deformed_mesh_fast_mode_multiple_person(mesh_canonical.vertices, smpl_tfs, person_idx)
                    verts_deformed = (1/scale.squeeze().detach().cpu()) * verts_deformed
                    mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
                    verts = torch.tensor(mesh_deformed.vertices).cuda().float()[None]
                    faces = torch.tensor(mesh_deformed.faces).cuda()[None]
                    colors = torch.tensor(mesh_deformed.visual.vertex_colors).float().cuda()[None, ..., :3] / 255



                verts_list.append(verts)
                faces_list.append(faces)
                colors_list.append(colors)
                P = batch_inputs["P"][0].cpu().numpy()
                smpl_joints = smpl_outputs["smpl_all_jnts"].detach().cpu().numpy().squeeze()
                smpl_joints = smpl_joints[:27]  # original smpl point + nose + eyes
                pix_list = []
                # get the ground truth image
                img_size = results[0]["img_size"]
                rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
                input_img = rgb_gt.reshape(*img_size, -1).detach().cpu().numpy()
                input_img = (input_img * 255).astype(np.uint8)

                for j in range(0, smpl_joints.shape[0]):
                    padded_v = np.pad(smpl_joints[j], (0, 1), 'constant', constant_values=(0, 1))
                    temp = P @ padded_v.T  # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
                    pix = (temp / temp[2])[:2]
                    output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
                    pix_list.append(pix.astype(np.int32))
                pix_tensor = np.stack(pix_list, axis=0)
                keypoint_list[person_idx].append(pix_tensor)
            renderer_depth_map = renderer.render_multiple_depth_map(verts_list, faces_list, colors_list)
            img_size = results[0]["img_size"]
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            input_img = rgb_gt.reshape(*img_size, -1).detach().cpu().numpy()
            input_img = (input_img * 255).astype(np.uint8)
            reshape_depth_map_list = []
            for map_id, depth_map_i in enumerate(renderer_depth_map):
                depth_map_i = depth_map_i[0,:,:,0].data.cpu().numpy()
                reshape_depth_map_list.append(depth_map_i)
            # get front depth map
            max_depth_map_list = []
            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                depth_map_processed = np.copy(depth_map_i)
                no_interaction = depth_map_processed < 0
                max_depth = 999
                depth_map_processed[no_interaction] = max_depth
                max_depth_map_list.append(depth_map_processed)
            max_depth_map = np.stack(max_depth_map_list, axis=0)
            front_depth_map = np.min(max_depth_map, axis=0)
            # get instance mask from depth map
            instance_mask_list = []
            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                instance_mask = (depth_map_i == front_depth_map)
                instance_mask_list.append(instance_mask)
                all_red_image = np.ones((instance_mask.shape[0], instance_mask.shape[1], 3)) * np.array([255, 0, 0]).reshape(1, 1, 3)
                instance_mask = instance_mask[:, :, np.newaxis]
                output_img_person_1 = (all_red_image * instance_mask + input_img * (1 - instance_mask)).astype(np.uint8)
                os.makedirs(f"stage_depth_instance_mask/{self.current_epoch:05d}", exist_ok=True)
                cv2.imwrite(os.path.join(f"stage_depth_instance_mask/{self.current_epoch:05d}",
                                         f'{map_id}_smpl_render_%04d.png' % idx), output_img_person_1[:, :, ::-1])
            all_instance_mask_depth = np.stack(instance_mask_list, axis=0)
            all_instance_mask_depth_list.append(all_instance_mask_depth)


            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                # Processing depth map for better visualization
                depth_map_processed = np.copy(depth_map_i)

                # Assigning no interaction areas (-1s) to max value for visualization
                no_interaction = depth_map_processed < 0
                # depth_map_processed[no_interaction] = np.max(depth_map_processed[~no_interaction])
                min_depth = 2.5
                max_depth = 5
                depth_map_processed[no_interaction] = max_depth
                depth_map_processed = np.clip(depth_map_processed, min_depth, max_depth)
                # do the min max normalization manually
                depth_map_processed = (depth_map_processed - min_depth) / (max_depth - min_depth)
                depth_map_processed = (depth_map_processed * 255).astype(np.uint8)

                # Apply the reversed 'JET' colormap
                depth_map_processed = cv2.applyColorMap(255 - depth_map_processed, cv2.COLORMAP_JET)
                os.makedirs(f"stage_depth_map/{self.current_epoch:05d}", exist_ok=True)
                cv2.imwrite(os.path.join(f"stage_depth_map/{self.current_epoch:05d}",
                                         f'{map_id}_smpl_render_%04d.png' % idx), depth_map_processed)

        all_instance_mask_depth_list = np.array(all_instance_mask_depth_list)
        print("all_instance_mask_depth_list.shape ", all_instance_mask_depth_list.shape)
        keypoint_list = np.array(keypoint_list)
        keypoint_list = keypoint_list.transpose(1, 0, 2, 3)
        os.makedirs(f"stage_instance_mask/{self.current_epoch:05d}", exist_ok=True)
        np.save(f'stage_instance_mask/{self.current_epoch:05d}/all_person_smpl_mask.npy', all_instance_mask_depth_list)
        np.save(f'stage_instance_mask/{self.current_epoch:05d}/2d_keypoint.npy', keypoint_list)
        # shape (160, 2, 27, 2)
        print("keypoint_list.shape ", keypoint_list.shape)
        self.model.train()

    def query_oc(self, x, cond, person_id):
        
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.foreground_implicit_network_list[person_id](x, cond, person_id=person_id)[:,:,0].reshape(-1,1)
        return {'occ':mnfld_pred}

    def query_wc(self, x):
        
        x = x.reshape(-1, 3)
        w = self.model.deformer.query_weights(x)
    
        return w

    def query_od(self, x, cond, smpl_tfs, smpl_verts):
        
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        output = self.model.implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        
        return {'occ': sdf}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def get_deformed_mesh_fast_mode_multiple_person_torch(self, verts, smpl_tfs, person_id):
        # verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer_list[person_id].query_weights(verts[0])
        verts_deformed = skinning(verts, weights, smpl_tfs)
        # verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def get_deformed_mesh_fast_mode_multiple_person(self, verts, smpl_tfs, person_id):
        verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer_list[person_id].query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def validation_step(self, batch, *args, **kwargs):
        outputs = []
        outputs.append(self.validation_step_single_person(batch, id=-1))
        if self.num_person > 1:
            for i in range(self.num_person):
                outputs.append(self.validation_step_single_person(batch, id=i))

        return outputs


    def validation_step_single_person(self, batch, id):

        output = {}
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        self.model.eval()

        device = inputs["smpl_params"].device
        if self.opt_smpl:
            # body_model_params = self.body_model_params(batch_idx)
            body_params_list = [self.body_model_list[i](inputs['image_id']) for i in range(self.num_person)]
            inputs['smpl_trans'] = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                               dim=1)
            inputs['smpl_shape'] = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                               dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)
        else:
            inputs['smpl_pose'] = inputs["smpl_params"][..., 4:76]
            inputs['smpl_shape'] = inputs["smpl_params"][..., 76:]
            inputs['smpl_trans'] = inputs["smpl_params"][..., 1:4]

        mesh_canonical_list = []
        if id == -1:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                # cond = {'smpl': inputs["smpl_pose"][:, person_id, 3:] / np.pi}

                cond_pose = inputs["smpl_pose"][:, person_id, 3:] / np.pi
                if self.model.use_person_encoder:
                    # import pdb;pdb.set_trace()
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(inputs["smpl_pose"].device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(inputs["smpl_pose"].shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}

                try:
                    mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id),
                                                   smpl_server.verts_c[0], point_batch=10000, res_up=4)
                    mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces)
                    mesh_canonical_list.append(mesh_canonical)
                except:
                    print("mesh generation failed, mainly due to error: Surface level must be within volume data range")
                    mesh_canonical = trimesh.Trimesh()
                    mesh_canonical_list.append(mesh_canonical)

        output.update({
            'canonical_weighted': mesh_canonical_list
        })

        split = idr_utils.split_input(inputs, targets["total_pixels"][0], n_pixels=min(targets['pixel_per_batch'],
                                                                                       targets["img_size"][0] *
                                                                                       targets["img_size"][1]))

        res = []
        for s in split:
            if id==-1:
                out = self.model(s)
            else:
                out = self.model(s, id)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append({
                'rgb_values': out['rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'fg_rgb_values': out['fg_rgb_values'].detach(),
            })
        batch_size = targets['rgb'].shape[0]

        model_outputs = idr_utils.merge_output(res, targets["total_pixels"][0], batch_size)

        output.update({
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
            **targets,
        })
        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end_person(self, outputs, person_id):
        img_size = outputs[0]["img_size"]

        rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([output["fg_rgb_values"] for output in outputs], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([output["normal_values"] for output in outputs], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)
        if 'normal' in outputs[0].keys():
            normal_gt = torch.cat([output["normal"] for output in outputs], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        os.makedirs("normal", exist_ok=True)
        os.makedirs('fg_rendering', exist_ok=True)

        canonical_mesh_list = outputs[0]['canonical_weighted']
        for i, canonical_mesh in enumerate(canonical_mesh_list):
            canonical_mesh.export(f"rendering/{self.current_epoch}_{i}.ply")

        cv2.imwrite(f"rendering/{self.current_epoch}_person{person_id}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch}_person{person_id}.png", normal[:, :, ::-1])
        cv2.imwrite(f"fg_rendering/{self.current_epoch}_person{person_id}.png", fg_rgb[:, :, ::-1])

    def validation_epoch_end(self, outputs) -> None:
        # import pdb; pdb.set_trace()
        if self.num_person < 2:
            self.validation_epoch_end_person([outputs[0][0]], person_id=-1)
        else:
            self.validation_epoch_end_person([outputs[0][0]], person_id=-1)
            for i in range(self.num_person):
                self.validation_epoch_end_person([outputs[0][i+1]], person_id=i)
    
    def test_step_mesh(self, batch, id=-1):

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=2)

        if self.opt_smpl:
            body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
            smpl_trans = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                     dim=1)
            smpl_shape = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                     dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            smpl_pose = torch.cat((global_orient, body_pose), dim=2)

        if id == -1:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                # cond = {'smpl': inputs["smpl_pose"][:, person_id, 3:] / np.pi}
                smpl_outputs = smpl_server(inputs["smpl_params"][:, person_id, 0], smpl_trans[:, person_id],
                                           smpl_pose[:, person_id], smpl_shape[:, person_id])
                smpl_tfs = smpl_outputs['smpl_tfs']
                smpl_verts = smpl_outputs['smpl_verts']

                cond_pose = smpl_pose[:, person_id, 3:] / np.pi
                if self.model.use_person_encoder:
                    # import pdb;pdb.set_trace()
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id),
                                               smpl_server.verts_c[0],
                                               point_batch=10000, res_up=4)
                self.model.deformer_list[person_id].K = 7
                verts_deformed = self.get_deformed_mesh_fast_mode_multiple_person(mesh_canonical.vertices, smpl_tfs,
                                                                                  person_id)
                self.model.deformer_list[person_id].K = 1
                mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
                os.makedirs(f"test_mesh_res4/{person_id}", exist_ok=True)
                mesh_canonical.export(f"test_mesh_res4/{person_id}/{int(idx.cpu().numpy()):04d}_canonical.ply")
                mesh_deformed.export(f"test_mesh_res4/{person_id}/{int(idx.cpu().numpy()):04d}_deformed.ply")

    def test_step_each_person(self, batch, id):
        os.makedirs(f"test_mask/{id}", exist_ok=True)
        os.makedirs(f"test_rendering/{id}", exist_ok=True)
        os.makedirs(f"test_fg_rendering/{id}", exist_ok=True)
        os.makedirs(f"test_normal/{id}", exist_ok=True)
        os.makedirs(f"test_mesh/{id}", exist_ok=True)

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=2)

        if self.opt_smpl:

            body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
            smpl_trans = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                               dim=1)
            smpl_shape = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                               dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            smpl_pose = torch.cat((global_orient, body_pose), dim=2)

        if id == -1:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                smpl_outputs = smpl_server(inputs["smpl_params"][:, person_id, 0], smpl_trans[:, person_id], smpl_pose[:, person_id], smpl_shape[:, person_id])
                smpl_tfs = smpl_outputs['smpl_tfs']
                smpl_verts = smpl_outputs['smpl_verts']

                cond_pose = smpl_pose[:, person_id, 3:] / np.pi
                if self.model.use_person_encoder:
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id), smpl_server.verts_c[0],
                                               point_batch=10000, res_up=4)
                self.model.deformer_list[person_id].K = 7
                verts_deformed = self.get_deformed_mesh_fast_mode_multiple_person(mesh_canonical.vertices, smpl_tfs,
                                                                                  person_id)
                self.model.deformer_list[person_id].K = 1
                mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
                os.makedirs(f"test_mesh/{person_id}", exist_ok=True)
                mesh_canonical.export(f"test_mesh/{person_id}/{int(idx.cpu().numpy()):04d}_canonical.ply")
                mesh_deformed.export(f"test_mesh/{person_id}/{int(idx.cpu().numpy()):04d}_deformed.ply")

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                 min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:,: , 4:76],
                            "smpl_shape": inputs["smpl_params"][:,: , 76:],
                            "smpl_trans": inputs["smpl_params"][:,:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if self.opt_smpl:
                body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                batch_inputs['smpl_trans'] = torch.stack(
                    [body_model_params['transl'] for body_model_params in body_params_list],
                    dim=1)
                batch_inputs['smpl_shape'] = torch.stack(
                    [body_model_params['betas'] for body_model_params in body_params_list],
                    dim=1)
                global_orient = torch.stack(
                    [body_model_params['global_orient'] for body_model_params in body_params_list],
                    dim=1)
                body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                        dim=1)
                batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            # with torch.no_grad():
            with torch.inference_mode(mode=False):
                batch_clone = {key: value.clone() for key, value in batch_inputs.items()}
                model_outputs = self.model(batch_clone, id)
            results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if id==-1:
            instance_mask = torch.cat([result["acc_person_list"] for result in results], dim=0)
            instance_mask = instance_mask.reshape(*img_size, -1)
            for i in range(instance_mask.shape[2]):
                instance_mask_i = instance_mask[:, :, i]
                os.makedirs(f"test_instance_mask/{i}", exist_ok=True)
                cv2.imwrite(f"test_instance_mask/{i}/{int(idx.cpu().numpy()):04d}.png", instance_mask_i.cpu().numpy() * 255)


        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask/{id}/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering/{id}/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{id}/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering/{id}/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])

    def test_step_each_person_novel(self, batch, id, novel_view):
        novel_view = novel_view.item()
        os.makedirs(f"test_mask_{novel_view}/{id}", exist_ok=True)
        os.makedirs(f"test_rendering_{novel_view}/{id}", exist_ok=True)
        os.makedirs(f"test_fg_rendering_{novel_view}/{id}", exist_ok=True)
        os.makedirs(f"test_normal_{novel_view}/{id}", exist_ok=True)
        os.makedirs(f"test_mesh_{novel_view}/{id}", exist_ok=True)

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                 min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:,: , 4:76],
                            "smpl_shape": inputs["smpl_params"][:,: , 76:],
                            "smpl_trans": inputs["smpl_params"][:,:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            # with torch.no_grad():
            with torch.inference_mode(mode=False):
                batch_clone = {key: value.clone() for key, value in batch_inputs.items()}
                model_outputs = self.model(batch_clone, id)
            results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)


        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask_{novel_view}/{id}/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering_{novel_view}/{id}/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal_{novel_view}/{id}/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering_{novel_view}/{id}/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])

    def test_step_each_person_canonical(self, batch, id, prefix='canonical'):
        os.makedirs(f"test_mask_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_rendering_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_fg_rendering_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_normal_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_mesh_{prefix}/{id}", exist_ok=True)

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                 min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:,: , 4:76],
                            "smpl_shape": inputs["smpl_params"][:,: , 76:],
                            "smpl_trans": inputs["smpl_params"][:,:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if self.opt_smpl:
                body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                batch_inputs['smpl_trans'] = torch.stack(
                    [body_model_params['transl'] for body_model_params in body_params_list],
                    dim=1)
                batch_inputs['smpl_shape'] = torch.stack(
                    [body_model_params['betas'] for body_model_params in body_params_list],
                    dim=1)
                global_orient = torch.stack(
                    [body_model_params['global_orient'] for body_model_params in body_params_list],
                    dim=1)
                body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                        dim=1)
                batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                self.model.deformer_list[person_id].K = 7
                self.model.deformer_list[person_id].max_dist = 0.15

            with torch.inference_mode(mode=False):
                batch_clone = {key: value.clone() for key, value in batch_inputs.items()}
                model_outputs = self.model(batch_clone, id, canonical_pose=True)
            results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if id==-1:
            instance_mask = torch.cat([result["acc_person_list"] for result in results], dim=0)
            instance_mask = instance_mask.reshape(*img_size, -1)
            for i in range(instance_mask.shape[2]):
                instance_mask_i = instance_mask[:, :, i]
                os.makedirs(f"test_instance_mask_{prefix}/{i}", exist_ok=True)
                cv2.imwrite(f"test_instance_mask_{prefix}/{i}/{int(idx.cpu().numpy()):04d}.png", instance_mask_i.cpu().numpy() * 255)


        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])

    def test_step_each_person_denoisy(self, batch, id, prefix='denoisy'):
        os.makedirs(f"test_mask_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_rendering_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_fg_rendering_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_normal_{prefix}/{id}", exist_ok=True)
        os.makedirs(f"test_mesh_{prefix}/{id}", exist_ok=True)

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=2)

        if self.opt_smpl:
            body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
            smpl_trans = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                               dim=1)
            smpl_shape = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                               dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            smpl_pose = torch.cat((global_orient, body_pose), dim=2)

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                 min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:,: , 4:76],
                            "smpl_shape": inputs["smpl_params"][:,: , 76:],
                            "smpl_trans": inputs["smpl_params"][:,:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if self.opt_smpl:
                body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                batch_inputs['smpl_trans'] = torch.stack(
                    [body_model_params['transl'] for body_model_params in body_params_list],
                    dim=1)
                batch_inputs['smpl_shape'] = torch.stack(
                    [body_model_params['betas'] for body_model_params in body_params_list],
                    dim=1)
                global_orient = torch.stack(
                    [body_model_params['global_orient'] for body_model_params in body_params_list],
                    dim=1)
                body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                        dim=1)
                batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            with torch.inference_mode(mode=False):
                batch_clone = {key: value.clone() for key, value in batch_inputs.items()}
                model_outputs = self.model(batch_clone, id)
            results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if id==-1:
            instance_mask = torch.cat([result["acc_person_list"] for result in results], dim=0)
            instance_mask = instance_mask.reshape(*img_size, -1)
            for i in range(instance_mask.shape[2]):
                instance_mask_i = instance_mask[:, :, i]
                os.makedirs(f"test_instance_mask_{prefix}/{i}", exist_ok=True)
                cv2.imwrite(f"test_instance_mask_{prefix}/{i}/{int(idx.cpu().numpy()):04d}.png", instance_mask_i.cpu().numpy() * 255)


        if results[0]['rgb'] is not None:
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = (rgb * 255).astype(np.uint8)

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering_{prefix}/{id}/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])

    def test_step(self, batch, *args, **kwargs):
        self.model.eval()

        inputs, targets, pixel_per_batch, total_pixels, idx = batch

        if "novel_view" in inputs.keys():
            if idx==0:
                print(f"novel view {inputs['novel_view']} with pose from dataset")
            self.test_step_each_person_novel(batch, id=-1, novel_view=inputs["novel_view"])
        elif "free_view" in inputs.keys():
            print("free view prefix")
            self.test_step_each_person_denoisy(batch, id=-1, prefix='free_view')
            if self.num_person > 1:
                for i in range(self.num_person):
                    self.test_step_each_person_denoisy(batch, id=i, prefix='free_view')
        else:
            # test all persons
            self.test_step_each_person(batch, id=-1)
            # # uncomment this to test each person
            # if self.num_person > 1:
            #     for i in range(self.num_person):
            #         self.test_step_each_person(batch, id=i)
