import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.opacity_sparse_weight = opt.opacity_sparse_weight
        self.in_shape_weight = opt.in_shape_weight
        self.sam_mask_weight = opt.sam_mask_weight
        self.smpl_surface_weight = opt.get('smpl_surface_weight', 0)
        self.zero_pose_weight = opt.get('zero_pose_weight', 0)
        self.sam_start_epoch = opt.get('sam_start_epoch', 200)
        self.increase_sam = opt.get('increase_sam', False)
        self.temporal_loss_weight = opt.get('temporal_loss_weight', 1.0)
        self.eps = 1e-6
        self.milestone = 200
        self.sam_milestone = 1000
        self.smpl_surface_milestone = opt.smpl_surface_milestone
        self.depth_loss_milestone = 1000
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l1_sum_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
    
    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_los(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # Global opacity sparseness regularization 
    def get_opacity_sparse(self, acc_map, index_off_surface):
        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss

    def get_sam_mask_loss(self, sam_mask, acc_person):
        sam_mask = self.sigmoid(sam_mask)
        valid_mask = sam_mask.sum(dim=1) <= (1 + 1e-2)
        loss = self.l1_loss(acc_person[valid_mask], sam_mask[valid_mask])
        # import pdb;pdb.set_trace()
        return loss

    def get_sam_mask_clip_loss(self, sam_mask, acc_person):
        batch_size = sam_mask.shape[0]
        person_num = sam_mask.shape[1]
        sam_mask = self.sigmoid(sam_mask)
        valid_mask = sam_mask.sum(dim=1) <= (1 + 1e-2)
        valid_acc_person = acc_person[valid_mask].reshape(-1)
        valid_sam_mask = sam_mask[valid_mask].reshape(-1)
        min_min = torch.logical_and(valid_acc_person < 0.04, valid_sam_mask < 0.04)
        max_max = torch.logical_and(valid_acc_person > 0.96, valid_sam_mask > 0.96)
        clip_mask = torch.logical_or(min_min, max_max)
        clip_mask = torch.logical_not(clip_mask)
        # check if clip_mask is all False
        if clip_mask.sum() == 0:
            print('clip_mask is all False')
            clip_mask[0] = True
        loss = self.l1_sum_loss(valid_acc_person[clip_mask], valid_sam_mask[clip_mask]) / (batch_size * person_num)
        # import pdb;pdb.set_trace()
        return loss

    def get_depth_order_loss(self, t_list, fg_rgb_values_each_person_list_nan_filter, mean_hitted_vertex_list, rgb_gt_nan_filter_hitted, cam_loc):
        front_person_index = np.argmin(t_list, axis=0)
        rgb_loss = torch.norm(fg_rgb_values_each_person_list_nan_filter - rgb_gt_nan_filter_hitted.reshape(1, -1, 3), dim=-1)
        correct_rgb_person_index = torch.argmin(rgb_loss, dim=0)
        # mean_hitted_vertex_list shape (2, 7, 3)
        d1, d2, d3= mean_hitted_vertex_list.shape
        # import pdb;pdb.set_trace()
        front_vertex_position = mean_hitted_vertex_list[front_person_index, torch.arange(d2), :]
        correct_vertex_position = mean_hitted_vertex_list[correct_rgb_person_index, torch.arange(d2), :]
        dist_correct = torch.norm(correct_vertex_position - cam_loc, dim=-1)
        dist_front = torch.norm(front_vertex_position - cam_loc, dim=-1)
        loss = torch.log(1+torch.exp(dist_correct - dist_front)).sum()
        return loss
        # front_vertex_position = front_vertex_position[:, torch.arange(d2)]

    def get_depth_order_loss_samGT(self, t_list, mean_hitted_vertex_list, sam_mask, cam_loc):
        front_person_index = np.argmin(t_list, axis=0)
        # check sam_mask shape (number_pixel, number_person)
        correct_rgb_person_index = np.argmax(sam_mask.cpu().numpy(), axis=1)
        d1, d2, d3 = mean_hitted_vertex_list.shape
        # import pdb;pdb.set_trace()
        front_vertex_position = mean_hitted_vertex_list[front_person_index, torch.arange(d2), :]
        correct_vertex_position = mean_hitted_vertex_list[correct_rgb_person_index, torch.arange(d2), :]
        dist_correct = torch.norm(correct_vertex_position - cam_loc, dim=-1)
        dist_front = torch.norm(front_vertex_position - cam_loc, dim=-1)
        loss = torch.log(1 + torch.exp(dist_correct - dist_front)).sum()
        return loss

    def forward(self, model_outputs, ground_truth):
        if isinstance(model_outputs['fg_rgb_values_each_person_list'], list):
            depth_order_loss = torch.zeros((1),device=model_outputs['acc_map'].device)
        else:
            if 'sam_mask' in model_outputs.keys() and True:
                sam_mask = model_outputs['sam_mask']
                sam_mask_reorder = sam_mask[model_outputs["hitted_mask_idx"]]
                sam_mask_reorder_filter = sam_mask_reorder
                depth_order_loss = self.get_depth_order_loss_samGT(model_outputs['t_list'],
                                                             model_outputs['mean_hitted_vertex_list'],
                                                             sam_mask_reorder_filter, model_outputs['cam_loc'][model_outputs["hitted_mask_idx"]])

        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bce_loss = self.get_bce_los(model_outputs['acc_map'])
        if bce_loss.isnan():
            print("Nan: bce_loss")
            print(model_outputs['acc_map'])
            bce_loss = torch.zeros((1),device=bce_loss.device)
        # opacity_sparse_loss = self.get_opacity_sparse(model_outputs['acc_map'], model_outputs['index_off_surface'])
        opacity_sparse_loss = torch.zeros((1),device=bce_loss.device)
        if model_outputs['index_in_surface'] is not None:
            in_shape_loss = self.get_in_shape_loss(model_outputs['acc_map'], model_outputs['index_in_surface'])
        else:
            in_shape_loss = torch.zeros((1),device=bce_loss.device)
        if in_shape_loss.isnan():
            print("Nan: in_shape_loss")
            print(model_outputs['acc_map'])
            print(model_outputs['index_in_surface'])
            in_shape_loss = torch.zeros((1),device=bce_loss.device)
        curr_epoch_for_loss = min(self.milestone, model_outputs['epoch']) # will not increase after the milestone
        temporal_loss = model_outputs['temporal_loss']
        smpl_surface_loss = model_outputs['smpl_surface_loss'] * self.smpl_surface_weight
        if 'sam_mask' in model_outputs.keys() and model_outputs['epoch'] >= self.sam_start_epoch:
            sam_mask_loss = self.get_sam_mask_clip_loss(model_outputs['sam_mask'], model_outputs['acc_person_list'])
        else:
            sam_mask_loss = torch.zeros((1),device=temporal_loss.device)
        if model_outputs['epoch'] >= self.sam_start_epoch:
            depth_order_loss = 1.0 * depth_order_loss * (1 - min(self.depth_loss_milestone, model_outputs['epoch']) / self.depth_loss_milestone)
        else:
            depth_order_loss = torch.zeros((1), device=model_outputs['acc_map'].device)
        zero_pose_loss = model_outputs['zero_pose_loss'] * self.zero_pose_weight * (1 - min(1000, model_outputs['epoch']) / 1000)
        if self.increase_sam:
            increase_weight = min(1.0, model_outputs['epoch'] / 100)
        else:
            increase_weight = 1.0
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.bce_weight * bce_loss + \
               self.opacity_sparse_weight * (1 + curr_epoch_for_loss ** 2 / 40) * opacity_sparse_loss + \
               self.in_shape_weight * (1 - curr_epoch_for_loss / self.milestone) * in_shape_loss + \
               temporal_loss * self.temporal_loss_weight + \
               self.sam_mask_weight * sam_mask_loss * increase_weight + \
               smpl_surface_loss * (1 - min(self.smpl_surface_milestone, model_outputs['epoch']) / self.smpl_surface_milestone) + \
               depth_order_loss + zero_pose_loss
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'depth_order_loss': depth_order_loss,
            'eikonal_loss': eikonal_loss,
            'bce_loss': bce_loss,
            'opacity_sparse_loss': opacity_sparse_loss,
            'in_shape_loss': in_shape_loss,
            'temporal_loss': temporal_loss,
            'sam_mask_loss': sam_mask_loss,
            'smpl_surface_loss': smpl_surface_loss,
            'zero_pose_loss': model_outputs['zero_pose_loss'],
        }