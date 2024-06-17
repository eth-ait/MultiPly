import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import rend_util
def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack([
        dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]
    ], axis=1).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1

def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where((samples_uniform_row < bbox_min[0]) | (samples_uniform_row > bbox_max[0]) | (samples_uniform_col < bbox_min[1]) | (samples_uniform_col > bbox_max[1]))[0]
    return index_outside

def edge_sampling(data, num_sample, ratio_mask=0.5, ratio_edge=0.4,):
    assert ratio_mask >= 0.0
    assert ratio_edge >= 0.0
    assert ratio_edge + ratio_mask <= 1.0

    num_mask = int(num_sample * ratio_mask)
    num_edge = int(num_sample * ratio_edge)
    num_rand = num_sample - num_mask - num_edge
    mask = data["person_mask"].reshape(-1)
    mask_e = data["edge_mask"].reshape(-1)

    mask_loc, *_ = np.where(mask)
    edge_loc, *_ = np.where(mask_e)

    mask_idx = np.random.randint(0, len(mask_loc), num_mask)
    edge_idx = np.random.randint(0, len(edge_loc), num_edge)
    rand_idx = np.random.randint(0, len(mask), num_rand)

    mask_idx = mask_loc[mask_idx]
    edge_idx = edge_loc[edge_idx]

    indices = np.concatenate([mask_idx, edge_idx, rand_idx], axis=0)
    output = {}
    for key, val in data.items():
        val = val.reshape(len(mask), -1)
        output[key] = val[indices]
    return output


def weighted_sampling(data, img_size, num_sample):
    # calculate bounding box
    mask = data["object_mask"]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)

    num_sample_bbox = int(num_sample * 0.9)
    samples_bbox = np.random.rand(num_sample_bbox, 2)
    samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.rand(num_sample_uniform, 2)
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    # get indices for uniform samples outside of bbox
    index_outside = get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max) + num_sample_bbox

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack([
                bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                for i in range(val.shape[2])
            ], axis=-1)
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val
    
    return output, index_outside

class Hi4DDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = opt.start_frame
        self.end_frame = opt.end_frame 
        self.skip_step = 1
        self.images, self.img_sizes = [], []
        self.object_masks = []
        self.training_indices = list(range(opt.start_frame, opt.end_frame, self.skip_step))

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.n_images = len(self.img_paths)

        # masks
        mask_dir = os.path.join(root, "mask")
        self.mask_folder_list = sorted(glob.glob(f"{mask_dir}/*"))
        self.mask_path_list = []
        for mask_folder in self.mask_folder_list:
            mask_path = sorted(glob.glob(f"{mask_folder}/*.png"))
            mask_path = [mask_path[i] for i in self.training_indices]
            self.mask_path_list.append(mask_path)

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.num_person = self.shape.shape[0]
        self.poses = np.load(os.path.join(root, 'poses.npy'))[self.training_indices] # [::self.skip_step]
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))[self.training_indices] # [::self.skip_step]
        # cameras
        self.P, self.C = [], []

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)

        self.scale = 1 / scale_mats[0][0, 0]

        self.scale_mat_all = []
        self.world_mat_all = []
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            self.scale_mat_all.append(scale_mat)
            self.world_mat_all.append(world_mat)
            self.P.append(P)
            C = -np.linalg.solve(P[:3, :3], P[:3, 3])
            self.C.append(C)
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all) # == len(self.images)

        # other properties
        self.num_sample = opt.num_sample
        self.sampling_strategy = "weighted"
        self.using_SAM = opt.using_SAM
        self.pre_mask_path = ""
        self.pre_mask = None
        self.smpl_sam_iou = np.ones(len(self.img_paths))
        self.uncertain_thereshold = 0.0
        self.uncertain_frame_list = []
        
        self.init_params(opt)

        if self.edge_sampling:
            # edge mask
            edge_dir = os.path.join(root, "edge")
            self.edge_paths = sorted(glob.glob(f"{edge_dir}/*.png"))
        else:
            self.edge_paths = None
    

    def init_params(self, opt):
        # the higher the ratio, the more uncertain frames
        self.ratio_uncertain = opt.get("ratio_uncertain", 0.5)
        # ratio decrease for every 50 epochs
        self.ratio_decrease = opt.get("ratio_decrease", 0.0)
        self.edge_sampling = opt.get("edge_sampling", False)


    def __len__(self):
        return self.n_images # len(self.images)

    def load_body_model_params(self):
        body_model_params = {}
        return body_model_params
    def __getitem__(self, idx):
        is_certain = True
        if self.using_SAM:
            mask_list = sorted(glob.glob(f"stage_sam_mask/*"))
            if len(mask_list) == 0:
                # if idx == 0:
                #     print("epoch 0, not using sam mask")
                sam_mask = None
            else:
                mask_path = os.path.join(mask_list[-1], "sam_opt_mask.npy")
                if mask_path != self.pre_mask_path:
                    smpl_mask_list = sorted(glob.glob(f"stage_instance_mask/*"))
                    smpl_mask_path = os.path.join(smpl_mask_list[-1], "all_person_smpl_mask.npy")
                    smpl_mask = np.load(smpl_mask_path) > 0.8
                    try:
                        sam_mask_binary = np.load(mask_path) > 0.0
                    except:
                        mask_path = self.pre_mask_path
                        print("ERROR: cannot load current sam mask, use previous sam mask")
                        sam_mask_binary = np.load(mask_path) > 0.0
                    self.smpl_sam_iou = np.logical_and(sam_mask_binary, smpl_mask).sum(axis=(2, 3)) / np.logical_or(sam_mask_binary, smpl_mask).sum(axis=(2, 3))
                    self.smpl_sam_iou = self.smpl_sam_iou.mean(axis=-1)
                    # ascending order
                    sorted_smpl_sam_iou = np.sort(self.smpl_sam_iou)
                    self.uncertain_thereshold = sorted_smpl_sam_iou[int(len(sorted_smpl_sam_iou) * self.ratio_uncertain)]
                    self.ratio_uncertain -= self.ratio_decrease
                    self.uncertain_frame_list = []
                    for i, iou in enumerate(self.smpl_sam_iou):
                        if iou < self.uncertain_thereshold:
                            self.uncertain_frame_list.append(i)

                    # shape from (F, P, H, W) to (F, H, W, P)
                    sam_mask = np.load(mask_path).transpose(0, 2, 3, 1)
                    self.pre_mask_path = mask_path
                    self.pre_mask = sam_mask
                    sam_mask = sam_mask[idx]
                else:
                    mask_path = self.pre_mask_path
                    sam_mask = self.pre_mask
                    sam_mask = sam_mask[idx]
                # if idx == 0:
                #     print("current using sam mask from file ", mask_path)
                #     print("uncertain_thereshold ", self.uncertain_thereshold)
                #     print("uncertain_frame_list ", self.uncertain_frame_list)

            iou_frame_i = self.smpl_sam_iou[idx]
            is_certain = iou_frame_i >= self.uncertain_thereshold

        # normalize RGB
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        mask_array = []
        for mask_paths in self.mask_path_list:
            mask = cv2.imread(mask_paths[idx])
            # preprocess: BGR -> Gray -> Mask -> Tensor
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
            mask_array.append(mask)
        mask = np.stack(mask_array, axis=-1)
        mask = np.sum(mask, axis=-1)

        if self.edge_sampling:
            edge = cv2.imread(self.edge_paths[idx])
            # preprocess: BGR -> Gray -> Mask -> Tensor
            edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY) > 0
            edge_mask = np.logical_and(mask, edge)

        img_size = self.img_size # [idx]

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([self.num_person, 86]).float()
        smpl_params[:, 0] = torch.from_numpy(np.asarray(self.scale)).float()

        smpl_params[:, 1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[:, 4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[:, 76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
            }
            if self.using_SAM and sam_mask is not None:
                data.update({"sam_mask": sam_mask})
            samples, index_outside = weighted_sampling(data, img_size, self.num_sample)
            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                'index_outside': index_outside,
                "idx": idx,
                "smpl_sam_iou": self.smpl_sam_iou,
                "is_certain": is_certain,
            }
            if self.using_SAM and sam_mask is not None:
                inputs.update({"sam_mask": samples['sam_mask']})
                inputs.update({"org_sam_mask": sam_mask})
            images = {"rgb": samples["rgb"].astype(np.float32)}
            inputs.update({"org_img": img.astype(np.float32)})
            inputs.update({"img_size": self.img_size})
            if self.edge_sampling:
                data = {
                    "rgb": img,
                    "uv": uv,
                    "person_mask": mask,
                    "edge_mask": edge_mask,
                }
                if self.using_SAM and sam_mask is not None:
                    data.update({"sam_mask": sam_mask})
                edge_samples = edge_sampling(data, self.num_sample)
                inputs.update({"edge_uv": edge_samples["uv"].astype(np.float32)})
                images.update({"edge_rgb": edge_samples["rgb"].astype(np.float32)})
                if self.using_SAM and sam_mask is not None:
                    inputs.update({"edge_sam_mask": edge_samples['sam_mask']})

            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "idx": idx,
                "org_uv": uv,
                "org_img": img,
                "org_object_mask": mask,
            }
            inputs.update({"img_size": self.img_size})
            if self.using_SAM and sam_mask is not None:
                inputs.update({"org_sam_mask": sam_mask})
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size
            }
            return inputs, images

class Hi4DValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Hi4DDataset(opt)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))  
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "P": inputs["P"],
            "C": inputs["C"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            'image_id': image_id,
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': self.total_pixels
        }
        return inputs, images

class Hi4DTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Hi4DDataset(opt)

        self.img_size = self.dataset.img_size # [0]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch
        try:
            self.novel_view = opt.novel_view
            self.current_view = opt.current_view
            self.pair = opt.pair
            self.action = opt.action
            GT_DIR = opt.GT_DIR
            GT_folder = os.path.join(GT_DIR, self.pair, self.action)
            GT_camera_path = os.path.join(GT_folder, 'cameras', 'rgb_cameras.npz')
            print("novel_view: ", self.novel_view)
            print("current_view: ", self.current_view)
            print("Attention, we assume intrinsics is the same between gt and the intrinsics used during training. Also we assume we use GT pose for training and testing.")
        except:
            self.novel_view = None
            self.current_view = None
            print("Not novel view synthesis")

        if self.novel_view is not None and self.current_view is not None:
            cameras = dict(np.load(GT_camera_path))
            c_cur = int(np.where(cameras['ids'] == self.current_view)[0])
            self.gt_cam_intrinsics_cur = cameras['intrinsics'][c_cur]
            self.gt_cam_extrinsics_cur = cameras['extrinsics'][c_cur]
            c_tgt = int(np.where(cameras['ids'] == self.novel_view)[0])
            self.gt_cam_intrinsics_tgt = cameras['intrinsics'][c_tgt]
            self.gt_cam_extrinsics_tgt = cameras['extrinsics'][c_tgt]

            self.new_P = []
            self.new_C = []
            self.new_intrinsics_all = []
            self.new_pose_all = []
            for scale_mat, world_mat in zip(self.dataset.scale_mat_all, self.dataset.world_mat_all):
                intrinsics_training_cur, pose_training_cur = rend_util.load_K_Rt_from_P(None, world_mat[:3, :4])
                scale_factor = self.gt_cam_intrinsics_cur[0, 0] / intrinsics_training_cur[0, 0]
                R_cur = pose_training_cur[:3, :3].transpose()
                t_cur = -R_cur @ pose_training_cur[:3, 3]
                R3 = R_cur
                t3 = t_cur
                R1 = self.gt_cam_extrinsics_cur[:3, :3]
                t1 = self.gt_cam_extrinsics_cur[:3, 3]
                Rab = R3.transpose() @ R1
                tab = R3.transpose() @ (t1 - t3)
                R2 = self.gt_cam_extrinsics_tgt[:3, :3]
                t2 = self.gt_cam_extrinsics_tgt[:3, 3]
                R4 = R2 @ Rab.transpose()
                t4 = t2 - R4 @ tab
                novel_world_mat = np.eye(4)

                scaled_gt_cam_intrinsics_tgt = self.gt_cam_intrinsics_tgt[:3, :3].copy()
                scaled_gt_cam_intrinsics_tgt[0, 0] = scaled_gt_cam_intrinsics_tgt[0, 0] / scale_factor
                scaled_gt_cam_intrinsics_tgt[1, 1] = scaled_gt_cam_intrinsics_tgt[1, 1] / scale_factor
                scaled_gt_cam_intrinsics_tgt[0, 2] = scaled_gt_cam_intrinsics_tgt[0, 2] / scale_factor
                scaled_gt_cam_intrinsics_tgt[1, 2] = scaled_gt_cam_intrinsics_tgt[1, 2] / scale_factor

                novel_world_mat[:3, :4] = scaled_gt_cam_intrinsics_tgt @ np.concatenate((R4, t4.reshape(3,1)), axis=1)
                P = novel_world_mat @ scale_mat
                self.new_P.append(P)
                C = -np.linalg.solve(P[:3, :3], P[:3, 3])
                self.new_C.append(C)
                P = P[:3, :4]
                intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
                self.new_intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.new_pose_all.append(torch.from_numpy(pose).float())




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.novel_view is not None and self.current_view is not None:
            data = self.dataset[idx]
            inputs, images = data
            inputs = {
                "uv": inputs["uv"],
                "P": self.new_P[idx],
                "C": self.new_C[idx],
                "intrinsics": self.new_intrinsics_all[idx],
                "pose": self.new_pose_all[idx],
                "smpl_params": inputs["smpl_params"],
                "idx": inputs['idx'],
                "novel_view": self.novel_view,
            }
            images = {
                "rgb": images["rgb"],
                "img_size": images["img_size"]
            }
            return inputs, images, self.pixel_per_batch, self.total_pixels, idx



        else:
            data = self.dataset[idx]
            inputs, images = data
            inputs_new = {
                "uv": inputs["uv"],
                "P": inputs["P"],
                "C": inputs["C"],
                "intrinsics": inputs['intrinsics'],
                "pose": inputs['pose'],
                "smpl_params": inputs["smpl_params"],
                "idx": inputs['idx'],
                "img_size": torch.from_numpy(np.array(inputs["img_size"])),
            }
            if "org_sam_mask" in inputs.keys():
                inputs_new.update({"org_sam_mask": inputs["org_sam_mask"]})

            images = {
                "rgb": images["rgb"],
                "img_size": images["img_size"],
                "org_uv": inputs["org_uv"],
                "org_img": inputs["org_img"],
                "org_object_mask": inputs["org_object_mask"],
            }
            if "org_sam_mask" in inputs.keys():
                images.update({"org_sam_mask": inputs["org_sam_mask"]})
            return inputs_new, images, self.pixel_per_batch, self.total_pixels, idx

class Hi4DTestFreeDataset(torch.utils.data.Dataset):
    def __init__(self, opt, image_id, step_list, offset=0, scale_factor=1.0):
        self.dataset = Hi4DDataset(opt)
        self.name_offset = offset
        self.img_size = self.dataset.img_size # [0]
        self.step_list = step_list
        self.scale_factor = scale_factor

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch
        if True:
            start = 0
            steps = 60
            step_size = 6
            self.new_poses = []
            self.image_id = image_id
            self.data = self.dataset[self.image_id]
            self.img_size = self.dataset.img_size  # [self.image_id]
            self.total_pixels = np.prod(self.img_size)
            self.pixel_per_batch = 512
            target_inputs, images = self.data
            from scipy.spatial.transform import Rotation as scipy_R
            if len(self.step_list) > 0:
                for i in self.step_list:
                    rotation_angle_y = i
                    pose = target_inputs['pose'].clone()
                    new_pose = rend_util.get_new_cam_pose_fvr(pose, rotation_angle_y)
                    self.new_poses.append(new_pose)
            else:
                for i in range(self.name_offset,steps):
                    rotation_angle_y = start + i * (step_size)
                    pose = target_inputs['pose'].clone()
                    new_pose = rend_util.get_new_cam_pose_fvr(pose, rotation_angle_y)
                    self.new_poses.append(new_pose)




    def __len__(self):
        return len(self.new_poses)
        # return len(self.dataset)

    def __getitem__(self, idx):
        uv = np.mgrid[:self.img_size[0], :self.img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        target_inputs, images = self.data
        zoomed_intrinsics = target_inputs['intrinsics'].clone()
        zoomed_intrinsics[0, 0] = zoomed_intrinsics[0, 0] / self.scale_factor
        zoomed_intrinsics[1, 1] = zoomed_intrinsics[1, 1] / self.scale_factor
        inputs = {
            "uv": uv.reshape(-1, 2).astype(np.float32),
            "intrinsics": zoomed_intrinsics,
            "pose": self.new_poses[idx],  # target_inputs['pose'], # self.pose_all[idx],
            'P': target_inputs['P'],
            'C': target_inputs['C'],
            "smpl_params": target_inputs["smpl_params"],
            # 'image_id': self.image_id,
            'idx': self.image_id,
            "img_size": torch.from_numpy(np.array(self.img_size)),
            'free_view': self.image_id,
        }
        images = {
            "img_size": self.img_size}
        return inputs, images, self.pixel_per_batch, self.total_pixels, self.name_offset + 1000 * self.image_id + self.step_list[idx]