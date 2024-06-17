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

class ThreeDPWDataset(torch.utils.data.Dataset):
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
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.load(os.path.join(root, 'poses.npy'))[self.training_indices] # [::self.skip_step]
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))[self.training_indices] # [::self.skip_step]
        # cameras
        self.P, self.C = [], []

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
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


    def __len__(self):
        return self.n_images # len(self.images)

    def load_body_model_params(self):
        body_model_params = {}
        return body_model_params
    def __getitem__(self, idx):

        # normalize RGB
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        mask = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask -> Tensor
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0

        img_size = self.img_size # [idx]

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float() 

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
            }
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
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "idx": idx
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size
            }
            return inputs, images

class ThreeDPWValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = ThreeDPWDataset(opt)
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

class ThreeDPWTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = ThreeDPWDataset(opt)

        self.img_size = self.dataset.img_size # [0]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "P": inputs["P"],
            "C": inputs["C"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            # "normal": images["normal"],
            "img_size": images["img_size"]
        }
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx