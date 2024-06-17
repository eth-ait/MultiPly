import numpy as np
import pickle as pkl
import torch
import trimesh
import cv2
import os
from tqdm import tqdm
import glob
import argparse
from preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer,RendererNew, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
from loss import joints_2d_loss, pose_temporal_loss, get_loss_weights
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from rotation import axis_to_rot6D, rot6D_to_axis

def interpolate_rotations(rotations, ts_in, ts_out):
    """
    Interpolate rotations given at timestamps `ts_in` to timestamps given at `ts_out`. This performs the equivalent
    of cubic interpolation in SO(3).
    :param rotations: A numpy array of rotations of shape (F, N, 3), i.e. rotation vectors.
    :param ts_in: Timestamps corresponding to the given rotations, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    out = []
    for j in range(rotations.shape[1]):
        rs = R.from_rotvec(rotations[:, j])
        spline = RotationSpline(ts_in, rs)
        rs_interp = spline(ts_out).as_rotvec()
        out.append(rs_interp[:, np.newaxis])
    return np.concatenate(out, axis=1)

def interpolate_positions(positions, ts_in, ts_out):
    """
    Interpolate positions given at timestamps `ts_in` to timestamps given at `ts_out` with a cubic spline.
    :param positions: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given positions, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    cs = CubicSpline(ts_in, positions, axis=0)
    new_positions = cs(ts_out)
    return new_positions

def interpolate(n_frames ,frame_ids, pose_list, trans_list):
    """
    Replace the frames at the given frame IDs via an interpolation of its neighbors. Only the body pose as well
    as the root pose and translation are interpolated.
    :param frame_ids: A list of frame ids to be interpolated.
    """
    ids = np.unique(frame_ids)
    all_ids = np.arange(n_frames)
    mask_avail = np.ones(n_frames, dtype=np.bool)
    mask_avail[ids] = False

    # Interpolate poses.
    all_poses = pose_list
    ps = np.reshape(all_poses, (n_frames, -1, 3))
    ps_interp = interpolate_rotations(ps[mask_avail], all_ids[mask_avail], ids)
    all_poses[ids] = np.array(ps_interp.reshape(len(ids), -1))

    # Interpolate global translation.
    ts = trans_list
    ts_interp = interpolate_positions(ts[mask_avail], all_ids[mask_avail], ids)
    trans_list[ids] = np.array(ts_interp)
    return all_poses, trans_list


def transform_smpl_remain_extrinsic(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root

    target_extrinsic[:3, -1] = curr_extrinsic[:3, :3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3, :3] @ T_hip

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume

    smpl_trans = smpl_trans + (np.linalg.inv(target_extrinsic[:3, :3]) @ target_extrinsic[:3, -1])
    target_extrinsic[:3, -1] = np.zeros_like(target_extrinsic[:3, -1])

    return target_extrinsic, smpl_pose, smpl_trans

def main(args):
    max_human_sphere_all = 0
    device = torch.device("cuda:0")
    seq = args.seq
    DIR = './raw_data'
    img_dir = f'{DIR}/{seq}/frames'   
    trace_file_dir = f'{DIR}/{seq}/trace'
    if args.source == 'hi4d':
        img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    else:
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    if len(img_paths) == 0:
        img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))

    # format: [person_id, frame_id, ...]
    trace_file_path = f"{trace_file_dir}/{seq}.npz"
    trace_output = np.load(trace_file_path, allow_pickle=True)["results"][()]
    number_person = trace_output['smpl_betas'].shape[0]

    from smplx import SMPL
    smpl_model_list = []
    for i in range(number_person):
        smpl_model_list.append(SMPL('../code/lib/smpl/smpl_model', gender="NEUTRAL").to(device))
    
    input_img = cv2.imread(img_paths[0])
    if args.source == 'custom':
        focal_length = max(input_img.shape[0], input_img.shape[1])
        cam_intrinsics = np.array([[focal_length, 0., input_img.shape[1]//2],
                                   [0., focal_length, input_img.shape[0]//2],
                                   [0., 0., 1.]])
    elif args.source == 'neuman':
        NeuMan_DIR = '' # path to NeuMan dataset
        with open(f'{NeuMan_DIR}/{seq}/sparse/cameras.txt') as f:
            lines = f.readlines()
        cam_params = lines[3].split()
        cam_intrinsics = np.array([[float(cam_params[4]), 0., float(cam_params[6])], 
                                   [0., float(cam_params[5]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif args.source == 'deepcap':
        DeepCap_DIR = '' # path to DeepCap dataset
        with open(f'{DeepCap_DIR}/monocularCalibrationBM.calibration') as f:
            lines = f.readlines()

        cam_params = lines[5].split()
        cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
                                   [0., float(cam_params[6]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif args.source == 'hi4d':
        Hi4D_DIR = '' # path to hi4d dataset
        pair = '' # pair name
        action = '' # action name
        camera_path = f"{Hi4D_DIR}/{pair}/{action}/cameras/rgb_cameras.npz"
        cameras = dict(np.load(camera_path))
        cam_view = int(args.seq.split('_')[-1])
        print("cam_view", cam_view)
        c = int(np.where(cameras['ids'] == cam_view)[0])
        cam_intrinsics = cameras['intrinsics'][c]
        cam_intrinsics[0,1]=0.0
        gt_extrinsics = cameras['extrinsics'][c]
        print("cam_intrinsics", cam_intrinsics)
        print("gt_extrinsics", gt_extrinsics)

    elif args.source == 'iphone':
        cam_intrinsics = np.array([[1424,0,712.67],
                                    [0,1424,972.35],
                                    [0, 0, 1],])
    else:
        print('Please specify the source of the dataset (custom, neuman, deepcap). We will continue to update the sources in the future.')
        raise NotImplementedError
    renderer = RendererNew(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)

    if args.mode == 'mask':
        if not os.path.exists(f'{DIR}/{seq}/init_mask'):
            os.makedirs(f'{DIR}/{seq}/init_mask')
        if not os.path.exists(f'{DIR}/{seq}/init_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_smpl_files')
        if not os.path.exists(f'{DIR}/{seq}/init_smpl_image'):
            os.makedirs(f'{DIR}/{seq}/init_smpl_image')
        mean_shape = []
    elif args.mode == 'refine':
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_mask'):
            os.makedirs(f'{DIR}/{seq}/init_refined_mask')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl_files')
        init_smpl_dir = f'{DIR}/{seq}/init_smpl_files'
        init_smpl_paths = sorted(glob.glob(f"{init_smpl_dir}/*.pkl"))
        if args.vitpose:
            print("using vitpose")
            openpose_dir = f'{DIR}/{seq}/vitpose'
            openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))
        else:
            print("using openpose")
            openpose_dir = f'{DIR}/{seq}/openpose'
            openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))
        if args.openpose:
            print("using openpose")
            openpose_dir_true = f'{DIR}/{seq}/openpose'
            openpose_paths_true = sorted(glob.glob(f"{openpose_dir_true}/*.npy"))

        opt_num_iters=150
        weight_dict = get_loss_weights()
        cam = PerspectiveCamera(focal_length_x=torch.tensor(cam_intrinsics[0, 0], dtype=torch.float32),
                                focal_length_y=torch.tensor(cam_intrinsics[1, 1], dtype=torch.float32),
                                center=torch.tensor(cam_intrinsics[0:2, 2]).unsqueeze(0)).to(device)
        mean_shape = []
        if args.vitpose:
            print("using vitpose")
            smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco17'), dtype=torch.long).cuda()
        else:
            print("using openpose")
            smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()
        if args.openpose:
            print("using openpose")
            smpl2op_mapping_openpose = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()
        
        J_regressor_extra9 = np.load('./J_regressor_extra.npy')
        J_regressor_extra9 = torch.tensor(J_regressor_extra9, dtype=torch.float32).cuda()
    elif args.mode == 'final':
        refined_smpl_dir = f'{DIR}/{seq}/init_refined_smpl_files'
        refined_smpl_mask_dir = f'{DIR}/{seq}/init_refined_mask'
        refined_smpl_paths = sorted(glob.glob(f"{refined_smpl_dir}/*.pkl"))
        refined_smpl_mask_paths = [sorted(glob.glob(f"{refined_smpl_mask_dir}/{person_i}/*.png")) for person_i in range(number_person)]

        save_dir = f'../data/{seq}'
        if not os.path.exists(os.path.join(save_dir, 'image')):
            os.makedirs(os.path.join(save_dir, 'image'))
        if not os.path.exists(os.path.join(save_dir, 'mask')):
            os.makedirs(os.path.join(save_dir, 'mask'))

        scale_factor = args.scale_factor
        smpl_shape = np.load(f'{DIR}/{seq}/mean_shape.npy')

        K = np.eye(4)
        K[:3, :3] = cam_intrinsics
        K[0, 0] = K[0, 0] / scale_factor
        K[1, 1] = K[1, 1] / scale_factor
        K[0, 2] = K[0, 2] / scale_factor
        K[1, 2] = K[1, 2] / scale_factor

        dial_kernel = np.ones((20, 20),np.uint8)

        output_trans = []
        output_pose = []
        output_P = {}

    last_j3d = None
    actor_id = 0
    last_smpl_verts = []
    last_pj2d = []
    last_j3d = []
    last_smpl_shape = []
    last_smpl_pose = []
    last_smpl_trans = []
    last_cam_trans = []

    last_pose = [[] for _ in range(number_person)]
    cam_extrinsics = np.eye(4)
    R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
    T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
    if args.mode == 'refine':
        print('start interpolate smpl')
        init_pose_list = []
        init_shape_list = []
        init_trans_list = []
        for idx, img_path in enumerate(tqdm(img_paths)):
            seq_file = pkl.load(open(init_smpl_paths[idx], 'rb'))
            init_pose_list.append(seq_file['pose'])
            init_shape_list.append(seq_file['shape'])
            init_trans_list.append(seq_file['trans'])
        init_pose_list = np.array(init_pose_list)
        init_shape_list = np.array(init_shape_list)
        init_trans_list = np.array(init_trans_list)
        # you can set the interpolate_frame_id / tracking_frame_id to interpolate or track the pose from previous frame for missing pose estimation
        interpolate_frame_list = []
        tracking_frame_list = []
        tracking_person_id = []
        if len(interpolate_frame_list) > 0:
            for person_i in range(number_person):
                init_pose_list[:, person_i] , init_trans_list[:, person_i] = interpolate(init_pose_list.shape[0], interpolate_frame_list, init_pose_list[:, person_i], init_trans_list[:, person_i])

    normalize_shift_first_frame = None
    for idx, img_path in enumerate(tqdm(img_paths)):
        input_img = cv2.imread(img_path)
        if args.mode == 'mask':
            seq_file = trace_output
            cur_smpl_verts = []
            cur_pj2d = []
            cur_j3d = []
            cur_smpl_shape = []
            cur_smpl_pose = []
            cur_smpl_trans = []
            cur_cam_trans = []
            opt_pose_list = []
            opt_trans_list = []
            opt_shape_list = []

            for person_i in range(number_person):
                if len(seq_file['smpl_thetas']) >= number_person:
                    actor_id = person_i
                    smpl_verts = seq_file['verts'][actor_id, idx]
                    pj2d_org = seq_file['pj2d_org'][actor_id, idx]
                    joints3d = seq_file['joints'][actor_id, idx]
                    smpl_shape = seq_file['smpl_betas'][actor_id, idx][:10]
                    smpl_pose = seq_file['smpl_thetas'][actor_id, idx]
                    cam_trans = seq_file['cam_trans'][actor_id, idx]

                # undetected person
                if len(seq_file['smpl_thetas']) < number_person:
                    # in trace, undetected person should not happened
                    assert False

                cur_cam_trans.append(cam_trans)
                cur_j3d.append(joints3d)
                cur_pj2d.append(pj2d_org)
                cur_smpl_verts.append(smpl_verts)

                opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_cam_tran = torch.tensor(cam_trans[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_zero = torch.tensor(np.zeros((1, 3)), dtype=torch.float32, requires_grad=True, device=device)

                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                         body_pose=opt_pose[:, 3:],
                                         global_orient=opt_pose[:, :3],
                                         transl=opt_zero)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_joints_3d = smpl_output.joints.data.cpu().numpy().squeeze()
                tra_pred = estimate_translation_cv2(smpl_joints_3d[:24], pj2d_org[:24], proj_mat=cam_intrinsics)
                opt_trans = torch.tensor(tra_pred[None], dtype=torch.float32, requires_grad=True, device=device)
                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                         body_pose=opt_pose[:, 3:],
                                         global_orient=opt_pose[:, :3],
                                         transl=opt_trans)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                cur_smpl_trans.append(tra_pred)
                cur_smpl_shape.append(smpl_shape)
                cur_smpl_pose.append(smpl_pose)

                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]

                os.makedirs(f'{DIR}/{seq}/init_mask/{person_i}', exist_ok=True)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_mask/{person_i}', '%04d.png' % idx), valid_mask*255)
                output_img = (rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                for keypoint_idx in range(len(pj2d_org)):
                    cv2.circle(output_img, (int(pj2d_org[keypoint_idx, 0]), int(pj2d_org[keypoint_idx, 1])), 5, (0, 0, 255), -1)
                os.makedirs(f'{DIR}/{seq}/init_smpl_image/{person_i}', exist_ok=True)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smpl_image/{person_i}', '%04d.png' % idx), output_img)

            last_smpl_verts.append(np.stack(cur_smpl_verts, axis=0))
            if idx != 0:
                last_j3d.append(np.stack(cur_j3d, axis=0))
                last_pj2d.append(np.stack(cur_pj2d, axis=0))
                last_cam_trans.append(np.stack(cur_cam_trans, axis=0))
            last_smpl_shape.append(np.stack(cur_smpl_shape, axis=0))
            last_smpl_pose.append(np.stack(cur_smpl_pose, axis=0))
            last_smpl_trans.append(np.stack(cur_smpl_trans, axis=0))

            smpl_dict = {}
            smpl_dict['pose'] = np.array(cur_smpl_pose)
            smpl_dict['trans'] = np.array(cur_smpl_trans)
            smpl_dict['shape'] = np.array(cur_smpl_shape)
            mean_shape.append(smpl_dict['shape'])
            pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smpl_files', '%04d.pkl' % idx), 'wb'))

        if args.mode == 'refine':
            if idx != 0:
                last_opt_pose = opt_pose_list
                last_trans_list = opt_trans_list
                last_shape_list = opt_shape_list

            opt_pose_list = []
            opt_trans_list = []
            opt_shape_list = []
            for person_i in range(number_person):
                if (idx in tracking_frame_list) and (person_i in tracking_person_id):
                    smpl_shape = last_shape_list[person_i]
                    smpl_pose = last_opt_pose[person_i]
                    tra_pred = last_trans_list[person_i]
                else:
                    smpl_shape = init_shape_list[idx, person_i]
                    smpl_pose = init_pose_list[idx, person_i]
                    tra_pred = init_trans_list[idx, person_i]
                opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_trans = torch.tensor(tra_pred[None], dtype=torch.float32, requires_grad=True, device=device)

                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                        body_pose=opt_pose[:, 3:],
                                                        global_orient=opt_pose[:, :3],
                                                        transl=opt_trans)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                if args.mode == 'refine':
                    openpose = np.load(openpose_paths[idx])
                    if idx in interpolate_frame_list:
                        openpose_j2d = 0
                        openpose_conf = 0
                    else:
                        openpose_j2d = torch.tensor(openpose[person_i, :, :2][None], dtype=torch.float32,
                                                    requires_grad=False, device=device)
                        openpose_conf = torch.tensor(openpose[person_i, :, -1][None], dtype=torch.float32,
                                                     requires_grad=False, device=device)
                        keypoint_threshold = 0.6
                        openpose_conf[0, openpose_conf[0, :] < keypoint_threshold] = 0.0

                    
                    if args.openpose:
                        openpose_true = np.load(openpose_paths_true[idx])
                        if idx in interpolate_frame_list:
                            openpose_j2d_true = 0
                            openpose_conf_true = 0
                        else:  
                            openpose_j2d_true = torch.tensor(openpose_true[person_i, :, :2][None], dtype=torch.float32,
                                                        requires_grad=False, device=device)
                            openpose_conf_true = torch.tensor(openpose_true[person_i, :, -1][None], dtype=torch.float32,
                                                        requires_grad=False, device=device)
                            keypoint_threshold_true = 0.4
                            openpose_conf_true[0, openpose_conf_true[0, :] < keypoint_threshold_true] = 0.0
                            openpose_conf_true[0,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] = 0.0
                            # if openpose is far away from the vitpose, then we ignore the openpose
                            if torch.norm(openpose_j2d[0, 15,:] - openpose_j2d_true[0,14,:]) > 30:
                                openpose_conf_true[0, [19,20,21]] = 0.0
                            if torch.norm(openpose_j2d[0, 16,:] - openpose_j2d_true[0,11,:]) > 30:
                                openpose_conf_true[0, [22,23,24]] = 0.0
                            # print('distance L_Ankle ', torch.norm(openpose_j2d[0, 15,:] - openpose_j2d_true[0,14,:]))
                            # print('distance R_Ankle ', torch.norm(openpose_j2d[0, 16,:] - openpose_j2d_true[0,11,:]))

                    smpl_trans = tra_pred

                    opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_trans = torch.tensor(smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_params = [{'params': opt_betas, 'lr': 1e-3},
                                {'params': opt_pose, 'lr': 1e-3},
                                {'params': opt_trans, 'lr': 1e-3}]
                    optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
                    if idx == 0:
                        last_pose[person_i].append(opt_pose.detach().clone())
                        previous_trans = opt_trans.detach().clone()
                    else:
                        previous_trans = torch.tensor(last_trans_list[person_i], dtype=torch.float32, requires_grad=False, device=device)
                        previous_trans = previous_trans[None]
                    loop = tqdm(range(opt_num_iters))
                    for it in loop:
                        optimizer.zero_grad()

                        smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                                body_pose=opt_pose[:, 3:],
                                                                global_orient=opt_pose[:, :3],
                                                                transl=opt_trans)
                        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                        extra_joints9 = torch.einsum('bik,ji->bjk', [smpl_output.vertices, J_regressor_extra9])
                        smpl_54joints = torch.cat([smpl_output.joints, extra_joints9], dim=1)
                        smpl_joints_2d = cam(torch.index_select(smpl_54joints, 1, smpl2op_mapping))
                        if args.openpose:
                            smpl_joints_2d_openpose = cam(torch.index_select(smpl_output.joints, 1, smpl2op_mapping_openpose))
                        previous_pose = last_pose[person_i][0]
                        current_pose = opt_pose
                        previous_pose = previous_pose.reshape(24, 3)
                        current_pose = current_pose.reshape(24, 3)

                        previous_pose_6d = axis_to_rot6D(previous_pose)
                        current_pose_6d = axis_to_rot6D(current_pose)

                        loss = dict()
                        weight = 1
                        if idx in interpolate_frame_list:
                            opt_pose = previous_pose.reshape(1, 72)
                            opt_trans = previous_trans
                            smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                                body_pose=opt_pose[:, 3:],
                                                                global_orient=opt_pose[:, :3],
                                                                transl=opt_trans)
                            smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                            break
                        else:
                            weight = 1
                            loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf, args.vitpose) * weight
                            if args.openpose:
                                loss['J2D_Loss'] = loss['J2D_Loss'] + joints_2d_loss(openpose_j2d_true, smpl_joints_2d_openpose, openpose_conf_true) * weight
                            loss['Temporal_Loss'] = pose_temporal_loss(previous_pose_6d, current_pose_6d) * weight * 5 + pose_temporal_loss(previous_trans, opt_trans)

                        w_loss = dict()
                        for k in loss:
                            w_loss[k] = weight_dict[k](loss[k], it)

                        tot_loss = list(w_loss.values())
                        tot_loss = torch.stack(tot_loss).sum()
                        tot_loss.backward()
                        optimizer.step()

                        l_str = 'Iter: %d' % it
                        for k in loss:
                            l_str += ', %s: %0.4f' % (k, weight_dict[k](loss[k], it).mean().item())
                            loop.set_description(l_str)

                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3, :3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3, :3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]

                if args.mode == 'refine':
                    output_img = (rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(
                        np.uint8)
                    os.makedirs(os.path.join(f'{DIR}/{seq}/init_refined_smpl/{person_i}'), exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_smpl/{person_i}', '%04d.png' % idx), output_img)
                    os.makedirs(os.path.join(f'{DIR}/{seq}/init_refined_mask/{person_i}'), exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_mask/{person_i}', '%04d.png' % idx),
                                valid_mask * 255)
                    last_pose[person_i].pop(0)
                    last_pose[person_i].append(opt_pose.detach().clone())
                    smpl_dict = {}
                    smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
                    smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
                    smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()
                    opt_pose_list.append(smpl_dict['pose'])
                    opt_trans_list.append(smpl_dict['trans'])
                    opt_shape_list.append(smpl_dict['shape'])

            if args.mode == 'refine':
                smpl_dict = {}
                smpl_dict['pose'] = np.array(opt_pose_list)
                smpl_dict['trans'] = np.array(opt_trans_list)
                smpl_dict['shape'] = np.array(opt_shape_list)
                mean_shape.append(smpl_dict['shape'])
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_refined_smpl_files', '%04d.pkl' % idx), 'wb'))

        elif args.mode == 'final':
            input_img = cv2.resize(input_img, (input_img.shape[1] // scale_factor, input_img.shape[0] // scale_factor))
            seq_file = pkl.load(open(refined_smpl_paths[idx], 'rb'))
            cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), input_img)
            smpl_pose_list = []
            smpl_trans_list = []
            smpl_verts_list = []

            for i, smpl_model in enumerate(smpl_model_list):
                mask = cv2.imread(refined_smpl_mask_paths[i][idx])
                mask = cv2.resize(mask, (mask.shape[1] // scale_factor, mask.shape[0] // scale_factor))

                # dilate mask to obtain a coarse bbox
                mask = cv2.dilate(mask, dial_kernel)
                os.makedirs(os.path.join(save_dir, f'mask/{i}'), exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f'mask/{i}/%04d.png' % idx), mask)

                smpl_pose = seq_file['pose'][i]
                smpl_trans = seq_file['trans'][i]

                # transform the spaces such that our camera has the same orientation as the OpenGL camera
                target_extrinsic = np.eye(4)
                target_extrinsic[1:3] *= -1
                T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape[i])[None].float().to(device)).squeeze().cpu().numpy()
                target_extrinsic, smpl_pose, smpl_trans = transform_smpl_remain_extrinsic(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
                smpl_output = smpl_model(betas=torch.tensor(smpl_shape[i])[None].float().to(device),
                                         body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                         global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                         transl=torch.tensor(smpl_trans)[None].float().to(device))
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_pose_list.append(smpl_pose)
                smpl_trans_list.append(smpl_trans)
                smpl_verts_list.append(smpl_verts)

            smpl_verts_all = np.concatenate(smpl_verts_list, axis=0)
            v_max = smpl_verts_all.max(axis=0)
            v_min = smpl_verts_all.min(axis=0)
            if args.source == 'hi4d':
                if idx == 0:
                    normalize_shift = -(v_max + v_min) / 2.
                    normalize_shift_first_frame = normalize_shift
                else:
                    normalize_shift = normalize_shift_first_frame
            else:
                normalize_shift = -(v_max + v_min) / 2.
            smpl_trans = np.stack(smpl_trans_list, axis=0)
            smpl_pose = np.stack(smpl_pose_list, axis=0)
            trans = smpl_trans + normalize_shift.reshape(1, 3)

            smpl_verts_all = smpl_verts_all + normalize_shift.reshape(1, 3)
            max_human_sphere = np.linalg.norm(smpl_verts_all, axis=1).max()
            if max_human_sphere > max_human_sphere_all:
                max_human_sphere_all = max_human_sphere
            
            target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

            P = K @ target_extrinsic
            output_trans.append(trans)
            output_pose.append(smpl_pose)
            output_P[f"cam_{idx}"] = P

    if args.mode == 'refine':
        mean_shape = np.array(mean_shape)
        np.save(f'{DIR}/{seq}/mean_shape.npy', mean_shape.mean(0))
    if args.mode == 'final':
        np.save(os.path.join(save_dir, 'gender.npy'), np.array(['neutral' for _ in range(number_person)]))
        np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))
        np.save(os.path.join(save_dir, 'mean_shape.npy'), smpl_shape)
        np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
        np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)
        np.save(os.path.join(save_dir, "max_human_sphere.npy"), np.array(max_human_sphere_all))
        print("max_human_sphere_all: ", max_human_sphere_all)
        print('output_pose', np.array(output_pose).shape)
        print('mean_shape', smpl_shape.shape)
        print('normalize_trans', np.array(output_trans).shape)

        smpl_shape = smpl_shape[1]
        smpl_pose = smpl_pose[1]
        trans = trans[1]
        smpl_output = smpl_model_list[1](betas=torch.tensor(smpl_shape)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_pose[3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_pose[:3])[None].cuda().float(),
                                         transl=torch.tensor(trans)[None].cuda().float())
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        _ = trimesh.Trimesh(smpl_verts).export(os.path.join(save_dir, 'test1.ply'))

        for j in range(0, smpl_verts.shape[0]):
            padded_v = np.pad(smpl_verts[j], (0, 1), 'constant', constant_values=(0, 1))
            temp = P @ padded_v.T 
            pix = (temp / temp[2])[:2]
            output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(save_dir, 'test1.png'), output_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    # video source
    parser.add_argument('--source', type=str, default='custom', help="custom video or dataset video")
    # sequence name
    parser.add_argument('--seq', type=str)
    # mode
    parser.add_argument('--mode', type=str, help="mask mode or refine mode: mask or refine or final")
    # scale factor for the input image
    parser.add_argument('--scale_factor', type=int, default=1, help="scale factor for the input image")

    parser.add_argument('--vitpose', action='store_true', help="use vitpose", default=False)
    parser.add_argument('--openpose', action='store_true', help="use openpose", default=False)
    args = parser.parse_args()
    main(args)