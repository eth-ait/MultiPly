# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import glob
from tqdm import tqdm
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def get_mask_center(mask_path):
    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    center = where.mean(axis=1)[::-1]

    indices = np.argwhere(mask)

    # Get the minimum and maximum x and y coordinates
    x_min = np.min(indices[:, 1])
    y_min = np.min(indices[:, 0])
    x_max = np.max(indices[:, 1])
    y_max = np.max(indices[:, 0])
    w = x_max - x_min
    h = y_max - y_min
    # x_min = x_min - w * 0.03
    # y_min = y_min - h * 0.03
    # x_max = x_max + w * 0.03
    # y_max = y_max + h * 0.03
    # x_min = np.min(indices[:, 1])-20
    # y_min = np.min(indices[:, 0])
    # x_max = np.max(indices[:, 1])+50
    # y_max = np.max(indices[:, 0])+30
    
    # center bounding box with center
    x_min = center[0] - w * 0.5
    y_min = center[1] - h * 0.5
    x_max = center[0] + w * 0.5
    y_max = center[1] + h * 0.5
    

    # Convert the coordinates to xyxy format
    bounding_box = {'bbox': np.array([x_min, y_min, x_max, y_max, 1.0])}
    diag = np.linalg.norm(np.array([x_max, y_max]) - np.array([x_min, y_min]))
    return center, bounding_box, diag

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('--det_config', help='Config file for detection', default='/media/ubuntu/hdd/ViTPose/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py')
    # parser.add_argument('--det_checkpoint', help='Checkpoint file for detection', default='/media/ubuntu/hdd/ViTPose/ViTPose/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth')
    parser.add_argument('--pose_config', help='Config file for pose', default='./vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py')
    parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose', default='./vitpose/checkpoints/vitpose-h-multi-coco.pth')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    # parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.7, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    # assert args.det_config is not None
    # assert args.det_checkpoint is not None

    # det_model = init_detector(
    #     args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    # img_paths = sorted(glob.glob(os.path.join(args.img_root, '*.png')))
    # if len(img_paths) == 0:
    #     img_paths = sorted(glob.glob(os.path.join(args.img_root, '*.jpg')))
    
    # img_dir = f'{DIR}/{args.seq}/frames'
    img_dir = args.img_root
    # this line below will lead to a segfault
    imagePaths = sorted(glob.glob(f'{img_dir}/*.png'))
    if len(imagePaths) == 0:
        imagePaths = sorted(glob.glob(f'{img_dir}/*.jpg'))
    # imagePaths = sorted(glob.glob(f'{img_dir}/*.png'))
    # imagePaths = op.get_images_on_directory(img_dir)
    maskPath_list = sorted(glob.glob(f'{img_dir}/../init_mask/*'))
    number_person = len(maskPath_list)
    mask_path_list = [sorted(glob.glob(f'{img_dir}/../init_mask/{i}/*.png')) for i in range(number_person)]
    if not os.path.exists(f'{img_dir}/../vitpose'):
            os.makedirs(f'{img_dir}/../vitpose')
    
    for idx, image_name in enumerate(tqdm(imagePaths)):

        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        # image_name = os.path.join(args.img_root, args.img)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        # mmdet_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes.
        # person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        mask_center_list = []
        bbox_list = []
        diag_list = []
        for person_id in range(number_person):
            maskPath = mask_path_list[person_id][idx]
            center, bbox, diag = get_mask_center(maskPath)
            mask_center_list.append(center)
            bbox_list.append(bbox)
            diag_list.append(diag)
        mask_center_np = np.stack(mask_center_list, 0)

        # import pdb; pdb.set_trace()
        # TODO if fully occluded, then should skip it or use previous frame
        person_results = bbox_list
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{image_name}')

        poseKeypoints = []
        for single_pose_result in pose_results:
            poseKeypoints_i = single_pose_result['keypoints']
            poseKeypoints.append(poseKeypoints_i)
        poseKeypoints = np.stack(poseKeypoints, 0)

        center_2D = (poseKeypoints[:, :, :2] * poseKeypoints[:, :, [-1]]).sum(axis=1) / poseKeypoints[:, :, -1].sum(axis=1, keepdims=True)
        # [person, 1]
        center_2D_weight = poseKeypoints[:, :, -1].sum(axis=1, keepdims=True)
        is_remain = [True for _ in range(center_2D.shape[0])]
        # start NMS
        for person_i in range(center_2D.shape[0]):
            for person_j in range(center_2D.shape[0]):
                if person_i == person_j:
                    continue
                # TODO, using keypoint distance instand of center distance
                keypoint_distance = (np.linalg.norm(poseKeypoints[person_i,:,:2] - poseKeypoints[person_j,:,:2], axis=1) * poseKeypoints[person_i,:,2:3] * poseKeypoints[person_j,:,2:3]).mean()
                if np.linalg.norm(center_2D[person_i] - center_2D[person_j]) + keypoint_distance < 50:
                    if center_2D_weight[person_i] > center_2D_weight[person_j]:
                        is_remain[person_j] = False
                    else:
                        is_remain[person_i] = False
        center_2D = center_2D[is_remain]
        poseKeypoints = poseKeypoints[is_remain]
        # nbrs = NearestNeighbors(n_neighbors=1)
        # nbrs.fit(center_2D)
        # distances, indices = nbrs.kneighbors(center_2D)
        # print(distances)
        # print(indi

        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]
        # skeleton-based NMS
        if poseKeypoints.shape[0] > 1:
            for person_i in range(poseKeypoints.shape[0]):
                for person_j in range(poseKeypoints.shape[0]):
                    if person_i == person_j:
                        continue
                    for skeleton_i in skeleton:
                        # if poseKeypoints[person_i, skeleton_i[0], -1] > 0.5 and poseKeypoints[person_j, skeleton_i[1], -1] > 0.5:
                        d1 = np.linalg.norm(poseKeypoints[person_i, skeleton_i[0], :2] - poseKeypoints[person_j, skeleton_i[0], :2])
                        d2 = np.linalg.norm(poseKeypoints[person_i, skeleton_i[1], :2] - poseKeypoints[person_j, skeleton_i[1], :2])
                        ratio = (d1+d2) / (diag_list[person_i] + diag_list[person_j])
                        weight1 = poseKeypoints[person_i, skeleton_i[0], -1] + poseKeypoints[person_i, skeleton_i[1], -1]
                        weight2 = poseKeypoints[person_j, skeleton_i[0], -1] + poseKeypoints[person_j, skeleton_i[1], -1]
                        if ratio < 0.02:
                            if weight1 > weight2:
                                # is_remain[person_j] = False
                                poseKeypoints[person_j, skeleton_i[0], :] = 0.0
                                poseKeypoints[person_j, skeleton_i[1], :] = 0.0
                            else:
                                # is_remain[person_i] = False
                                poseKeypoints[person_i, skeleton_i[0], :] = 0.0
                                poseKeypoints[person_i, skeleton_i[1], :] = 0.0

        if idx == 0:
            last_output_2d = np.zeros((number_person, poseKeypoints.shape[1], 3))
        # calculate cost matrix
        cost_matrix = np.zeros((number_person, center_2D.shape[0]))
        for person_i in range(number_person):
            for i in range(center_2D.shape[0]):
                cost_matrix[person_i, i] = np.linalg.norm(center_2D[i] - mask_center_np[person_i]) # 3D
                # TODO: tracking by J2D
                # cost_matrix[person_i, i] = cost_matrix[person_i, i] + (np.linalg.norm(poseKeypoints[i,:,:2] - last_output_2d[person_i,:,:2], axis=1) * poseKeypoints[i,:,2:3] * last_output_2d[person_i,:,2:3]).mean() # 2D
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # print(cost_matrix)
        output_2d = np.zeros((number_person, poseKeypoints.shape[1], 3))
        for person_i, keypoint_i in zip(row_ind, col_ind):
            if cost_matrix[person_i, keypoint_i] < 200:
            # if cost_matrix[person_i, keypoint_i] < 900:
                output_2d[person_i] = poseKeypoints[keypoint_i]

        last_output_2d = output_2d
        np.save(f'{img_dir}/../vitpose/%04d.npy' % idx, output_2d)
        # output_img = datum.cvOutputData
        # output_img = cv2.imread(image_name)
        # show the results
        output_img = vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

        color = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255,255,0), (255,0,255), (0,255,255), (255,255,255), (0,0,0), (128,128,128), (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
        for color_i, (person_i, keypoint_i) in enumerate(zip(row_ind, col_ind)):
            color_jit = color[person_i]
            # poseKeypoints_0 = poseKeypoints[keypoint_i]
            poseKeypoints_0 = output_2d[person_i]
            for point in poseKeypoints_0:
                if point[-1] > args.kpt_thr:
                    cv2.circle(output_img, (int(point[0]), int(point[1])), 5, color_jit, -1)
            cv2.circle(output_img, (int(center_2D[keypoint_i][0]),int(center_2D[keypoint_i][1])), 10, color_jit, -1)
            cv2.circle(output_img, (int(mask_center_np[person_i][0]),int(mask_center_np[person_i][1])), 10, color_jit, -1)
            
        cv2.imwrite(f'{img_dir}/../vitpose/%04d.png' % idx, output_img)



if __name__ == '__main__':
    main()
