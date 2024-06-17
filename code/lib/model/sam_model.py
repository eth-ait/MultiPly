import hydra
import numpy as np
import torch
# import matplotlib.pyplot as plt
import cv2
import glob
import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
# # for visualization
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
#                linewidth=0.5)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
#                linewidth=0.5)


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class SAMServer():
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)
        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        self.training_indices = list(range(opt.start_frame, opt.end_frame, 1))
        self.img_paths = [self.img_paths[i] for i in self.training_indices]

        model_type = "vit_h"
        checkpoint_path = "./outputs/sam_vit_h_4b8939.pth"
        # model_type = "vit_b"
        # checkpoint_path = "./outputs/sam_vit_b_01ec64.pth"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=hydra.utils.to_absolute_path(checkpoint_path))
        sam.to(device=device)
        self.sam = sam
        self.opt = opt
        self.predictor = SamPredictor(sam)

    def get_sam_mask(self, current_epoch):
        np.random.seed(42)
        smpl_mask = np.load(f'stage_instance_mask/{current_epoch:05d}/all_person_smpl_mask.npy')
        smpl_joint = np.load(f'stage_instance_mask/{current_epoch:05d}/2d_keypoint.npy')
        output_mask_list = []
        for i, img_path in enumerate(tqdm(self.img_paths)):
            # i = i+109
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
            image_mask_all = smpl_mask[i]
            output_mask_per_frame = []
            for person_id in range(image_mask_all.shape[0]):
                image_mask = image_mask_all[person_id]
                negative_image_mask_list = []
                for neg_person_i in range(image_mask_all.shape[0]):
                    if neg_person_i != person_id:
                        negative_image_mask_list.append(image_mask_all[neg_person_i])
                negative_image_mask = np.stack(negative_image_mask_list, axis=0)
                # fusion the negative mask
                negative_image_mask = np.max(negative_image_mask, axis=0)
                # get the xyxy bounding box from the binary mask
                indices = np.argwhere(image_mask)

                # Get the minimum and maximum x and y coordinates
                x_min = np.min(indices[:, 1])
                y_min = np.min(indices[:, 0])
                x_max = np.max(indices[:, 1])
                y_max = np.max(indices[:, 0])

                # expand the bounding box by 6%
                x_min = max(0, x_min - int(0.03 * (x_max - x_min)))
                y_min = max(0, y_min - int(0.03 * (y_max - y_min)))
                x_max = min(image_mask.shape[1], x_max + int(0.03 * (x_max - x_min)))
                y_max = min(image_mask.shape[0], y_max + int(0.03 * (y_max - y_min)))

                # Convert the coordinates to xyxy format
                bounding_box = np.array([x_min, y_min, x_max, y_max])

                # Get dimensions of the mask image
                height, width = image_mask.shape

                # Find the max dimension
                max_dim = max(height, width)

                # Create a black canvas with the max dimension
                canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)

                # Copy the mask image onto the canvas
                if height > width:
                    canvas[0:height, 0:width] = image_mask
                else:
                    canvas[0:height, max_dim - width:max_dim] = image_mask

                resized_mask = cv2.resize(canvas, (256, 256))

                positive_point_candidate = smpl_joint[i, person_id, :27]
                negative_point_candidate_list = []
                for neg_person_i in range(image_mask_all.shape[0]):
                    if neg_person_i != person_id:
                        negative_point_candidate_list.append(smpl_joint[i, neg_person_i, :27])
                negative_point_candidate = np.concatenate(negative_point_candidate_list, axis=0)
                point_list = []
                for j in range(positive_point_candidate.shape[0]):
                    p = positive_point_candidate[j]
                    try:
                        if image_mask[p[1], p[0]] > 0.7:
                            point_list.append(p)
                    except:
                        pass
                        # print('positive point out of image')
                point_list = np.array(point_list)

                positive_points = point_list
                # Load mask image
                mask = image_mask

                binary_mask = mask

                if len(positive_points) == 0:
                    positive_point_list = []
                    print("all keypoints are out of the mask, sample one point randomly")
                    max_try_time = 10000000
                    try_time = 0
                    while len(positive_points) < 1 and try_time < max_try_time:
                        x = np.random.randint(0, mask.shape[1])
                        y = np.random.randint(0, mask.shape[0])
                        try_time += 1
                        if image_mask[y, x] > 0.7:
                            positive_point_list.append([x, y])
                            positive_points = np.array(positive_point_list)
                            break
                    if len(positive_points) == 0:
                        print("ERROR: sample point failed, use random keypoint")
                        positive_point_list.append(positive_point_candidate[-1])
                        positive_points = np.array(positive_point_list)
                num_positive_points = len(positive_points)
                positive_labels = np.ones(num_positive_points)

                # Sample a small number of positive points
                num_samples = num_positive_points
                sampled_positive_indices = np.random.choice(num_positive_points, num_samples, replace=False)
                sampled_positive_points = positive_points[sampled_positive_indices]
                sampled_positive_labels = positive_labels[sampled_positive_indices]

                num_negative_points = 10
                negative_points = []
                while len(negative_points) < num_negative_points:
                    x = np.random.randint(0, mask.shape[1])
                    y = np.random.randint(0, mask.shape[0])
                    if binary_mask[y, x] == 0:
                        negative_points.append([x, y])

                for j in range(negative_point_candidate.shape[0]):
                    p = negative_point_candidate[j]
                    try:
                        if image_mask[p[1], p[0]] < 0.7 and negative_image_mask[p[1], p[0]] > 0.7:
                            negative_points.append([p[0], p[1]])
                    except:
                        pass
                        # print('negative point outside image')

                negative_labels = np.zeros(len(negative_points))

                # Convert negative points to a numpy array
                negative_points = np.array(negative_points)

                sampled_points = np.concatenate((sampled_positive_points, negative_points), axis=0)
                sampled_labels = np.concatenate((sampled_positive_labels, negative_labels))

                input_point = sampled_points
                input_label = sampled_labels

                resized_mask_logit = torch.special.logit(torch.from_numpy(resized_mask), eps=1e-6)
                resized_mask_logit = resized_mask_logit.numpy()
                masks, _, mask_logit = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=(resized_mask_logit[None, :, :]),
                    box=bounding_box[None, :],
                    multimask_output=False,
                    return_logits=True,
                )

                masks, _, mask_logit = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=mask_logit,
                    box=bounding_box[None, :],
                    multimask_output=False,
                    return_logits=True,
                )

                masks, _, mask_logit = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=mask_logit,
                    box=bounding_box[None, :],
                    multimask_output=False,
                    return_logits=True,
                )


                if not os.path.exists(f'stage_sam_mask/{current_epoch:05d}/{person_id}'):
                    os.makedirs(f'stage_sam_mask/{current_epoch:05d}/{person_id}')  

                # # for visualization
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image)
                # show_mask(masks > self.predictor.model.mask_threshold, plt.gca())
                # show_box(bounding_box, plt.gca())
                # show_points(input_point, input_label, plt.gca(), marker_size=30)
                # plt.axis('off')
                # if current_epoch % 200 == 0:
                #     plt.savefig(os.path.join(f'stage_sam_mask/{current_epoch:05d}/{person_id}', '%04d.png' % i), bbox_inches='tight', pad_inches=0.0)
                # plt.close()

                output_mask_per_frame.append(masks)
            output_mask_per_frame = np.concatenate(output_mask_per_frame, axis=0)
            output_mask_list.append(output_mask_per_frame)
        output = np.stack(output_mask_list, axis=0)
        np.save(f"stage_sam_mask/{current_epoch:05d}/sam_opt_mask.npy", output)
        print("sam mask output shape", output.shape)