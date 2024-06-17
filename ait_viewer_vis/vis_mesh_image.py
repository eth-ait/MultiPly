import cv2
import glob
import json
import numpy as np
import os
import torch
import tqdm
import pickle as pkl
import shutil
import trimesh
from aitviewer.configuration import CONFIG as C

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from scipy.spatial.transform import Rotation as R
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material

def main_without_extrinsics(input_root, output_root, visualize_mesh):
    v = Viewer(size=(1920, 1080))
    v.playback_fps = 15.0

    if visualize_mesh:
        camPs = np.load(os.path.join(input_root, 'cameras.npz'))
        norm_camPs = np.load(os.path.join(input_root, 'cameras_normalize.npz'))
        scale = norm_camPs['scale_mat_0'].astype(np.float32)[0,0]
        image_files = sorted(glob.glob(os.path.join(input_root, "image/*.png")))
        print("number of image ",len(image_files))

        COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0],[0.412,1.0,0.663,1.0],[0.412,0.412,0.663,1.0],[0.412,0.0,0.0,1.0],[0.0,0.0,0.663,1.0],[0.0,0.412,0.0,1.0],[1.0,0.0,0.0,1.0]]
        mean_list = []
        number_person = len(sorted(glob.glob(os.path.join(f'{output_root}/test_mesh/*')))) - 1
        print("number_person ",number_person)
        extrinsics = []
        intrinsics = None
        for p in range(number_person):
            vertices = []
            faces = []
            vertex_normals = []
            uvs = []
            texture_paths = []
            num_frame = len(sorted(glob.glob(os.path.join(f'{output_root}/test_mesh/0/*')))) // 2
            print("number_frame ",num_frame)
            start_frame = 0
            end_frame = num_frame
            for idx in range(start_frame, end_frame):
                mesh = trimesh.load(os.path.join(f'{output_root}/test_mesh/{p}/{idx:04d}_deformed.ply'), process=False)
                if p == 0:
                    mean_list.append(mesh.vertices.mean(axis=0))
                mesh.vertices = mesh.vertices - mean_list[idx-start_frame]
                scaled_vertices = mesh.vertices * scale
                vertices.append(scaled_vertices)
                faces.append(mesh.faces)
                vertex_normals.append(mesh.vertex_normals)
                if p == 0:
                    out = cv2.decomposeProjectionMatrix(camPs[f'cam_{idx}'][:3, :])
                    # if idx == 0:
                    if idx == start_frame:
                        intrinsics = out[0]
                    render_R = out[1]
                    cam_center = out[2]
                    cam_center = (cam_center[:3] / cam_center[3])[:, 0]
                    cam_center = cam_center - (mean_list[idx-start_frame] * scale)
                    render_T = -render_R @ cam_center
                    ext = np.zeros((4, 4))
                    ext[:3, :3] = render_R
                    ext[:3, 3] = render_T
                    extrinsics.append(ext[np.newaxis])
            m=Material(color = COLORS[p], diffuse=1.0, ambient=0.0)
            meshes = VariableTopologyMeshes(vertices,
                                            faces,
                                            vertex_normals,
                                            preload=True,
                                            color = COLORS[p],
                                            name=f"mesh_{p}",
                                            material=m,
                                            )
            v.scene.add(meshes)
        
        image_files = image_files[start_frame: end_frame]
        extrinsics = np.concatenate(extrinsics, axis=0)[:, :3]
        img_temp = cv2.imread(image_files[0])
        cols, rows = img_temp.shape[1], img_temp.shape[0]
        cameras = OpenCVCamera(intrinsics, extrinsics, cols=cols, rows=rows, viewer=v)
        bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files)
        v.scene.add(cameras, bb)


    v.run()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default="../data/taichi01_vitpose_openpose", help='seq name')
    parser.add_argument('--output_root', type=str, default="../code/outputs/Hi4D/taichi01_sam_delay_depth_loop_2_MLP_vitpose_openpose", help='seq name')
    opt = parser.parse_args()
    input_root = opt.input_root
    output_root = opt.output_root
    main_without_extrinsics(input_root, output_root, visualize_mesh=True)