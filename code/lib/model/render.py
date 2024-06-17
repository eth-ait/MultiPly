from pytorch3d.renderer import (
    PerspectiveCameras,
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    SoftSilhouetteShader,
    AmbientLights,
)
from pytorch3d.structures import Meshes
import numpy as np
import torch
import trimesh
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures.meshes import join_meshes_as_scene
import cv2
import os
from tqdm import tqdm
import glob
import pickle as pkl
from tqdm import tqdm

class Renderer():

    def __init__(self, focal_length=None, principal_point=None, img_size=None, cam_intrinsic=None):

        super().__init__()
        # img_size=[1080, 1920]
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        self.focal_length = torch.tensor([(cam_intrinsic[0, 0]).astype(np.float32),
                                          (cam_intrinsic[1, 1]).astype(np.float32)
                                          ]).to(self.device).unsqueeze(0)
        principal_point = [cam_intrinsic[0, 2], cam_intrinsic[1, 2]]
        self.principal_point = torch.tensor(principal_point).float().to(self.device).unsqueeze(0)

        self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                [0., -1., 0.],
                                                [0., 0., 1.]])).cuda().float().unsqueeze(0)

        self.cam_T = torch.zeros((1, 3)).cuda().float()

        self.cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=self.cam_R,
            T=self.cam_T,
            in_ndc=False,
            image_size=torch.tensor(self.image_size).unsqueeze(0),
            device=self.device)

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 0.0]], ambient_color=((1, 1, 1),),
                                  diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
        # self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        # self.raster_settings = RasterizationSettings(image_size=self.render_img_size, faces_per_pixel=10, blur_radius=0,
        #                                              max_faces_per_bin=90000)
        self.raster_settings = RasterizationSettings(image_size=(int(self.image_size[0]),int(self.image_size[1])), faces_per_pixel=10, blur_radius=0,
                                                     max_faces_per_bin=90000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)


    def set_camera(self, R, T):
        cam_R = R.clone()
        cam_T = T.clone()
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :2] *= -1.0
        cam_R = torch.transpose(cam_R, 1, 2)
        self.cameras = PerspectiveCameras(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            R=cam_R,
            T=cam_T,
            in_ndc=False,
            image_size=torch.tensor(self.image_size).unsqueeze(0),
            device=self.device)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
        self.blend_params = BlendParams(sigma=5e-5, gamma=1e-4)
        self.soft_raster_settings = RasterizationSettings(
            image_size=(int(self.image_size[0]),int(self.image_size[1])),
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma,
            faces_per_pixel=100,
            max_faces_per_bin=90000,
        )
        self.ambient_light = AmbientLights(device=self.device, ambient_color=((1, 1, 1)))
        self.soft_shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.ambient_light)
        self.soft_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.soft_raster_settings
            ),
            shader=self.soft_shader
        )

    def render_multiple_meshes(self, verts_list, faces_list, verts_colors_list):
        '''
        verts_list: list of verts
        faces_list: list of faces
        verts_colors_list: list of verts_colors
        '''
        # with torch.no_grad():
        mesh_list = []
        for v,f,c in zip(verts_list, faces_list, verts_colors_list):
            mesh_list.append(Meshes(v, f, textures=Textures(verts_rgb=c)))
        scene = join_meshes_as_scene(mesh_list)
        image = self.renderer(scene)
        return image

    def softrender_multiple_meshes(self, verts_list, faces_list, verts_colors_list):
        '''
        verts_list: list of verts
        faces_list: list of faces
        verts_colors_list: list of verts_colors
        '''
        # with torch.no_grad():
        mesh_list = []
        for v,f,c in zip(verts_list, faces_list, verts_colors_list):
            mesh_list.append(Meshes(v, f, textures=Textures(verts_rgb=c)))
        scene = join_meshes_as_scene(mesh_list)
        image = self.soft_renderer(scene)
        return image

    def render_multiple_depth_map(self, verts_list, faces_list, verts_colors_list=None):
        '''
        verts_list: list of verts
        faces_list: list of faces
        verts_colors_list: list of verts_colors
        '''
        # with torch.no_grad():
        mesh_list = []
        depth_map_list = []
        if verts_colors_list is None:
            verts_colors_list = [None] * len(verts_list)
        for v,f,c in zip(verts_list, faces_list, verts_colors_list):
            if c is None:
                m = Meshes(v, f)
            else:
                m = Meshes(v, f, textures=Textures(verts_rgb=c))
            mesh_list.append(m)
            fragment = self.rasterizer(m)
            # import pdb;pdb.set_trace()
            depth_map_list.append(fragment.zbuf)
        # scene = join_meshes_as_scene(mesh_list)
        # image = self.renderer(scene)
        return depth_map_list




    def render_mesh_recon(self, verts, faces, R=None, T=None, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = -torch.tensor([0, 0, -1]).float().to(verts.device)
            shades = (normals * front_light.view(1, 1, 3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
            results = []
            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)
            # normal
            if 'n' in mode:
                # import pdb
                # pdb.set_trace()

                normals_vis = normals * 0.5 + 0.5  # -1*normals* 0.5 + 0.5
                normals_vis = normals_vis[:, :, [2, 1, 0]]
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # albedo
            if 'a' in mode:
                assert (colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)

            # albedo*shading
            if 't' in mode:
                assert (colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return torch.cat(results, axis=1)


def render_trimesh(renderer ,mesh, R, T, mode='np'):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, ..., :3] / 255
    renderer.set_camera(R, T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)

    return image