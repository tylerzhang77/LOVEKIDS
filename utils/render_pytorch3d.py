import numpy as np
import pytorch3d
import pytorch3d.renderer
import torch
from scipy.spatial.transform import Rotation



def render_mesh_batch(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    vertices = vertices + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    faces = faces.expand(bs, *faces.shape).to(device)

    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=None,  # 在每次迭代中动态设置相机参数
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=None,  # 在每次迭代中动态设置相机参数
            lights=lights,
            materials=materials
        )
    )

    # 创建与批量大小匹配的空矩阵
    rendered_images = torch.zeros((bs, height, width, 4), device=device)

    for i in range(bs):
        # 获取当前批次的数据
        current_vertices = torch.unsqueeze(vertices[i], dim=0)
        current_faces = torch.unsqueeze(faces[i], dim=0)

        current_mesh = pytorch3d.structures.Meshes(verts=current_vertices, faces=current_faces, textures=textures[i])
        
        # 初始化相机
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=((2 * focal_length[i] / min(height, width), 2 * focal_length[i] / min(height, width)),),
            device=device,
        )
        # 在渲染器中设置当前批次的相机参数
        renderer.rasterizer.cameras = cameras
        renderer.shader.cameras = cameras

        # Do rendering
        img = renderer(current_mesh)
        rendered_images[i] = img
    return rendered_images

def render_mesh(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]  # batch_size

    # add the translation 平移变换
    vertices = vertices + translation[:, None, :]  

    # upside down the mesh 旋转变换
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
  
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3) # (3,3) —— (bs,3,3)
    faces = faces.expand(bs, *faces.shape).to(device)
    
    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)  # (bs,1,3)-(bs,3,1)-(bs,1,3)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3) 
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization栅格化 and shading着色.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,  # 模糊半径
        faces_per_pixel=1,  # 每个pixel使用的face数目
        bin_size=0
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((0.9, 0.7, 0.7),),  # 环境光
        diffuse_color=((1.0, 1.0, 1.0),),  # 漫反射光
        specular_color=((0.0, 0.0, 0.0),),  # 镜面反射光
        shininess=64,  # 镜面反射的粗糙度
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs


def render_mesh_single_frame(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    assert vertices.shape[0] == 1
    vertices = vertices[0]
    translation = translation[0]

    # add the translation
    vertices = vertices + translation

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device)
    faces = faces.to(device)

    vertices = torch.matmul(rot, vertices.T).T

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices], faces=[faces], textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width),
                      2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        # image_size=(height, width),   # (H, W)
        image_size=height,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(
        device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs[0]


def render_mesh_video(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]  # batch_size

    # add the translation 平移变换
    vertices = vertices + translation[:, None, :]  
    # upside down the mesh 旋转变换
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
  
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3) # (3,3) —— (bs,3,3)
    faces = faces.expand(bs, *faces.shape).to(device)
    
    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)  # (bs,1,3)-(bs,3,1)-(bs,1,3)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3) 
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization栅格化 and shading着色.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,  # 模糊半径
        faces_per_pixel=1,  # 每个pixel使用的face数目
        bin_size=0
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((0.9, 0.7, 0.7),),  # 环境光
        diffuse_color=((1.0, 1.0, 1.0),),  # 漫反射光
        specular_color=((0.0, 0.0, 0.0),),  # 镜面反射光
        shininess=64,  # 镜面反射的粗糙度
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs