import itertools
import numpy as np
import open3d as o3d
import img
import math
import pymeshlab as mlab
from openni import openni2

def get_o3d_imgs():
    """
    This module saves a colored or depth frame from a openni device into {script directory}/images/{colored or depth}.
    This module requires python 3.10.0 or above, opencv 4.8.0 or above and openni2 2.2.0.33 Beta or above and open3d.
    Use get_colored_png(name:str)->None to save colored frame.
    Use get_depth_png(name:str)->None to save colored frame.
    """
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    depth_stream.start()
    color_stream.start()
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()
    depth_frame_buffer = depth_frame.get_buffer_as_uint16()
    color_frame_buffer = color_frame.get_buffer_as_uint8()
    depth_array = np.frombuffer(depth_frame_buffer, dtype=np.uint16)
    color_array = np.frombuffer(color_frame_buffer, dtype=np.uint8)
    depth_frame_image = depth_array.reshape((depth_frame.height, depth_frame.width, 1))
    color_frame_image = color_array.reshape((color_frame.height, color_frame.width, 3))  
    depth_o3d_image = o3d.geometry.Image(depth_frame_image)
    colored_o3d_image = o3d.geometry.Image(color_frame_image)  
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    return colored_o3d_image,depth_o3d_image

def get_pcd_from_o3d_colored_and_depth_images(colored_o3d_image,depth_o3d_image):
    """ Generates pcd from color and depth images.
        Takes rgb and depth images' paths.
        Return point cloud data.
    """
    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(colored_o3d_image, depth_o3d_image, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
    # Calibrate the cam
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 571.26,571.26, 320, 340)
    intrinsic.intrinsic_matrix = [[571.26, 0, 320], [0, 571.26, 340], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])

    # Generate pcd from rgbd image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])#Flip it otherwise pcd will be upside down

    o3d.visualization.draw_geometries([pcd])
    return pcd

def apply_pass_through_filter(pcd,min_x= -math.inf, min_y = -math.inf, min_z= -math.inf, max_x=math.inf, max_y=math.inf, max_z=math.inf):
    # Create bounding box:
    bounds = [[min_x, max_x], [min_y, max_y], [min_z, max_z]]  # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
    # Crop the point cloud using the bounding box:
    pcd_filtered = pcd.crop(bounding_box)
    o3d.visualization.draw_geometries([pcd_filtered])
    return pcd_filtered

def get_mesh_from_pcd(pcd):
    
    # Calculate normals for the point cloud
    pcd.estimate_normals()
    #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    o3d.io.write_triangle_mesh("dataset10.ply", mesh)
    o3d.visualization.draw_geometries([mesh])

def convertMesh():
    colored_img,depth_img = get_o3d_imgs()

    pcd= get_pcd_from_o3d_colored_and_depth_images(colored_img, depth_img)

    mesh = get_mesh_from_pcd(pcd)
