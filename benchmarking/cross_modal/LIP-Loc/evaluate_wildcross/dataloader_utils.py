import numpy as np 
from scipy.spatial.transform import Rotation as R

def get_extrinsic_transform(config):
    # with open(config_filepath, 'r') as f:
    #     config = yaml.safe_load(f)

    trans = config['centre-camera']['extrinsics']['translation']
    rot = R.from_quat(config['centre-camera']['extrinsics']['rotation']) 

    T_B_CAM = np.eye(4)
    T_B_CAM[:3,:3] = rot.as_matrix()
    T_B_CAM[:3,3] = np.array(trans)
    
    # print("Loaded T_B_CAM")
    # print(T_B_CAM)
    # print(np.linalg.inv(T_B_CAM))

    return T_B_CAM

def transform_image_trajectory(coords, config):
    T_B_CAM = get_extrinsic_transform(config)
    
    # Apply rotation
    coords = R.from_matrix(T_B_CAM[:3,:3]).apply(coords, inverse=True)
    # Apply translation
    coords = coords + T_B_CAM[:3,3]
    return coords 