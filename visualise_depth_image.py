import numpy as np 
import seaborn as sns 
import numpy.typing as npt
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors


def visualise_depth_image(depth_image: npt.NDArray[np.float32], rgb_image: npt.NDArray[np.uint8] = None):
    '''
    Allows for easy visualisation of the depth image 
    Inputs:
    - depth_image: Depth image of size [HxW]
    - rgb_image (optional): RGB image of size [HxWx3].  If provided, will visualise the depth overlaid on the RGB image
    Returns:
    - depth visualisation image: RGB image of size [HxWx3], with depth visualised either by itself or superimposed over
      the corresponding RGB image 
    '''
    depth_image = depth_image / 1000. # Convert to metres
    no_depth_measurement_mask = depth_image == 0
    
    cmap = mpl.colormaps.get_cmap('gist_ncar_r')
    cmap.set_bad('black')
    norm = mcolors.Normalize(vmin=0., vmax=60.)
    
    norm_depth_image = norm(depth_image)
    depth_colorised = (cmap(norm_depth_image)[..., :3] * 255).astype(np.uint8)
    depth_colorised[no_depth_measurement_mask] = 0
    
    if rgb_image is not None:
        depth_colorised[no_depth_measurement_mask] = rgb_image[no_depth_measurement_mask] // 2
    
    return depth_colorised
