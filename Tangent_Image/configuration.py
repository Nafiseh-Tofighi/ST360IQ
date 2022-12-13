import os
import shutil
import numpy as np

from config import Config
from tan_extracting import tangent_image

# config file
config = Config({ 

    'source_path': '',                          # direction to the dataset
    'target_path':'',                           # direction to save output tangent images
    'smap_path':'',                             # path of saliency maps

    'txt_file_name':'',                         # list of images in the database and their IQA scores

    'n_tan_patch':16,                           # total number of tangen images extracted from a single image
    'sampling_method': 'top',                   # [top, stochastic, random] ---> selecting tan patches method
    'sampling_region_size': [128, 128],         # size of regions in ERP image that their saliency scores are 
                                                # calculated and the tangent images will generate from centers 
                                                # of these regiones
    'tan_stride':64,                            # stride of sampling regions in ERP image
    'fov':(256, 256)
    })



if not os.path.exists(config.target_path):
    os.makedirs(config.target_path)
else:
    shutil.rmtree(config.target_path)
    os.makedirs(config.target_path)


seed = np.random.random()
random_seed = int(seed*10)
np.random.seed(random_seed)

with open(config.txt_file_name, 'r') as the_file:
    img_list=the_file.read().split('\n')

np.random.shuffle(img_list)

img_scenes={img.split()[0]:{'score':img.split()[1]} 
                            for img in img_list }

tan_scenes= tangent_image(config, img_scenes)

np.savetxt(os.path.join(config.target_path, 'tan_imgs_list.txt'), tan_scenes, fmt='%s')

# Removing empty lines
with open(os.path.join(config.target_path, 'tan_imgs_list.txt')) as f_input:
    data = f_input.read().rstrip('\n')

with open(os.path.join(config.target_path, 'tan_imgs_list.txt'), 'w') as f_output:    
    f_output.write(data)