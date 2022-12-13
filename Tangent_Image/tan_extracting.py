from tan_list import tan_list
import os
import cv2
import numpy as np
from tqdm import tqdm


def N(image):
        from scipy.ndimage.filters import maximum_filter
        M = 8
        image = cv2.convertScaleAbs(image, alpha=M/image.max(), beta=0.)
        w,h = image.shape
        maxima = maximum_filter(image, size=(w/10,h/1))
        maxima = (image == maxima)
        mnum = maxima.sum()
        maxima = np.multiply(maxima, image)
        mbar = float(maxima.sum()) / mnum
        return image * (M-mbar)**2

def normalize_map(s_map):
        norm_s_map = (s_map - np.min(s_map)) / (s_map.max() - s_map.min())
        return norm_s_map 

def tangent_image(config, scenes):
    tan_scenes=[]
    source_emb_list=find_source_emb(scenes)

    for d_img_name in tqdm(scenes):
        d_img = cv2.imread(os.path.join(config.source_path, d_img_name))
        d_img=cv2.resize(d_img,(1024,768))


        s_map = cv2.imread(os.path.join(config.smap_path, d_img_name), cv2.IMREAD_GRAYSCALE)
        s_map=normalize_map(N(s_map))
        s_map=cv2.resize(s_map,(1024,768))

        score=scenes[d_img_name]['score']

        extracted_tans_names = tan_list(d_img_name, d_img, config.target_path, s_map,config, score, source_emb_list[d_img_name])
        tan_scenes.extend(extracted_tans_names)

    return tan_scenes

def find_source_emb(scenes):
    source_emb= np.random.rand(len(scenes))
    # source_emb_list is a dictionary: {image name: its source embedding}
    source_emb_list={}

    for img_idx, img_name in enumerate(scenes):
        source_emb_list[img_name]=source_emb[img_idx]

    return source_emb_list
