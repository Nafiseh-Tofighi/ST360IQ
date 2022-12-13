import os
import torch
import numpy as np
import cv2
from patchify import patchify, unpatchify
from random import randint

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, config, transform, scene_list):
        super(IQADataset, self).__init__()
        self.config=config
        self.transform = transform
        self.scene_list = scene_list
        self.n_enc_seq=config.n_enc_seq
        self.patch_size=config.patch_size
        self.tan_data_path=config.tan_data_path

        self.data_dict = IQADatalist(
            scene_list = self.scene_list,
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):

        d_img_name = self.data_dict['d_img_list'][idx]
        data=torch.load(os.path.join(self.tan_data_path, d_img_name))
        center=data['center']
        source_emb=data['source_emb']
        score=float(data['score'])
        d_img=data['d_img_org']

        h, w=self.n_enc_seq[0]*self.patch_size, self.n_enc_seq[1]*self.patch_size
        d_img=cv2.resize(d_img,(h,w))

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255

        sample = { 
            'd_img': d_img,
            'center':center,
            'source_emb':source_emb,
            'score': score
            }

        if self.transform: 
            sample = self.transform(sample)
        return sample

class IQADatalist():
        def __init__(self, scene_list):
            self.scene_list = scene_list
    
        def load_data_dict(self):
            d_img_list=[img for img in self.scene_list] 

            return {'d_img_list': d_img_list}
