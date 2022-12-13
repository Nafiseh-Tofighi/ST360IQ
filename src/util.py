import torch
import numpy as np
import re
    

class RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img = sample['d_img']
        score = sample['score']
        center=sample['center']
        source_emb=sample['source_emb']

        prob_lr = np.random.random()
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()

        sample = {
            'd_img': d_img,
            'center':center,
            'source_emb':source_emb,
            'score': score
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):

        d_img = sample['d_img']
        score = sample['score']
        center=sample['center']
        source_emb=sample['source_emb']

        d_img[:, :, 0] = (d_img[:, :, 0] - self.mean[0]) / self.var[0]
        d_img[:, :, 1] = (d_img[:, :, 1] - self.mean[1]) / self.var[1]
        d_img[:, :, 2] = (d_img[:, :, 2] - self.mean[2]) / self.var[2]
        
        sample = {
            'd_img': d_img,
            'center':center,
            'source_emb':source_emb, 
            'score': score
        }
        return sample


class ToTensor(object):
    def __call__(self, sample):

        d_img = sample['d_img']
        score = np.array(sample['score'])
        center=sample['center']
        source_emb=sample['source_emb']

        d_img = np.transpose(d_img, (2, 0, 1))
        d_img = torch.from_numpy(d_img)

        sample = {
            'd_img': d_img,
            'center':center, 
            'source_emb':source_emb,
            'score': score
        }
        return sample



def RandShuffle(config):
        
    if config.split_avail:

        seed = np.random.random()
        random_seed = int(seed*10)
        np.random.seed(random_seed)

        with open(config.train_list, 'r') as the_file:
            train_list=the_file.read().split('\n')


        with open(config.test_list, 'r') as the_file:
            test_list=the_file.read().split('\n')

        with open(config.txt_file_tans, 'r') as the_file:
            data_list=the_file.read().split('\n')

        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

        train_scenes=[img for img in data_list if re.sub(r'-\d+'+'.pt', '', img) in train_list[:int(0.8*len(train_list))]]
        val_scenes=[img for img in data_list if re.sub(r'-\d+'+'.pt', '', img) in train_list[int(0.8*len(train_list)):]]                            
        test_scenes=[img for img in data_list if re.sub(r'-\d+'+'.pt', '', img) in test_list]

    else:
        with open(config.txt_file_tans, 'r') as the_file:
            data_list=the_file.read().split('\n')

        np.random.shuffle(data_list)
        train_scenes=[img for img in data_list[:int(0.6*len(data_list))] ]
        val_scenes=[img for img in data_list[int(0.6*len(data_list)):int(0.8*len(data_list))] ]                   
        test_scenes=[img for img in data_list[int(0.8*len(data_list)):] ]

    
    return train_scenes, val_scenes, test_scenes