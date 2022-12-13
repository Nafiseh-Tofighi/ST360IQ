import torch
import torch.nn.functional as F
import numpy as np
import math
import torch
import numpy as np
from patchify import patchify
import random
import os

def tan_list(d_img_name, d_img, target_path, s_map,config, score, source_emb):
    pt_list=[]
    maxIdx, smapPatches_shape=find_max_index(s_map, config)
    tan_list, centers = find_tan_imgs(config, d_img, maxIdx, smapPatches_shape)

    for tan_idx, tan_img in enumerate(tan_list):
        data={'d_img_org': tan_img, 'score':score, 'center':centers[tan_idx], 'source_emb': source_emb}
        pt_name= f'{d_img_name}-{tan_idx}.pt'
        pt_list.append(pt_name)
        torch.save(data, os.path.join(target_path, pt_name))
    return pt_list



def find_tan_imgs(config, d_img, maxIdx, smapPatches_shape):
    tan_list,centers=[],[]
    H,W,_ = d_img.shape

    for idx in maxIdx:
        h=int(idx/smapPatches_shape[0])
        h=(h+1)*config.tan_stride

        w= idx%smapPatches_shape[0]
        w=(w+1)*config.tan_stride

        phi= ((h/H)*180)-90
        theta= -1*(((w/W)*360)-180)

        tan_img, center= get_tan_imgs(d_img, phi, theta, config.fov)

        tan_list.append(tan_img)
        centers.append(center)
    return tan_list, centers


def find_max_index(s_map, config):

    smapPatches=patchify(s_map, (config.sampling_region_size[0],config.sampling_region_size[1]),config.tan_stride)
    smapPatches_shape=smapPatches.shape

    flatSmapPathces=np.reshape(smapPatches,(smapPatches_shape[0],smapPatches_shape[1],smapPatches_shape[2]*smapPatches_shape[3]))


    meansSmap=flatSmapPathces.mean(axis=2 ,dtype=int)
    meansSmap=np.reshape(meansSmap,(smapPatches_shape[0]*smapPatches_shape[1]))

    maxIdx=[]
    if config.sampling_method=="stochastic":
        while len(maxIdx) < config.n_tan_patch:
            item = random.choices(np.arange(smapPatches_shape[0]*smapPatches_shape[1]), weights=meansSmap.flatten())
            if item not in maxIdx:
                maxIdx.append(item[0])
    elif config.sampling_method=="random":

        maxIdx=random.choices(np.arange(smapPatches_shape[0]*smapPatches_shape[1]),k=config.n_tan_patch)
    else:
        maxIdx = np.argsort(-meansSmap, axis=0)[0:config.n_tan_patch]

    return maxIdx, smapPatches_shape



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype = np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz

def equi2pers(erp_img, fov, patch_size,theta, phi):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)


    num_cols = np.full(1,1)
    phi_center = phi 

            
    phi_interval = 180 

    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 

            theta_center = theta

            center = [theta_center, phi_center]
            all_combos.append(center)
            up = phi_center + phi_interval / 2
            down = phi_center - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    center_np=cp.numpy()



    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)
  
    grid_tmp = torch.stack([lon, lat], -1)
    xyz = uv2xyz(grid_tmp)
    xyz = xyz.reshape(num_patch, height, width, 3).transpose(0, 3, 1, 2)
    xyz = torch.from_numpy(xyz).to(pers.device).contiguous()
    
    uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
    uv = uv.contiguous()

    return pers, xyz, uv, center_np[0]

def get_tan_imgs(img,phi,theta,fov=(256, 256)):

    patch_size=(1024, 1024)

    update_img = img.astype(np.float32) 
    update_img = np.transpose(update_img, [2, 0, 1])
    update_img = torch.from_numpy(update_img)
    update_img = update_img.unsqueeze(0)

    pers,_,_, centers = equi2pers(update_img, fov=fov, patch_size=patch_size,theta=theta, phi=phi)
    pers = pers[0].numpy()
    pers=np.squeeze(pers)

    tangent_pers = pers[:,:,:].transpose(1, 2, 0).astype(np.uint8)
    return tangent_pers.astype(np.uint8), centers
