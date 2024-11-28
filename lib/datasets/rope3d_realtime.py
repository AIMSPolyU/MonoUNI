"""
This class is modified based on the Rope3D script and is used to access the Carla camera.

"""

import os
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import copy

from lib.datasets.rope3d_utils import get_objects_from_label
from lib.datasets.rope3d_utils import Calibration, Denorm

class Rope3D_Realtime(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 4
        self.max_objs = 50
        self.class_name =  ['car','big_vehicle','pedestrian','cyclist']
        self.cls2id = {'car': 0,'big_vehicle': 1,'pedestrian': 2,'cyclist': 3}
        self.resolution = np.array([960, 512])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        if cfg['load_data_once']:
            self.load_data_once = True
        else:
            self.load_data_once = False
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.288762253204939, 1.6939648801353426, 4.25589251897889],
                                       [1.7199308570318539, 1.7356837654961508, 4.641152817981265],
                                       [2.682263889273618, 2.3482764551684268, 6.940250839428722],
                                       [2.9588510594399073, 2.5199248789610693, 10.542197736838778]])
        # data split loading
        # print(split)
        self.split = 'val'
        # split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        # self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = root_dir
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.label_dir = os.path.join(self.data_dir, 'label_2_4cls_for_train')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.denorm_dir = os.path.join(self.data_dir, 'denorm')
        self.box3d_dense_depth_dir = os.path.join(self.data_dir, 'box3d_depth_dense')

        self.interval_max = cfg['interval_max']
        self.interval_min = cfg['interval_min']

        # data augmentation configuration
        self.data_augmentation = False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']
        self.crop_with_optical_center = cfg['crop_with_optical_center']
        self.crop_with_optical_center_with_fx_limit = True
        self.scale_expand = cfg['scale_expand']
        

        self.labels = []
        # print('load_all_labels')
        label_file = os.path.join(self.label_dir, '1632_fa2sd4a11North151_420_1613710840_1613716786_1_obstacle.txt')
        assert os.path.exists(label_file)
        self.labels.append(get_objects_from_label(label_file))

        # print('load_all_calib')
        self.calib = []
        calib_file = os.path.join(self.calib_dir, '1632_fa2sd4a11North151_420_1613710840_1613716786_1_obstacle.txt')
        if os.path.exists(calib_file) != True:
            print("calib_file : ", calib_file)
        assert os.path.exists(calib_file)
        self.calib.append(Calibration(calib_file))

        self.denorms = []
        denorm_file = os.path.join(self.denorm_dir, '1632_fa2sd4a11North151_420_1613710840_1613716786_1_obstacle.txt')
        assert os.path.exists(denorm_file)
        self.denorms.append(Denorm(denorm_file))

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

        # carla client set carla in here
        from lib.carlaClient.carla_client import CarlaClient
        self.carla_client = CarlaClient(ip="host.docker.internal", port=2000,
                            camera_position=(2.0, 0.0, 1.5),
                            camera_rotation=(0.0, 180.0, 0.0),
                            resolution=(1920, 1080),
                            output_folder="carla_images",
                            save_image=True)
        self.carla_client.setup_camera()
        
    def fetch_image(self):
        """
        fetch image from carla clinet
        return:
            image data array
            image name frame id
        """
        return self.carla_client.get_image()
        

    def get_label(self, idx):
        return self.labels[0]
        # label_file = os.path.join(self.label_dir,  idx+'.txt')
        # assert os.path.exists(label_file)
        # return get_objects_from_label(label_file)

    def get_calib(self, idx):
        return self.calib[0]
        # calib_file = os.path.join(self.calib_dir,  idx+'.txt')
        # assert os.path.exists(calib_file)
        # return Calibration(calib_file)

    def get_denorm(self,idx):
        # denorm_file = os.path.join(self.denorm_dir, '%s.txt' % idx)
        # assert os.path.exists(denorm_file)
        # return Denorm(denorm_file)
        return copy.deepcopy(self.denorms[0])

    def __len__(self):
        """
        """
        # return self.idx_list.__len__()
        return 1

    def Flip_with_optical_center(self, img, calib,mean):
        cx = calib.P2[0, 2]
        cy = calib.P2[1, 2]
        h,w,_ = img.shape
        if cx < w/2:
            x_min = 0
            x_max = int(cx*2)+1
        else:
            x_max = int(w)
            x_min = int(x_max- (x_max - cx-1)*2)-1
        crop_box = [x_min, 0, x_max, int(h)]
        crop_img = img[0:int(h),x_min:x_max, :]
        flip_img = crop_img[:, ::-1, :]
        res_img = np.ones((h, w, 3), dtype=img.dtype)
        res_img *= np.array(mean, dtype=img.dtype)
        res_img[0:int(h),x_min:x_max,:] = flip_img
        return res_img,crop_box

        
    def __getitem__(self, item):
        inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin = self.get_data(item)
        return inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin
    

    def get_data(self, item):
        """
        index: 
        """
        from torchvision import transforms
        #  ============================   get inputs   ===========================
        # index = self.idx_list[item]  # index mapping, get real data id
        # image loading
        # print(index)
        # img = self.get_image(index)
        img, img_name = self.fetch_image() # fecth image from carla
        
        calib = copy.deepcopy(self.get_calib(0))
        img_size = np.array([img.shape[1],img.shape[0]])
        center = np.array(img_size) /2.
        Denorm_ = self.get_denorm(0)
        
        new_img_size = [img.shape[1],img.shape[0]]

        img = cv2.resize(img,(self.resolution[0],self.resolution[1]))
        # img_cp = copy.deepcopy(img)
        img = img.astype(np.float32)  / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  
                
        features_size = self.resolution // self.downsample# W * H

        crop_size =  np.array(new_img_size)
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32) 


        #  ============================   get labels   ==============================
        targets = {}
        # collect return data
        inputs = img
        info = {'img_id': [img_name],
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size,}
        
        
        return inputs, calib.P2, coord_range, targets, info, Denorm_.pitch_cos, Denorm_.pitch_sin   #calib.P2
    
    def _ioa_matrix(self, a, b):
        max_i = np.maximum(a[:2],b[:2])
        min_i = np.minimum(a[2:],b[2:])

        area_i = np.prod(min_i - max_i) * (max_i < min_i).all()
        area_a = np.prod(a[2:] - a[:2])
        # area_b = np.prod(b[2:] - b[:2], axis=1)
        # area_o = (area_a+ area_b - area_i)
        return area_i / (area_a + 1e-10)
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = Rope3D_Realtime('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.bool))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
