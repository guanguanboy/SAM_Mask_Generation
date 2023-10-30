# This script batch segments pictures in <source> directory using SAM（Segment Anything）.
# And save the semantic pictures in <source_semantic> for colored versions,
# and <source_semanticB> for binary versions

# Author: Qihan Zhao

# Prerequisite:
# 1. download this script in directory `SAM`
# 2. download SAM weights in directory `SAM`
# 3. add <source> file in directory `SAM`

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#LOLv2-real
#/data/liguanlin/Datasets/LowLight/LOL-v2/Real_captured/Train/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Real_captured/Test/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Synthetic/Test/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Synthetic/Train/Low

#LOLv1
#data/LOL/eval15/low
#data/LOL/our485/low

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='data/LOL/eval15/low', help='directory of data to be segmented')
args = parser.parse_args()


sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


import os
sourcedir = args.source_dir
save_root_dir = sourcedir + '_semantic_gray_reverted_new'


for i, filename in enumerate(os.listdir(sourcedir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print(f'{i}th pic: {filename}')
        # read image 
        img_path = os.path.join(sourcedir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # segment
        masks = mask_generator.generate(image)
        
        # save semantic
        gray_img = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 1), dtype=np.int8)
        #img[:,:,3] = 0
            
        #Color
        os.makedirs(save_root_dir, exist_ok=True)
        save_path = os.path.join(save_root_dir, f'{os.path.splitext(filename)[0]}_semantic.png')
        print(save_path)

        original_gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        region_mean_list = []
        for i, mask in enumerate(masks):
            mask_bool = mask['segmentation']

            region_mean = original_gray_image[mask_bool].mean()
            region_mean_list.append(region_mean)
        
        sorted_indices = np.argsort(region_mean_list)
        
        #逆序操作
        reverse_sorted_indices = sorted_indices[::-1]

        masks_sorted = [masks[i] for i in reverse_sorted_indices]

        #根据mask个数确定step
        step = 255.0 / len(masks)
        for i, mask in enumerate(masks_sorted):
            if i < 127:
                color_value = 2 * (i + 1)
            else:
                color_value = 2 * (255 - i) - 1  
        cv2.imwrite(save_path, gray_img)
        