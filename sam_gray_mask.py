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

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='data/LOL/our485/high', help='directory of data to be segmented')
args = parser.parse_args()




sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


import os
sourcedir = args.source_dir
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
            
            # Binary
        os.makedirs(f'{sourcedir}_semanticB', exist_ok=True)
        for i, mask in enumerate(masks):
            save_path = os.path.join(f'{sourcedir}_semanticB', f'{os.path.splitext(filename)[0]}_semanticB_{i}.png')
            cv2.imwrite(save_path, np.uint8(mask["segmentation"]) * 255)
            
            #Color
        os.makedirs(f'{sourcedir}_semantic_gray', exist_ok=True)
        save_path = os.path.join(f'{sourcedir}_semantic_gray', f'{os.path.splitext(filename)[0]}_semantic.png')
        print(save_path)
        for i, mask in enumerate(masks):
            mask_bool = mask['segmentation']
            if i < 127:
                color_value = 2 * (i + 1)
            else:
                color_value = 2 * (255 - i) - 1
            gray_img[mask_bool] = color_value
        cv2.imwrite(save_path, gray_img)
        