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
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='/data/liguanlin/Datasets/LowLight/LOL-v1/our485/low', help='directory of data to be segmented')
args = parser.parse_args()

sourcedir = args.source_dir

save_dir = '/data/liguanlin/Datasets/LowLight/LOL-v1/our485'
for i, filename in enumerate(os.listdir(sourcedir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print(f'{i}th pic: {filename}')
        # read image 
        img_path = os.path.join(sourcedir, filename)
        image = cv2.imread(img_path)
        
            
        # Binary
        folder_name = os.path.join(save_dir, 'sam_whole_masks')
        os.makedirs(folder_name, exist_ok=True)

        mask_save_path = os.path.join(folder_name, filename)

        height, width, channels = image.shape

        mask_array = np.ones([height, width, 1], dtype=np.uint8) * 255
        cv2.imwrite(mask_save_path, mask_array)


        