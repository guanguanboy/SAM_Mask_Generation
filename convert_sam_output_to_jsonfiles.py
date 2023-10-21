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
import json

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='data/LOL/eval15/high', help='directory of data to be segmented')
args = parser.parse_args()




sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam, output_mode='coco_rle')

start_id = 20000
seg_start_id = 1000000
import os
sourcedir = args.source_dir

json_file_path = sourcedir +'_json'
if not os.path.exists(json_file_path):
    os.mkdir(json_file_path)

for i, filename in enumerate(os.listdir(sourcedir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print(f'{i}th pic: {filename}')
        # read image 
        img_path = os.path.join(sourcedir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        output_dict = {}
        image_info = {}
        image_info['image_id'] = i + start_id
        image_info['file_name'] = filename
        height, width, channels = image.shape
        image_info['width'] = width
        image_info['height'] = height
        output_dict['image'] = image_info

        # segment
        masks = mask_generator.generate(image)
        
        annotation_list = []


        for i, mask in enumerate(masks):
            mask['id'] = seg_start_id + i
            annotation_list.append(mask)


        output_dict['annotations'] = annotation_list


        # 将字典保存为 JSON 文件

        json_file_name = os.path.join(json_file_path, filename[:-4] + '.json')
        with open(json_file_name, "w") as file:
            json.dump(output_dict, file)
       
        