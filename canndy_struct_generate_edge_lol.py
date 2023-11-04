from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator
import os

#LOLv2-real
#/data/liguanlin/Datasets/LowLight/LOL-v2/Real_captured/Train/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Real_captured/Test/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Synthetic/Test/Low
#/data/liguanlin/Datasets/LowLight/LOL-v2/Synthetic/Train/Low

#LOLv1
#data/LOL/eval15/low
#data/LOL/our485/low


def normalize_image(image):
    # Normalize the image to the range [0, 1]
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    return image


def main():

    sourcedir = 'data/LOL/eval15/low'
    save_root_dir = sourcedir + '_structcanndyreversebirnaryedge'

    for i, filename in enumerate(os.listdir(sourcedir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            print(f'{i}th pic: {filename}')
            
            # read image 
            img_path = os.path.join(sourcedir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            edges = normalize_image(image)
            edges = edges.astype(np.float32)
            edge_detection = cv2.ximgproc.createStructuredEdgeDetection(
                './model/model.yml.gz')
            orimap = edge_detection.computeOrientation(edges)
            edges = edge_detection.edgesNms(edges, orimap)

            # save edge
            os.makedirs(save_root_dir, exist_ok=True)
            save_path = os.path.join(save_root_dir, f'{os.path.splitext(filename)[0]}_semantic.png')
            print(save_path)
            #plt.imsave(save_path, edges, cmap='binary')
            #plt.imsave(save_path, edges, cmap='gray')
            #plt.imshow(1edges, cmap='binary',vmin=1, vmax=0)
            #plt.savefig(save_path)
            plt.imsave(save_path, 1-edges, cmap='binary')
            
             

if __name__ == "__main__":
    main()