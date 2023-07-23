

import os
import json
import csv
import random
import pickle
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from PIL import Image
from sklearn.model_selection import train_test_split



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from torchvision.transforms import functional as F


class GlaucomaDataset(Dataset):
    def __init__(self, root_dir, split='train', output_size=(256,256), max_images=None):
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.segs = []
        self.max_images = max_images
        self.img_width, self.img_height = output_size

        # Load data index
        for direct in self.root_dir:
            self.image_filenames = []
            for path in os.listdir(os.path.join(direct, "Images_Square")):
                if(not path.startswith('.')):
                    self.image_filenames.append(path)

            num_images = 0
            for k in range(len(self.image_filenames)):
                # Skip loading if max_images is specified and the limit has been reached
                if max_images is not None and num_images >= max_images:
                    break

                print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(direct, "Images_Square", self.image_filenames[k])
                img = np.array(Image.open(img_name).convert('RGB'))

                if split != 'test':
                    seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
                    mask = np.array(Image.open(seg_name, mode='r'))
                    od = (mask==1.).astype(np.float32)
                    oc = (mask==2.).astype(np.float32)
                    
                    # Check if both masks are not empty, i.e., they contain at least one non-zero pixel
                    if np.any(od) and np.any(oc):
                        img = transforms.functional.to_tensor(img)
                        img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
                        self.images.append(img)
                        od = torch.from_numpy(od[None,:,:])
                        oc = torch.from_numpy(oc[None,:,:])
                        od = transforms.functional.resize(od, output_size, interpolation=Image.Resampling.NEAREST)
                        oc = transforms.functional.resize(oc, output_size, interpolation=Image.Resampling.NEAREST)
                        self.segs.append(torch.cat([od, oc], dim=0))
                        num_images += 1

            print('Succesfully loaded {} dataset.'.format(split) + ' '*50)
    # def __init__(self, root_dir, split='train', output_size=(256,256), max_images=None):
    #     self.output_size = output_size
    #     self.root_dir = root_dir
    #     self.split = split
    #     self.images = []
    #     self.segs = []
    #     self.max_images = max_images
    #     self.img_width, self.img_height = output_size

    #     # Load data index
    #     for direct in self.root_dir:
    #                 self.image_filenames = []
    #                 for path in os.listdir(os.path.join(direct, "Images_Square")):
    #                     if(not path.startswith('.')):
    #                         self.image_filenames.append(path)

    #                 num_images = 0
    #                 for k in range(len(self.image_filenames)):
    #                     # Skip loading if max_images is specified and the limit has been reached
    #                     if max_images is not None and num_images >= max_images:
    #                         break

    #                     print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
    #                     img_name = os.path.join(direct, "Images_Square", self.image_filenames[k])
    #                     img = np.array(Image.open(img_name).convert('RGB'))

    #                     if split != 'test':
    #                         seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
    #                         mask = np.array(Image.open(seg_name, mode='r'))
    #                         od = (mask==1.).astype(np.float32)
    #                         oc = (mask==2.).astype(np.float32)
                            
    #                         # Check if both masks are not empty, i.e., they contain at least one non-zero pixel
    #                         if np.any(od) and np.any(oc):
    #                             img = transforms.functional.to_tensor(img)
    #                             img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)
    #                             self.images.append(img)
    #                             od = torch.from_numpy(od[None,:,:])
    #                             oc = torch.from_numpy(oc[None,:,:])
    #                             od = transforms.functional.resize(od, output_size, interpolation=Image.Resampling.NEAREST)
    #                             oc = transforms.functional.resize(oc, output_size, interpolation=Image.Resampling.NEAREST)
    #                             self.segs.append(torch.cat([od, oc], dim=0))
    #                             num_images += 1

    #                 print('Succesfully loaded {} dataset.'.format(split) + ' '*50)

    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        # load image
        img = self.images[idx]
        # load segmentation masks (for both optic disk and optic cup)
        seg = self.segs[idx]
        # For instance segmentation, each mask should be a binary mask of shape (H, W).
        # Therefore, we need to split the combined mask into two separate masks.
        od_mask, oc_mask = seg[0], seg[1]
        

        # Find bounding boxes around each mask. The bounding box is represented as
        # [xmin, ymin, width, height], which is the format expected by Mask R-CNN.
        # Find bounding boxes around each mask. The bounding box is represented as
        # [xmin, ymin, width, height], which is the format expected by Mask R-CNN.
        od_bbox = torch.tensor(self.mask_to_bbox(od_mask.numpy()))
        oc_bbox = torch.tensor(self.mask_to_bbox(oc_mask.numpy()))

        # print("od_bbox:", od_bbox, "Type:", type(od_bbox), "Shape:", od_bbox.shape)
        # print("oc_bbox:", oc_bbox, "Type:", type(oc_bbox), "Shape:", oc_bbox.shape)



        # od_bbox = np.array(od_bbox, dtype=np.float32)
        # oc_bbox = np.array(oc_bbox, dtype=np.float32)
        # boxes = torch.tensor([od_bbox, oc_bbox], dtype=torch.float32)
        od_bbox = np.array(od_bbox)
        oc_bbox = np.array(oc_bbox)
        
        boxes = torch.tensor(np.array([od_bbox, oc_bbox]))





        # Check that the bounding boxes are valid
        img_height, img_width = self.img_height, self.img_width
        for bbox in [od_bbox, oc_bbox]:
                assert bbox[0] >= 0, "xmin should be non-negative"
                assert bbox[1] >= 0, "ymin should be non-negative"
                assert bbox[2] > 0, "width should be positive"
                assert bbox[3] > 0, "height should be positive"
                try:
                    bbox[0] = min(bbox[0], img_width - bbox[2])
                    bbox[1] = min(bbox[1], img_height - bbox[3])
                    assert bbox[0] + bbox[2] <= img_width, "xmin + width should be within the image width"
                    assert bbox[1] + bbox[3] <= img_height, "ymin + height should be within the image height"
                except AssertionError:
                    print(f"Error with bounding box {bbox} for image of size {img.shape}")
                    raise

        # The labels are a tensor of class IDs. In this case, you might want to use
        # 1 for optic disk and 2 for optic cup, as you did when creating the masks.
        labels = torch.tensor([ 1,2], dtype=torch.int64)

        # Now, we need to put the masks and bounding boxes into the right format.
        # The masks should be a tensor of shape (num_objs, H, W),
        # and the bounding boxes should be in a (num_objs, 4) tensor.
        masks = torch.stack([od_mask, oc_mask])
        boxes = torch.tensor([od_bbox, oc_bbox])


        # Pack the bounding boxes and labels into a dictionary
        target = {"boxes": boxes, "labels": labels, "masks": masks}

        # Convert bounding box format
        target['boxes'][:, 2] += target['boxes'][:, 0]  # xmax = xmin + width
        target['boxes'][:, 3] += target['boxes'][:, 1]  # ymax = ymin + height

        

        return img, target



    @staticmethod
    def mask_to_bbox(mask):
        # Find the bounding box of a binary mask.
        # This method assumes that the input is a binary mask with 0s and 1s.
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax - xmin, ymax - ymin]


    
    # def __getitem__(self, idx):
    #     # load image
    #     img = self.images[idx]
    #     # load segmentation masks (for both optic disk and optic cup)
    #     seg = self.segs[idx]
    #     # For instance segmentation, each mask should be a binary mask of shape (H, W).
    #     # Therefore, we need to split the combined mask into two separate masks.
    #     od_mask, oc_mask = seg[0], seg[1]

    #     # Find bounding boxes around each mask. The bounding box is represented as
    #     # [xmin, ymin, width, height], which is the format expected by Mask R-CNN.
    #     od_bbox = self.mask_to_bbox(od_mask.numpy())
    #     oc_bbox = self.mask_to_bbox(oc_mask.numpy())

    #     # The labels are a tensor of class IDs. In this case, you might want to use
    #     # 1 for optic disk and 2 for optic cup, as you did when creating the masks.
    #     labels = torch.tensor([1, 2], dtype=torch.int64)

    #     # Now, we need to put the masks and bounding boxes into the right format.
    #     # The masks should be a tensor of shape (num_objs, H, W),
    #     # and the bounding boxes should be in a (num_objs, 4) tensor.
    #     masks = torch.stack([od_mask, oc_mask])
    #     boxes = torch.tensor([od_bbox, oc_bbox])

    #     # Pack the bounding boxes and labels into a dictionary
    #     target = {"boxes": boxes, "labels": labels, "masks": masks}

    #     return img, target

    # @staticmethod
    # def mask_to_bbox(mask):
    #     # Find the bounding box of a binary mask.
    #     # This method assumes that the input is a binary mask with 0s and 1s.
    #     pos = np.where(mask)
    #     xmin = np.min(pos[1])
    #     xmax = np.max(pos[1])
    #     ymin = np.min(pos[0])
    #     ymax = np.max(pos[0])
    #     return [xmin, ymin, xmax - xmin, ymax - ymin]
