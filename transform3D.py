# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 22:26:46 2025

@author: ZML
"""

import torch
import numpy as np
import random
from scipy import ndimage
import numbers


class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, label=None):
        if label is not None:
            for t in self.transforms:
                img, label = t(img, label)
            return img, label
        else:
            for t in self.transforms:
                img = t(img)
            return img


class ToTensor3D:
    def __call__(self, img, label=None):
        # 确保数据是float32类型
        if isinstance(img, list):
            img = [torch.from_numpy(img_.astype(np.float32)) for img_ in img]
        else:
            img = torch.from_numpy(img.astype(np.float32))
        
        if label is not None:
            label = torch.from_numpy(label.astype(np.int64))

            return img, label
        return img


def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img, label=None):
        
        if self.mean is None:
            high = np.max(img)
            low = np.min(img)
            img = (img - low) / (high - low)
        else:
            img = (img - self.mean) / self.std
        
        if label is not None:
            
            return img, label
        return img


class RandomCrop3D:
    def __init__(self, size):

        self.size = size
    
    def __call__(self, img, label=None):
        
        if isinstance(img, list):
            d, h, w = img[0].shape
            if d <= self.size[0]:
                d_start = 0
            else:
                d_start = random.randint(0, d - self.size[0])
                
            if h <= self.size[1]:
                h_start = 0
            else:
                h_start = random.randint(0, h - self.size[1])
                
            if w <= self.size[2]:
                w_start = 0
            else:
                w_start = random.randint(0, w - self.size[2])

            img_new = []
            for img_ in img:
                img_cropped = img_[
                    d_start:d_start + self.size[0],
                    h_start:h_start + self.size[1],
                    w_start:w_start + self.size[2]
                ]
                img_new.append(img_cropped)
            img_cropped = img_new
        else:
            d, h, w = img.shape
            '''
            if d < self.size[0]:
                if label is not None:
                    
                    return img, label
                return img
            '''
            if d <= self.size[0]:
                d_start = 0
            else:
                d_start = random.randint(0, d - self.size[0])
                
            if h <= self.size[1]:
                h_start = 0
            else:
                h_start = random.randint(0, h - self.size[1])
                
            if w <= self.size[2]:
                w_start = 0
            else:
                w_start = random.randint(0, w - self.size[2])
     
            img_cropped = img[
                d_start:d_start + self.size[0],
                h_start:h_start + self.size[1],
                w_start:w_start + self.size[2]
            ]
            
        if label is not None:
            label_cropped = label[
                d_start:d_start + self.size[0],
                h_start:h_start + self.size[1],
                w_start:w_start + self.size[2]
            ]

            return img_cropped, label_cropped
        
        return img_cropped
    


class CenterCrop3D:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img, label=None):
        
        if isinstance(img, list):
            d, h, w = img[0].shape

            d_start = (d - self.size[0]) // 2
            h_start = (h - self.size[1]) // 2
            w_start = (w - self.size[2]) // 2
        
            img_new = []
            for img_ in img:
                img_cropped = img_[
                    d_start:d_start + self.size[0],
                    h_start:h_start + self.size[1],
                    w_start:w_start + self.size[2]
                ]
                img_new.append(img_cropped)
            img_cropped = img_new
        else:
            d, h, w = img.shape
            
            d_start = (d - self.size[0]) // 2
            h_start = (h - self.size[1]) // 2
            w_start = (w - self.size[2]) // 2

            img_cropped = img[
                d_start:d_start + self.size[0],
                h_start:h_start + self.size[1],
                w_start:w_start + self.size[2]
            ]
        
        if label is not None:
            label_cropped = label[
                d_start:d_start + self.size[0],
                h_start:h_start + self.size[1],
                w_start:w_start + self.size[2]
            ]
            return img_cropped, label_cropped
        
        return img_cropped


class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, label=None):
        
        if isinstance(img, list):
            if random.random() < self.p:
                img = [np.flip(img_, axis=0).copy() for img_ in img]
                if label is not None:
                    label = np.flip(label, axis=0).copy()
            
            if random.random() < self.p:
                img = [np.flip(img_, axis=1).copy() for img_ in img]
                if label is not None:
                    label = np.flip(label, axis=1).copy()
            
            if random.random() < self.p:
                img = [np.flip(img_, axis=2).copy() for img_ in img]
                if label is not None:
                    label = np.flip(label, axis=2).copy()
        else:
            
            if random.random() < self.p:
                img = np.flip(img, axis=0).copy()
                if label is not None:
                    label = np.flip(label, axis=0).copy()
                    
            if random.random() < self.p:
                img = np.flip(img, axis=1).copy()
                if label is not None:
                    label = np.flip(label, axis=1).copy()
            
            if random.random() < self.p:
                img = np.flip(img, axis=2).copy()
                if label is not None:
                    label = np.flip(label, axis=2).copy()
        
        if label is not None:
            
            return img, label
        return img


class RandomRotate3D:
    def __init__(self, angle_range=(-15, 15), axes=(0, 1), p=0.5):
        self.angle_range = angle_range
        self.axes = axes
        self.p = p
    
    def __call__(self, img, label=None):
        
        if random.random() < self.p:
            angle = random.uniform(*self.angle_range)
            
            if isinstance(img, list):
                img = [ndimage.rotate(img_, angle, axes=self.axes, reshape=False, order=1, mode='constant') for img_ in img]
            else:
                img = ndimage.rotate(img, angle, axes=self.axes, 
                                     reshape=False, order=1, mode='constant')
            
            if label is not None:
                label = ndimage.rotate(label, angle, axes=self.axes,
                                      reshape=False, order=0, mode='constant')
        
        if label is not None:
            return img, label
        return img


class RandomScale3D:
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            scale_factor = random.uniform(*self.scale_range)
            original_shape = img.shape
            
            new_shape = tuple(int(dim * scale_factor) for dim in original_shape)
            
            img_scaled = ndimage.zoom(img, scale_factor, order=1)
            
            if img_scaled.shape != original_shape:
                img = self._adjust_size(img_scaled, original_shape)
                
                if label is not None:
                    label_scaled = ndimage.zoom(label, scale_factor, order=0)
                    label = self._adjust_size(label_scaled, original_shape)
        
        if label is not None:
            return img, label
        return img
    
    def _adjust_size(self, img, target_shape):
        result = np.zeros(target_shape, dtype=img.dtype)
        slices = []
        target_slices = []
        
        for i, (new_dim, target_dim) in enumerate(zip(img.shape, target_shape)):
            if new_dim > target_dim:
                start = (new_dim - target_dim) // 2
                slices.append(slice(start, start + target_dim))
                target_slices.append(slice(0, target_dim))
            else:
                slices.append(slice(0, new_dim))
                start = (target_dim - new_dim) // 2
                target_slices.append(slice(start, start + new_dim))
        
        result[tuple(target_slices)] = img[tuple(slices)]
        return result


class RandomBrightnessContrast3D:
    def __init__(self, brightness_range=(0.8, 1.2), 
                 contrast_range=(0.8, 1.2), p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_range)
            if isinstance(img, list):
                img = [img_ * brightness_factor for img_ in img]
            else:
                img = img * brightness_factor
            
            contrast_factor = random.uniform(*self.contrast_range)
            if isinstance(img, list):
                img_new = []
                for img_ in img:
                    mean_val = np.mean(img_)
                    img_ = (img_ - mean_val) * contrast_factor + mean_val
                    img_ = np.clip(img_, 0, np.percentile(img_, 99.5))
                    img_new.append(img_)
                img = img_new
            else:
                mean_val = np.mean(img)
                img = (img - mean_val) * contrast_factor + mean_val
                img = np.clip(img, 0, np.percentile(img, 99.5))
        
        if label is not None:
            return img, label
        return img


class GaussianNoise3D:
    """Add Gaussian noise to the image"""
    def __init__(self, mean=0, std_range=(0, 0.1), p=0.5):
        self.mean = mean
        self.std_range = std_range
        self.p = p
    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            if isinstance(img, list):
                noise = np.random.normal(self.mean, std, img[0].shape)
                img_new = []
                for img_ in img:
                    img_ = img_ + noise
                    img_ = np.clip(img_, 0, np.percentile(img_, 99.5))
                    img_new.append(img_)
                img = img_new
            else:
                noise = np.random.normal(self.mean, std, img.shape)
                img = img + noise
                img = np.clip(img, 0, np.percentile(img, 99.5))
        
        if label is not None:
            return img, label
        return img


class RandomElasticDeformation3D:
    """Random elastic deformation (applicable to both images and labels)"""
    def __init__(self, alpha_range=(0, 100), sigma=10, p=0.3):
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img, label=None):
        if random.random() < self.p:
            alpha = random.uniform(*self.alpha_range)
            
            if isinstance(img, list):
                shape = img[0].shape
            else:
                shape = img.shape
            
            dx = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1) * alpha,
                self.sigma, mode="constant", cval=0
            )
            dy = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1) * alpha,
                self.sigma, mode="constant", cval=0
            )
            dz = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1) * alpha,
                self.sigma, mode="constant", cval=0
            )
            
            z, y, x = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            
            indices = (z + dz, y + dy, x + dx)
            
            if isinstance(img, list):
                img_deformed = [ndimage.map_coordinates(img_, indices, order=1, mode='constant') for img_ in img]
            else:
                img_deformed = ndimage.map_coordinates(img, indices, order=1, mode='constant')
            
            if label is not None:
                label_deformed = ndimage.map_coordinates(
                    label, indices, order=0, mode='constant'
                )
                return img_deformed, label_deformed
            
            return img_deformed
        
        if label is not None:
            return img, label
        return img


class Pad3D:
    """3D image padding to target size"""
    def __init__(self, target_size=None, padding_mode='constant', constant_values=0):
        self.target_size = target_size
        self.padding_mode = padding_mode
        self.constant_values = constant_values
    
    def __call__(self, img, label=None):
        if self.target_size is None:
            if label is not None:
                
                return img, label
            return img
        
        if isinstance(img, list):
            current_size = img[0].shape
        else:
            current_size = img.shape

        pad_width = []

        for curr_dim, target_dim in zip(current_size, self.target_size):
            if curr_dim < target_dim:
                total_pad = target_dim - curr_dim
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        
        if isinstance(img, list):
            if self.padding_mode == 'constant':
                img_padded = [np.pad(img_, pad_width, mode=self.padding_mode, constant_values=self.constant_values) for img_ in img]
            else:
                img_padded = [np.pad(img_, pad_width, mode=self.padding_mode) for img_ in img]
        else:
            if self.padding_mode == 'constant':
                img_padded = np.pad(img, pad_width, mode=self.padding_mode, constant_values=self.constant_values)
            else:
                img_padded = np.pad(img, pad_width, mode=self.padding_mode)
            

        if label is not None:
            if self.padding_mode == 'constant':
                label_padded = np.pad(
                    label, pad_width, 
                    mode='constant', 
                    constant_values=0  
                )
            else:
                label_padded = np.pad(
                    label, pad_width, 
                    mode='constant', 
                    constant_values=0  
                )
                
            return img_padded, label_padded
        
        return img_padded

def get_train_transforms_3d(crop_size=(128, 128, 128), 
                           normalize_mean=0.0, 
                           normalize_std=1.0):
    """获取训练时的transform管道"""
    train_transforms = Compose3D([
        RandomCrop3D(crop_size),            
        RandomFlip3D(p=0.5),                
        RandomRotate3D(angle_range=(-15, 15), p=0.3),  
        RandomScale3D(scale_range=(0.8, 1.2), p=0.3), 
        RandomBrightnessContrast3D(p=0.3), 
        GaussianNoise3D(p=0.3),             
        RandomElasticDeformation3D(p=0.1), 
        Normalize3D(mean=normalize_mean, std=normalize_std),  
        ToTensor3D()                         
    ])
    return train_transforms


def get_val_transforms_3d(crop_size=(128, 128, 128), 
                         normalize_mean=0.0, 
                         normalize_std=1.0):
    val_transforms = Compose3D([
        CenterCrop3D(crop_size),            
        Normalize3D(mean=normalize_mean, std=normalize_std),  
        ToTensor3D()                         
    ])
    return val_transforms

class MedicalDataset3D(torch.utils.data.Dataset):

    def __init__(self, data_list, transform=None, is_train=True):
        self.data_list = data_list
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label_path = self.data_list[idx]
        img = np.load(img_path)  
        label = np.load(label_path)
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        if self.transform:
            img, label = self.transform(img, label)
        else:
            img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  
            label = torch.from_numpy(label.astype(np.int64))
        
        return img, label



if __name__ == "__main__":
    data_list = [("img1.npy", "label1.npy"), ("img2.npy", "label2.npy")]
    
    train_transforms = get_train_transforms_3d(
        crop_size=(128, 128, 128),
        normalize_mean=0.0,
        normalize_std=1.0
    )
    
    train_dataset = MedicalDataset3D(
        data_list=data_list,
        transform=train_transforms,
        is_train=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")  # (B, 1, D, H, W)
        print(f"  Labels shape: {labels.shape}")  # (B, D, H, W)
        break