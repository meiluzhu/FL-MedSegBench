import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import math
import random
import copy
import functools
from PIL import Image
import os
import glob
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import SimpleITK as sitk
import math

class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float() 
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return self.images.shape[0]

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] 
        return image, label

class CustomSubset(torch.utils.data.Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        dataset.targets = torch.tensor(dataset.targets)
        # print(dataset.targets)
        self.targets = dataset.targets[indices]
        # print(len(self.targets))
        self.classes = dataset.classes 
        self.indices = indices

    def __getitem__(self, idx): 
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self):
        return len(self.indices)


class EyeDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # H, W, C
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch



class PolypDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # H, W, C
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch


class PathologyDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # H, W, C
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = mask.float()# H, W
        
        assert ((mask==0).sum()+(mask==1).sum()) == mask.shape[0]*mask.shape[1]
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch


class FundusDataset(Dataset):
    def __init__(self, args, data_path, image_list, transform=None, is_train = False):

        self.data_path = data_path
        self.images_list = image_list
        self.transform = transform
        self.args = args
        
    def __len__(self):

        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.images_list[idx])
        img = Image.open(img_path).convert('RGB')
        label_path = img_path.replace('Images', 'Labels')
        mask = Image.open(label_path).convert('L')

        img = np.array(img)
        mask = np.array(mask)
        
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_path
                 }
            
        return batch
    
class ProstateDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) #C H, W
        mask = np.array(mask) #H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']
            
        img = img/255#C H, W
        mask = (mask > 0).float() #H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    
    
class UltrasoundDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks


    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) #C H, W
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']
        
        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    
    
class PancreasDataset(Dataset): #3D
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks
            
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # D, H, W
        mask = np.array(mask) # D, H, W
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        img = img/255 # D, H, W
        img = img.float().unsqueeze(0) #1, D, H, W
        mask = mask.float()#D, H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    
    
class KiTS19Dataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        if self.images is None:
            return len(self.img_name_list)
            
        return len(self.images)

    def __getitem__(self, idx):
        
        if self.images is None:
            img = Image.open(self.img_name_list[idx])
            mask = Image.open(self.img_name_list[idx].replace('images', 'masks'))
        else:
            img = self.images[idx]
            mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # H, W, C
        if len(img.shape) == 2:
            img = img[:,:, None]
            img = np.repeat(img, 3, axis=2)
        mask = np.array(mask) # H, W, C
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = mask.float().unsqueeze(0)
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    
    
class MMSDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # D, H, W
        mask = np.array(mask) # D, H, W
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        img = img/255 # D, H, W
        img = img.float().unsqueeze(0) #1, D, H, W
        mask = mask.float() #D, H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    
    
class FeTS2022Dataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        if self.images is None:
            return len(self.img_name_list)
            
        return len(self.images)

    def __getitem__(self, idx):
        
        if self.images is None:
            patient_id = self.img_name_list[idx]
            img= np.load(os.path.join(self.args.data_path, patient_id+'.npz'))['img_list']
            mask= np.load(os.path.join(self.args.data_path, patient_id+'_seg.npz'))['mask']
        else:
            img = self.images[idx]
            mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = [np.array(img_).astype(np.float32) for img_ in img] # D, H, W
        mask = np.array(mask) # D, H, W
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        
        img = torch.stack(img, dim=0) #Channel, D, H, W
        img = img/255 # Channel, D, H, W
        img = img.float() #Channel, D, H, W
        mask = mask.float() #D, H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch
    

def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Data(object):
    def __init__(self, args):
        self.args = args
        if args.dataset == 'Fundus':
            transform_train = A.Compose([
                        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                train_data_path = os.path.join(base_dir, client_name, 'Train/Original/Images')
                train_images_list = os.listdir(train_data_path)
                train_images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(train_images_list)
                train_len = int(len(train_images_list)*0.8)
                train_filenames = train_images_list[:train_len]
                val_filenames = train_images_list[train_len:]
                
                train_datasets = FundusDataset(args, train_data_path, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', len(train_filenames))
                
                val_datasets = FundusDataset(args, train_data_path, val_filenames, transform=transform_train, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', len(val_filenames))
                
                test_data_path = os.path.join(base_dir, client_name, 'Test/Original/Images')
                test_images_list = os.listdir(test_data_path)
                test_images_list.sort()
                test_datasets = FundusDataset(args, test_data_path, test_images_list, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, ' test', len(test_images_list))
                
        elif args.dataset == 'Meibomian_Gland':
            input_size = (192, 384)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
            
            
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                if client_name == 'MGD1k':
                    data_path = os.path.join(base_dir, client_name, 'Subset/Original')
                if client_name == 'CAMG':
                    data_path = os.path.join(base_dir, client_name, 'Original')
                
                images_list = os.listdir(data_path)
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    img_path = os.path.join(data_path, filename)
                    img = Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.NEAREST)
                    img = np.array(img)
                    label_path = img_path.replace('Original', 'Meibomian_Gland_Labels')
                    if client_name == 'MGD1k':
                        label_path = label_path.replace('JPG', 'png')
                        label_path = label_path.replace('jpg', 'png')
                    if 'LTQ001_5_5' in label_path:
                        print(label_path)
                        label_path = label_path.replace('png', 'tif')
                    if client_name == 'CAMG':
                        label_path = label_path.replace('Image', 'Seg')
                        
                    mask = Image.open(label_path).convert('L').resize((input_size[1], input_size[0]), Image.NEAREST)
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = EyeDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    img_path = os.path.join(data_path, filename)
                    img = Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.NEAREST)
                    img = np.array(img)
                    label_path = img_path.replace('Original', 'Meibomian_Gland_Labels')
                    if client_name == 'MGD1k':
                        label_path = label_path.replace('JPG', 'png')
                        label_path = label_path.replace('jpg', 'png')
                    if 'LTQ001_5_5' in label_path:
                        label_path = label_path.replace('png', 'tif')
                    if client_name == 'CAMG':
                        label_path = label_path.replace('Image', 'Seg')
                        
                    mask = Image.open(label_path).convert('L').resize((input_size[1], input_size[0]), Image.NEAREST)
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = EyeDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))

                
                images, labels = [], []
                for filename in test_filenames:
                    img_path = os.path.join(data_path, filename)
                    img = Image.open(img_path).convert('RGB').resize((input_size[1], input_size[0]), Image.NEAREST)
                    img = np.array(img)
                    label_path = img_path.replace('Original', 'Meibomian_Gland_Labels')
                    if client_name == 'MGD1k':
                        label_path = label_path.replace('JPG', 'png')
                        label_path = label_path.replace('jpg', 'png')
                    if 'LTQ001_5_5' in label_path:
                        label_path = label_path.replace('png', 'tif')
                    if client_name == 'CAMG':
                        label_path = label_path.replace('Image', 'Seg')
                    mask = Image.open(label_path).convert('L').resize((input_size[1], input_size[0]), Image.NEAREST)
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = EyeDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))
                
        elif args.dataset == 'Polyp':
            import tifffile
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = PolypDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = PolypDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))

                
                images, labels = [], []
                for filename in test_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = PolypDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))
                
        elif args.dataset == 'Pathology_COSAS2024':
            
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])

            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'image'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    img = Image.open(os.path.join(data_path, 'image', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'mask', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = PathologyDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    img = Image.open(os.path.join(data_path, 'image', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'mask', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = PathologyDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))

                
                images, labels = [], []
                for filename in test_filenames:
                    img = Image.open(os.path.join(data_path, 'image', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'mask', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = PathologyDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))
        elif args.dataset == 'Prostate':
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(data_path)
                patients = [img_name[:6] for img_name in images_list]
                patients= list(set(patients))
                patients.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(patients)
                train_len = int(len(patients)*0.7)
                val_len = int(len(patients)*0.1)
                train_patients = patients[:train_len]
                val_patients = patients[train_len:train_len+val_len]
                test_patients = patients[train_len+val_len:]
                
                images, labels, slice_id = [], [], []
                for patient_id in train_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                    #print(label_v.shape, image_v.shape)
                    
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                    
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                train_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:', len(train_patients) ,len(images))
                
                images, labels, slice_id = [], [], []
                for patient_id in val_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                    
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                val_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=2, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:', len(val_patients) ,len(images))

                images, labels, slice_id = [], [], []
                for patient_id in test_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                    
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                test_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=2, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:', len(test_patients) ,len(images))

        elif args.dataset == 'FL_Breast_Ultrasound':
            
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])

            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = UltrasoundDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = UltrasoundDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))
                
                images, labels = [], []
                for filename in test_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = UltrasoundDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))


        elif args.dataset == 'MMS':
            import transform3D
            input_size = (16, 256, 256)
            transform_train = transform3D.Compose3D([
                        transform3D.GaussianNoise3D(p=0.2),
                        transform3D.RandomRotate3D(p=0.2),
                        transform3D.RandomElasticDeformation3D(p=0.2),
                        transform3D.RandomBrightnessContrast3D(p=0.2),
                        transform3D.RandomFlip3D(p=0.5),
                        transform3D.RandomCrop3D(input_size),
                        transform3D.Pad3D(input_size),
                        transform3D.ToTensor3D(),
                        ])
    
            transform_test = transform3D.Compose3D([
                        #transform3D.Pad3D(input_size),
                        transform3D.ToTensor3D(),
                        ])
    
            client_names = args.client_names#{'site1', 'site2', 'site3', 'site4', 'site5'}
            data_path = args.data_path #'/mnt/disk/meiluzhu/data/FL_MMS'
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            
            for client_name in client_names:
                patients = os.listdir(os.path.join(data_path, client_name, 'images'))
                patients.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(patients)
                train_len = int(len(patients)*0.7)
                test_len = int(len(patients)*0.2)
                train_patients = patients[:train_len]
                test_patients = patients[train_len:train_len+test_len]
                val_patients = patients[train_len+test_len:]
                
                images, labels, slice_id = [], [], []
                for i, patient_id in enumerate(train_patients):
                    imgdir = os.path.join(data_path, client_name, 'images', patient_id)
                    image_vs = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, client_name, 'masks', patient_id.replace('_sa.nii.gz','_sa_gt.nii.gz'))
                    label_vs = sitk.ReadImage(maskdir)
                    label_vs_ = sitk.GetArrayFromImage(label_vs)
                    for idx in range(label_vs_.shape[0]):
                        if label_vs_[idx].sum()!=0:
                            image_v = image_vs[:, :, :, idx]
                            label_v = label_vs[:, :, :, idx]
                            
                            w, h, d = image_v.GetSize()
                            
                            if d>16:
                                image_v = image_v[:,:, d//2-8:d//2+8]
                                label_v = label_v[:,:, d//2-8:d//2+8]
                                
                            if w>384:
                                image_v = image_v[w//8:w//8*7,:,:]
                                label_v = label_v[w//8:w//8*7,:,:]
                                
                            if h>384:
                                image_v = image_v[:,h//8:h//8*7,:]
                                label_v = label_v[:,h//8:h//8*7,:]
                                
                            image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                            image_v = convert_from_nii_to_png(image_v)
                            label_v = sitk.GetArrayFromImage(label_v) 
        
                            labels.append(label_v)
                            images.append(image_v)
                                
                            slice_id.append([client_name, patient_id, idx])
                
                train_datasets = MMSDataset(args, images, labels, slice_id, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=2, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:', len(train_patients) ,len(images))
                
                images, labels, slice_id = [], [], []
                for patient_id in val_patients:
                    imgdir = os.path.join(data_path, client_name, 'images', patient_id)
                    image_vs = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, client_name, 'masks', patient_id.replace('_sa.nii.gz','_sa_gt.nii.gz'))
                    label_vs = sitk.ReadImage(maskdir)
                    label_vs_ = sitk.GetArrayFromImage(label_vs)
                    for idx in range(label_vs_.shape[0]):
                        if label_vs_[idx].sum()!=0:
                            image_v = image_vs[:, :, :, idx]
                            label_v = label_vs[:, :, :, idx]
                            
                            w, h, d = image_v.GetSize()
                            
                            if d>16:
                                image_v = image_v[:,:, d//2-8:d//2+8]
                                label_v = label_v[:,:, d//2-8:d//2+8]
                                
                            if w>384:
                                image_v = image_v[w//8:w//8*7,:,:]
                                label_v = label_v[w//8:w//8*7,:,:]
                                
                            if h>384:
                                image_v = image_v[:,h//8:h//8*7,:]
                                label_v = label_v[:,h//8:h//8*7,:]
                                
                            image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                            image_v = convert_from_nii_to_png(image_v)
                            label_v = sitk.GetArrayFromImage(label_v) 
        
                            labels.append(label_v)
                            images.append(image_v)
                                
                            slice_id.append([client_name, patient_id, idx])
                
                val_datasets = MMSDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:', len(val_patients) ,len(images))

                images, labels, slice_id = [], [], []
                for patient_id in test_patients:
                    imgdir = os.path.join(data_path, client_name, 'images', patient_id)
                    image_vs = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, client_name, 'masks', patient_id.replace('_sa.nii.gz','_sa_gt.nii.gz'))
                    label_vs = sitk.ReadImage(maskdir)
                    label_vs_ = sitk.GetArrayFromImage(label_vs)
                    for idx in range(label_vs_.shape[0]):
                        if label_vs_[idx].sum()!=0:
                            image_v = image_vs[:, :, :, idx]
                            label_v = label_vs[:, :, :, idx]
                            
                            w, h, d = image_v.GetSize()
                            
                            if d>16:
                                image_v = image_v[:,:, d//2-8:d//2+8]
                                label_v = label_v[:,:, d//2-8:d//2+8]
                                
                            if w>384:
                                image_v = image_v[w//8:w//8*7,:,:]
                                label_v = label_v[w//8:w//8*7,:,:]
                                
                            if h>384:
                                image_v = image_v[:,h//8:h//8*7,:]
                                label_v = label_v[:,h//8:h//8*7,:]
                                
                            image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                            image_v = convert_from_nii_to_png(image_v)
                            label_v = sitk.GetArrayFromImage(label_v) 
        
                            labels.append(label_v)
                            images.append(image_v)
                                
                            slice_id.append([client_name, patient_id, idx])
                
                test_datasets = MMSDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:', len(test_patients) ,len(images))

        elif args.dataset == 'Pancreas':
            import transform3D
            input_size = (32, 256, 256)
            transform_train = transform3D.Compose3D([
                        transform3D.GaussianNoise3D(p=0.2),
                        transform3D.RandomRotate3D(p=0.2),
                        transform3D.RandomElasticDeformation3D(p=0.2),
                        transform3D.RandomBrightnessContrast3D(p=0.2),
                        transform3D.RandomFlip3D(p=0.5),
                        transform3D.RandomCrop3D(input_size),
                        transform3D.Pad3D(input_size),
                        transform3D.ToTensor3D(),
                        ])
    
            transform_test = transform3D.Compose3D([
                        #transform3D.Pad3D(input_size),
                        transform3D.ToTensor3D(),
                        ])
    
            client_names = args.client_names#{'AHN', 'NYU', 'MCA', 'NWU', 'MCF'}
            data_path = args.data_path #'/mnt/disk/meiluzhu/data/Pancreas_segmentation/t1'
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            
            patients_list = os.listdir(os.path.join(data_path, 'imagesTr'))
            print('Total patients:', len(patients_list))
            for client_name in client_names:
                patients =  []
                for patient in patients_list: 
                    if client_name in patient:
                        patients.append(patient)
                patients.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(patients)
                train_len = int(len(patients)*0.7)
                test_len = int(len(patients)*0.2)
                train_patients = patients[:train_len]
                test_patients = patients[train_len:train_len+test_len]
                val_patients = patients[train_len+test_len:]
                
                images, labels = [], []
                for patient_id in train_patients:
                    imgdir = os.path.join(data_path, 'imagesTr', patient_id)
                    image_v = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, 'labelsTr', patient_id.replace('_0000', ''))
                    label_v = sitk.ReadImage(maskdir)
                    w,h,d = image_v.GetSize()
                    
                    if d>32 and d<64:
                        image_v = image_v[:,:,d//8:d//8*7]
                        label_v = label_v[:,:,d//8:d//8*7]
                    if d>64:
                        image_v = image_v[:,:,d//4:d//4*3]
                        label_v = label_v[:,:,d//4:d//4*3]
                    
                    if w>384:
                        image_v = image_v[w//8:w//8*7,:,:]
                        label_v = label_v[w//8:w//8*7,:,:]
                        
                    if h>384:
                        image_v = image_v[:,h//8:h//8*7,:]
                        label_v = label_v[:,h//8:h//8*7,:]
                        
                    image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                    image_v = convert_from_nii_to_png(image_v)
                    label_v = sitk.GetArrayFromImage(label_v) 

                    labels.append(label_v)
                    images.append(image_v)
                
                train_datasets = PancreasDataset(args, images, labels, train_patients, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=2, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:', len(train_patients) ,len(images))
                
                images, labels = [], []
                for patient_id in val_patients:
                    imgdir = os.path.join(data_path, 'imagesTr', patient_id)
                    image_v = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, 'labelsTr', patient_id.replace('_0000', ''))
                    label_v = sitk.ReadImage(maskdir)
                    w,h,d = image_v.GetSize()
                    
                    if d>32 and d<64:
                        image_v = image_v[:,:,d//8:d//8*7]
                        label_v = label_v[:,:,d//8:d//8*7]
                    if d>64:
                        image_v = image_v[:,:,d//4:d//4*3]
                        label_v = label_v[:,:,d//4:d//4*3]
                    
                    if w>384:
                        image_v = image_v[w//8:w//8*7,:,:]
                        label_v = label_v[w//8:w//8*7,:,:]
                        
                    if h>384:
                        image_v = image_v[:,h//8:h//8*7,:]
                        label_v = label_v[:,h//8:h//8*7,:]
                        
                    image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                    image_v = convert_from_nii_to_png(image_v)
                    label_v = sitk.GetArrayFromImage(label_v) 

                    labels.append(label_v)
                    images.append(image_v)
                
                val_datasets = PancreasDataset(args, images, labels, val_patients, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:', len(val_patients) ,len(images))

                images, labels = [], []
                for patient_id in test_patients:
                    imgdir = os.path.join(data_path, 'imagesTr', patient_id)
                    image_v = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path, 'labelsTr', patient_id.replace('_0000', ''))
                    label_v = sitk.ReadImage(maskdir)
                    w,h,d = image_v.GetSize()
                    
                    if d>32 and d<64:
                        image_v = image_v[:,:,d//8:d//8*7]
                        label_v = label_v[:,:,d//8:d//8*7]
                    if d>64:
                        image_v = image_v[:,:,d//4:d//4*3]
                        label_v = label_v[:,:,d//4:d//4*3]
                    
                    if w>384:
                        image_v = image_v[w//8:w//8*7,:,:]
                        label_v = label_v[w//8:w//8*7,:,:]
                        
                    if h>384:
                        image_v = image_v[:,h//8:h//8*7,:]
                        label_v = label_v[:,h//8:h//8*7,:]
                        
                    image_v = sitk.GetArrayFromImage(image_v) #input_size[0],input_size[1],input_size[2]
                    image_v = convert_from_nii_to_png(image_v)
                    label_v = sitk.GetArrayFromImage(label_v) 

                    labels.append(label_v)
                    images.append(image_v)
                
                test_datasets = PancreasDataset(args, images, labels, test_patients, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:', len(test_patients) ,len(images))

        elif args.dataset == 'FeTS2022':
            import transform3D
            import pandas as pd
            from tqdm import tqdm
            transform_train = transform3D.Compose3D([
                        transform3D.RandomFlip3D(p=0.5),
                        transform3D.ToTensor3D(),
                        ])
    
            transform_test = transform3D.Compose3D([
                        transform3D.ToTensor3D(),
                        ])
            #https://github.com/OSUPCVLab/SegFormer3D/blob/main/data/brats2017_seg/brats2017_raw_data/brats2017_seg_preprocess.py
            client_names = args.client_names#{'site1', 'site2', 'site3', 'site10', 'site22', 'site24', 'site25', 'site26', 'site28'}
            data_path = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            #
            csv_path = os.path.join(data_path, 'partitioning_2.csv')
            df = pd.read_csv(csv_path, header=None, names=['Subject', 'Partition_ID'])
            
            for client_name in client_names:
                client_id = client_name[4:]
                patients = list(df[df['Partition_ID']==client_id]['Subject'])
                patients.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(patients)
                train_len = int(len(patients)*0.7)
                test_len = int(len(patients)*0.2)
                train_patients = patients[:train_len]
                test_patients = patients[train_len:train_len+test_len]
                val_patients = patients[train_len+test_len:]
                
                train_datasets = FeTS2022Dataset(args, None, None, train_patients, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:', len(train_patients))
                
                val_datasets = FeTS2022Dataset(args, None, None, val_patients, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=2, batch_size=self.args.batchsize, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:', len(val_patients))

                test_datasets = FeTS2022Dataset(args, None, None, test_patients, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=2, batch_size=self.args.batchsize, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:', len(test_patients))
        
        else:
            assert False



def resample_volume(image, new_spacing=None, new_size=None, interpolator=sitk.sitkLinear):
    """
    Resample image

    Args:
        image: SimpleITK image object
        new_spacing: Target voxel spacing (mm) [optional]
        new_size: Target size [pixels] [optional]
        interpolator: Interpolation method

    Returns:
        Resampled image
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    

    if new_spacing is not None and new_size is None:
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / new_spacing[2]))),
            int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
        ]
    elif new_size is not None and new_spacing is None:
        new_spacing = [
            original_spacing[0] * (original_size[0] / new_size[0]),
            original_spacing[1] * (original_size[1] / new_size[1]),
            original_spacing[2] * (original_size[2] / new_size[2])
        ]
    else:
        new_spacing = original_spacing
        new_size = original_size
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    
    return resampler.Execute(image)
