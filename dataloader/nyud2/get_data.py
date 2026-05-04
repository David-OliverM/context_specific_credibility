import os.path
import random

import torchvision.transforms as transforms
from PIL import Image,ImageFilter
from PIL import ImageFile
from torchvision.datasets.folder import make_dataset
import sys
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
from collections import Counter
from torchvision.transforms import functional as F
import copy
import skimage as sk
from skimage import filters as sk_filters
from collections import Counter


class contrast_rgb(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.c = [0.4, 0.3, 0.2, 0.1, 0.05, 0.01][severity - 1]

    def __call__(self, rgb_image):
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

        mean = rgb.mean(axis=(0, 1), keepdims=True)
        corrupted = np.clip((rgb - mean) * self.c + mean, 0.0, 1.0)

        # corruption mask in [0,1]
        mask = np.abs(corrupted - rgb).astype(np.float32)

        # return image (uint8) + mask (float)
        corrupted_img = (corrupted * 255).astype(np.uint8)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8))



class contrast_depth(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.c = [0.4, 0.3, 0.2, 0.1, 0.05, 0.01][severity - 1]

    def __call__(self, depth_image):
        # depth as float32
        depth = np.asarray(depth_image, dtype=np.float32)

        mean = depth.mean(keepdims=True)
        corrupted = (depth - mean) * self.c + mean

        # corruption magnitude (same units as depth)
        mask = np.abs(corrupted - depth).astype(np.float32)

        # return corrupted depth + mask
        corrupted_img = corrupted.astype(np.uint16)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8), mode='L')


class brightness_rgb(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][severity - 1]

    def __call__(self, rgb_image):
        # RGB in [0,1]
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

        hsv = sk.color.rgb2hsv(rgb)
        hsv[..., 2] = np.clip(hsv[..., 2] + self.c, 0.0, 1.0)
        corrupted = sk.color.hsv2rgb(hsv)

        # corruption magnitude (bounded by c)
        mask = np.abs(corrupted - rgb).astype(np.float32)

        # optional normalization to [0,1]
        mask = mask / self.c
        mask = np.clip(mask, 0.0, 1.0)

        corrupted_img = (corrupted * 255).astype(np.uint8)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8))



class brightness_depth(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][severity - 1]

    def __call__(self, depth_image):
        depth = np.asarray(depth_image, dtype=np.float32)

        max_depth = depth.max()
        corrupted = depth + self.c * max_depth

        # corruption magnitude (constant per pixel)
        mask = np.abs(corrupted - depth).astype(np.float32)

        # normalize to [0,1] using known bound
        mask = mask / (self.c * max_depth + 1e-6)
        mask = np.clip(mask, 0.0, 1.0)

        corrupted_img = corrupted.astype(np.uint16)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8), mode='L')
    

class gaussian_blur_rgb(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.sigma = [1, 2, 3, 4, 5, 6][severity - 1]

    def __call__(self, rgb_image):
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
        corrupted = sk_filters.gaussian(rgb, sigma=self.sigma, channel_axis=-1)

        # corruption magnitude per pixel
        mask = np.abs(corrupted - rgb).astype(np.float32)
        # normalize mask to [0,1]
        mask = mask / (mask.max() + 1e-6)

        corrupted_img = (np.clip(corrupted, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8))
    

class gaussian_blur_depth(object):
    def __init__(self, severity=1):
        self.severity = severity
        self.sigma = [1, 2, 3, 4, 5, 6][severity - 1]

    def __call__(self, depth_image):
        depth = np.asarray(depth_image, dtype=np.float32)
        corrupted = sk_filters.gaussian(depth, sigma=self.sigma)

        # corruption magnitude per pixel
        mask = np.abs(corrupted - depth).astype(np.float32)
        # normalize mask to [0,1] based on max possible change
        mask = mask / (mask.max() + 1e-6)

        corrupted_img = np.clip(corrupted, 0, 65535).astype(np.uint16)
        return Image.fromarray(corrupted_img), Image.fromarray((mask*255).astype(np.uint8), mode='L')


CORRUPTION_MAP = {
    'gaussian_blur': (gaussian_blur_rgb, gaussian_blur_rgb),
    'brightness':    (brightness_rgb,    brightness_rgb),
    'contrast':      (contrast_rgb,      contrast_rgb),
}



def assign_noise(file_indices, corruption_types, noise_severity_levels, num_modalities, setup, noise_severity=None, split = 'train'):
    data_noise_dict = {}
    # file_indices = load_indices_from_file(data_dir)
    if setup == "random_modality_random_noise_severity":
        noise_severity_levels = [2*noise for noise in noise_severity_levels]
        print(f"For each example, a random modality is corrupted with a random corruption from {corruption_types} of an intensity chosen from {noise_severity_levels}" )
        for index in range(len(file_indices)):
            corruption = np.random.choice(corruption_types)
            corrupt_modality = np.random.randint(0, num_modalities)
            noise_level = np.random.choice(noise_severity_levels)
            # data_noise_dict[index] = {'corruption': corruption, 'modality': corrupt_modality, 'noise_level': noise_level}
            if corrupt_modality == 0:
                data_noise_dict[index] = {'rgb': (corruption, noise_level), 'depth': ('none', None)}
            else:
                data_noise_dict[index] = {'rgb': ('none', None), 'depth': (corruption, noise_level)}

    
    elif setup == "both_modalities_one_twice_as_severe":
        if not noise_severity:
            print("Noise severity is set to None. No noise is being applied")
        else:
            print(f"Noise applied on both the modalities. A random corruption from {corruption_types} is applied on the data point")
            print("Applied Noise intensity is as follows:")
            if split == 'train':
                print(f"Gaussian_blur -- RGB : {noise_severity}, depth : {2*noise_severity}")
                print(f"Brightness -- RGB : {2*noise_severity}, depth : {noise_severity}")
                print(f"Contrast -- RGB : {2*noise_severity}, depth : {noise_severity}")
            else:
                print(f"Gaussian_blur -- RGB : {2* noise_severity}, depth : {noise_severity}")
                print(f"Brightness -- RGB : {noise_severity}, depth : {2*noise_severity}")
                print(f"Contrast -- RGB : {noise_severity}, depth : {2*noise_severity}")

        for index in range(len(file_indices)):
            if noise_severity is not None:
                corruption = np.random.choice(corruption_types)
                if corruption == 'gaussian_blur':
                    rgb_noise = noise_severity if split == 'train' else 2*noise_severity
                    depth_noise = 2*noise_severity if split == 'train' else noise_severity

                elif corruption in ['brightness', 'contrast']:
                    rgb_noise = 2*noise_severity if split == 'train' else noise_severity
                    depth_noise = noise_severity if split == 'train' else 2*noise_severity
                
                else:
                    rgb_noise = depth_noise = None
            else:
                corruption = 'none'
                rgb_noise = depth_noise = None

            # data_noise_dict[index] = {'corruption': corruption, 'modality': [0, 1], 'noise_level': [rgb_noise, depth_noise]}
            data_noise_dict[index] = {'rgb':(corruption, rgb_noise), 'depth':(corruption, depth_noise)}

    all_corruptions = [v['rgb'][0] if v['rgb'][0] is not None else v['depth'][0] for v in data_noise_dict.values()]
    counts = Counter(all_corruptions)
    print(f"Gaussian Blur: {counts.get('gaussian_blur', 0)}")
    print(f"Brightness:    {counts.get('brightness', 0)}")
    print(f"Contrast:      {counts.get('contrast', 0)}")
    print(f"No noise:      {counts.get('none', 0)}")
    return data_noise_dict



class AlignedConcNoisyDataset:

    def __init__(self, FINE_SIZE,LOAD_SIZE,  data_dir=None, transforms=None, labeled=True, split="train",
                 exp_setup= "both_modalities_one_twice_as_severe", noise_severity=None):
        self.transform = transforms
        self.data_dir = data_dir
        self.labeled = labeled
        self.LOAD_SIZE =LOAD_SIZE
        self.FINE_SIZE = FINE_SIZE
        self.split=split
        self.exp_setup = exp_setup
        self.noise_severity = noise_severity
        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')
        self.img_corr_cls = [0, 5, 7]
        self.depth_corr_cls = [2, 6, 9]
        # self.swap = 0.75 # define test set noise ratio here
        # if self.noise_severity:
        #     if self.split == 'train':
        #         self.swap = 0.7
        # else:
        #     if self.split == 'train':
        #         self.swap = 0.0

        self.swap = self.noise_severity
        
        print(f"Replacing {self.swap} of the {split} rgb samples belonging to {self.img_corr_cls} with that of {self.depth_corr_cls} respectively")
        print(f"Reaplcing {self.swap} of the {split} depth samples belonging to {self.depth_corr_cls} with that of {self.img_corr_cls} respectively")
        

        # self.noise_dict = assign_noise(file_indices = self.imgs, 
        #                             corruption_types = ['gaussian_blur', 'brightness', 'contrast', 'none'], 
        #                             noise_severity_levels = [1, 2, 3], 
        #                             num_modalities = 2, setup = self.exp_setup, noise_severity=self.noise_severity, split=self.split)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        corr_modalities = [False, False]
        rgb_noise_mask = np.zeros((224, 224, 3), dtype=np.float32)
        depth_noise_mask = np.zeros((224, 224, 3), dtype=np.float32)
        sample_corr = ['none', 'none']
        if self.noise_severity:
            sample_corr = ['none', 'none']
            ar = np.array(np.array(self.imgs)[:, 1], dtype=np.int16)
            if label in self.img_corr_cls:
                corrupt_img = np.random.rand()
                if corrupt_img < self.swap:
                    tmp = np.array(A).copy()
                    corruption = np.random.choice(np.where(ar == self.depth_corr_cls[self.img_corr_cls.index(label)])[0])
                    donor, _ = self.imgs[corruption]
                    # print("img ", donor)
                    donor_img = Image.open(donor).convert('RGB')
                    w, h = donor_img.size
                    w2 = int(w / 2)
                    if w2 > self.FINE_SIZE:
                        A = donor_img.crop((0, 0, w2, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
                    else:
                        A = donor_img.crop((0, 0, w2, h))
                    # print(np.array(A).shape, tmp.shape)
                    rgb_noise_mask = np.array(A) - tmp
                    depth_noise_mask = np.zeros((224, 224, 3), dtype=np.float32)
                    corr_modalities[0] = True
                    sample_corr[0] = 'replace'

            if label in self.depth_corr_cls:
                corrupt_depth = np.random.rand()
                if corrupt_depth < self.swap:
                    tmp = np.array(B).copy()
                    corruption = np.random.choice(np.where(ar == self.img_corr_cls[self.depth_corr_cls.index(label)])[0])
                    donor, _ = self.imgs[corruption]
                    # print("depth  ",donor)
                    donor_img = Image.open(donor).convert('RGB')
                    w, h = donor_img.size
                    w2 = int(w / 2)
                    if w2 > self.FINE_SIZE:
                        B = donor_img.crop((w2, 0, w, h)).resize((self.LOAD_SIZE, self.LOAD_SIZE), Image.BICUBIC)
                    else:
                        B = donor_img.crop((w2, 0, w, h))
                    # print(np.array(B).shape, tmp.shape)
                    depth_noise_mask = np.array(B) - tmp
                    rgb_noise_mask = np.zeros((224, 224, 3), dtype=np.float32)
                    corr_modalities[1] = True
                    sample_corr[1] = 'replace'

            
        rgb_noise_mask = Image.fromarray(rgb_noise_mask.astype(np.uint8))
        depth_noise_mask = Image.fromarray(depth_noise_mask.astype(np.uint8))

                


        
        
        # if (self.exp_setup == "random_modality_random_noise_severity") or self.noise_severity:
        #     rgb_mask = None
        #     depth_mask = None
        #     noise_info = self.noise_dict[index]
        #     rgb_corr, rgb_noise = noise_info['rgb']
        #     depth_corr, depth_noise = noise_info['depth']
            
        #     sample_corr = ['none', 'none']
        #     if rgb_corr != 'none':
        #         rgb_corruption_cls = CORRUPTION_MAP.get(rgb_corr)[0]
        #         rgb_mask = rgb_corruption_cls(severity=rgb_noise)
        #         sample_corr[0] = rgb_corr
        #         corr_modalities[0] = True

        #     if depth_corr != 'none':
        #         depth_corruption_cls = CORRUPTION_MAP.get(depth_corr)[1]
        #         depth_mask = depth_corruption_cls(severity = depth_noise)
        #         sample_corr[1] = depth_corr
        #         corr_modalities[1] = True
                
                
        #     rgb_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
        #     depth_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
        #     if rgb_mask is not None:
        #         A, rgb_noise_mask = rgb_mask(A)
        #     if depth_mask is not None:
        #         B, depth_noise_mask = depth_mask(B)
        # else:
        #     rgb_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
        #     depth_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
        #     sample_corr = ['none', 'none']
        
        sample = {'A': A, 'B': B, 'A_noise_mask' : rgb_noise_mask, 'B_noise_mask': depth_noise_mask}
        sample = self.transform(sample)



        if self.labeled:
            # sample = {'A': A, 'B': B, 'img_name': img_name, 'label': class_label, 'A_noise_mask' : rgb_noise_mask, 'B_noise_mask': depth_noise_mask}
            sample['label'] = label
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name}

        return (sample['A'], sample['B'], sample['label']), (index, sample_corr, torch.tensor(corr_modalities)), (sample['A_noise_mask'], sample['B_noise_mask'])


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding and self.padding> 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        sample['A_noise_mask'] = F.crop(sample['A_noise_mask'], i, j, h, w)
        sample['B_noise_mask'] = F.crop(sample['B_noise_mask'], i, j, h, w)

        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A, B, A_noise_mask, B_noise_mask= sample['A'], sample['B'], sample['A_noise_mask'], sample['B_noise_mask']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)
            A_noise_mask = F.hflip(A_noise_mask)
            B_noise_mask = F.hflip(B_noise_mask)


        sample['A'] = A
        sample['B'] = B
        sample['A_noise_mask'] = A_noise_mask
        sample['B_noise_mask'] = B_noise_mask
        return sample



class Resize(transforms.Resize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))
        sample['A_noise_mask'] = F.resize(sample['A_noise_mask'], (h, w))
        sample['B_noise_mask'] = F.resize(sample['B_noise_mask'], (h, w))

        return sample


class ToTensor(object):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        # if isinstance(sample, dict):
        #     for key, value in sample:
        #         _list = sample[key]
        #         sample[key] = [F.to_tensor(item) for item in _list]

        sample['A'] = F.to_tensor(A)
        sample['B'] = F.to_tensor(B)
        sample['A_noise_mask'] = F.to_tensor(sample['A_noise_mask'])
        sample['B_noise_mask'] = F.to_tensor(sample['B_noise_mask'])

        return sample



class Normalize:
    def __init__(self):
        self.mean_rgb = self.mean_depth = [0.4951, 0.3601, 0.4587]
        self.std_rgb = self.std_depth = [0.1474, 0.1950, 0.1646]

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.normalize(A, self.mean_rgb, self.std_rgb)
        sample['B'] = F.normalize(B, self.mean_depth, self.std_depth)
        return sample
    
class ToFloat:
    def __call__(self, sample):
        sample['A'] = sample['A'].float()
        sample['B'] = sample['B'].float()
        return sample
    

def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def get_dataloader(data_dir,FINE_SIZE, LOAD_SIZE, batch_size=40, num_workers=8, train_shuffle=True, noise_severity=None, test_noise = 0.0, exp_setup = "both_modalities_one_twice_as_severe"):
    train_transforms = transforms.Compose([
        Resize((LOAD_SIZE, LOAD_SIZE)),
        RandomCrop((FINE_SIZE, FINE_SIZE)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToFloat(),
        Normalize()
    ])

    val_transforms = transforms.Compose([
        Resize((FINE_SIZE, FINE_SIZE)),
        ToTensor(),
        ToFloat(),
        Normalize()
    ])

    noise = 0.7 if noise_severity else 0

    
    test_noise = test_noise # Modify this to set test noise
    train_set = AlignedConcNoisyDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'train'), 
                                                transforms= train_transforms, split='train', exp_setup=exp_setup, noise_severity=noise)
    val_set = AlignedConcNoisyDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'val'), 
                                                transforms= train_transforms, split='val', exp_setup=exp_setup, noise_severity=noise)
    test_set = AlignedConcNoisyDataset(FINE_SIZE = FINE_SIZE, LOAD_SIZE = LOAD_SIZE, data_dir=os.path.join(data_dir, 'test'), 
                                                transforms= train_transforms, split='test', exp_setup=exp_setup, noise_severity=test_noise)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader(data_dir="/nas1-nfs1/home/pxt220000/projects/datasets/nyud2/nyud2_trainvaltest", FINE_SIZE=224, LOAD_SIZE=256, noise_severity=None, test_noise=0.0, exp_setup = "both_modalities_one_twice_as_severe")
    image, depth = None, None
    for images, depths, labels in val_loader:
        image = images[0]
        depth = depths[0]
        label = labels[0]
        break