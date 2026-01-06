import os.path
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile, ImageFilter
from torchvision.datasets.folder import make_dataset
import sys
from torch.utils.data import DataLoader, random_split, Dataset, SubsetRandomSampler, SequentialSampler
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from collections import Counter
from torchvision.transforms import functional as F
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    'gaussian_blur': (gaussian_blur_rgb, gaussian_blur_depth),
    'brightness':    (brightness_rgb,    brightness_depth),
    'contrast':      (contrast_rgb,      contrast_depth),
}


def load_indices_from_file(filepath):
    """
    Loads a list of integer indices from a text file.
    """
    indices = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                #convert to a 6 digit integer and append to list
                indices.append(f"{int(line.strip()):06d}")
    except FileNotFoundError:
        print(f"Error: Index file not found at {filepath}")
        return None
    except ValueError:
        print(f"Error: Invalid content in index file. Please ensure it contains integers.")
        return None
    return indices



def assign_noise(file_indices, corruption_types, noise_severity_levels, num_modalities, setup, noise_severity=None, split = 'train'):
    data_noise_dict = {}
    # file_indices = load_indices_from_file(data_dir)
    if setup == "random_modality_random_noise_severity":
        noise_severity_levels = [2*noise for noise in noise_severity_levels]
        print(f"For each example, a random modality is corrupted with a random corruption from {corruption_types} of an intensity chosen from {noise_severity_levels}" )
        for index in file_indices:
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

        for index in file_indices:
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


class SUNRGBD_Dataset(Dataset):
    def __init__(self,  FINE_SIZE,LOAD_SIZE, data_dir=None, split="train", transforms=None, labeled=True,
                 exp_setup = "both_modalities_one_twice_as_severe", noise_severity=None):
        self.data_dir = data_dir
        self.labeled = labeled
        self.LOAD_SIZE =LOAD_SIZE
        self.FINE_SIZE = FINE_SIZE
        self.split = split
        self.transform = transforms
        self.noise_severity = noise_severity
        self.exp_setup = exp_setup

        
        if self.noise_severity:
            assert self.noise_severity in [1, 2, 3], "Noise severity must be an integer between 1 and 3 (inclusive)."


        self.imgs_folder = os.path.join(self.data_dir, 'image')
        self.depths_folder = os.path.join(self.data_dir, 'raw_depth')
        self.labels_folder = os.path.join(self.data_dir, 'scene')

        self.label_to_int = {'bathroom': 0, 'bedroom': 1, 'classroom': 2, 'computer_room': 3, 'conference_room': 4, 
                             'corridor': 5, 'dining_area': 6, 'dining_room': 7, 'discussion_area': 8, 'furniture_store': 9, 
                             'home_office': 10, 'kitchen': 11, 'lab': 12, 'lecture_theatre': 13, 'library': 14, 'living_room': 15, 
                             'office': 16, 'rest_space': 17, 'study_space': 18}

        all_file_indices = load_indices_from_file(os.path.join(self.data_dir, f'{split}_data_idx.txt'))
        
        
        self.dataset = []
        #filter out file names that don't have the label in the label_to_int mapping
        filtered_indices = []
        for file_index in all_file_indices:
            label_path = os.path.join(self.labels_folder, file_index + '.txt')
            try:
                with open(label_path, 'r') as f:
                    label = f.readline().strip()
                if label in self.label_to_int.keys():
                    self.dataset.append([file_index, self.label_to_int.get(label, -1)])
                    filtered_indices.append(file_index)
            except FileNotFoundError:
                # Handle cases where a label file might be missing
                continue
        
        # if (self.exp_setup == "random_modality_random_noise_severity") or self.noise_severity:
        self.noise_dict = assign_noise(file_indices = filtered_indices, 
                                    corruption_types = ['gaussian_blur', 'brightness', 'contrast', 'none'], 
                                    noise_severity_levels = [1, 2, 3], 
                                    num_modalities = 2, setup = self.exp_setup, noise_severity=self.noise_severity, split=self.split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.labeled:
            file_name = self.dataset[index][0]
            img_path = os.path.join(self.imgs_folder, file_name + '.jpg')
            depth_path = os.path.join(self.depths_folder, file_name + '.jpg')
            class_label = self.dataset[index][1]
        else:
            file_name = self.dataset[index][0]
            img_path = os.path.join(self.imgs_folder, file_name + '.jpg')
            depth_path = os.path.join(self.depths_folder, file_name + '.jpg')

        img_name = os.path.basename(img_path)
        
        A = Image.open(img_path).convert('RGB')
        B = Image.open(depth_path).convert('I;16')
        corr_modalities = [False, False]
        if (self.exp_setup == "random_modality_random_noise_severity") or self.noise_severity:
            rgb_mask = None
            depth_mask = None
            noise_info = self.noise_dict[file_name]
            rgb_corr, rgb_noise = noise_info['rgb']
            depth_corr, depth_noise = noise_info['depth']
            
            sample_corr = ['none', 'none']
            if rgb_corr != 'none':
                rgb_corruption_cls = CORRUPTION_MAP.get(rgb_corr)[0]
                rgb_mask = rgb_corruption_cls(severity=rgb_noise)
                sample_corr[0] = rgb_corr
                corr_modalities[0] = True

            if depth_corr != 'none':
                depth_corruption_cls = CORRUPTION_MAP.get(depth_corr)[1]
                depth_mask = depth_corruption_cls(severity = depth_noise)
                sample_corr[1] = depth_corr
                corr_modalities[1] = True

            
            rgb_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
            depth_noise_mask = Image.fromarray(np.zeros((224, 224), dtype=np.float32).astype(np.uint8))
            if rgb_mask is not None:
                A, rgb_noise_mask = rgb_mask(A)
            if depth_mask is not None:
                B, depth_noise_mask = depth_mask(B)
        else:
            rgb_noise_mask = Image.fromarray(np.zeros((224, 224, 3), dtype=np.float32).astype(np.uint8))
            depth_noise_mask = Image.fromarray(np.zeros((224, 224), dtype=np.float32).astype(np.uint8))
            sample_corr = 'none'
        
        sample = {'A': A, 'B': B, 'A_noise_mask' : rgb_noise_mask, 'B_noise_mask': depth_noise_mask}
        sample = self.transform(sample)



        if self.labeled:
            sample['label'] = class_label
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name}

        return (sample['A'], sample['B'], sample['label']), (index, sample_corr, torch.Tensor(corr_modalities)), (sample['A_noise_mask'], sample['B_noise_mask'])
    


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
        self.mean_rgb = [0.6983, 0.3918, 0.4474]
        self.std_rgb = [0.1648, 0.1359, 0.1644]
        self.mean_depth = [0.5]
        self.std_depth = [0.5]

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
    

def get_dataloader(data_dir,FINE_SIZE, LOAD_SIZE, batch_size=40, num_workers=8, train_shuffle=True, noise_severity=None, 
                   exp_setup = "both_modalities_one_twice_as_severe"):
    

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

    train_dataset = SUNRGBD_Dataset(FINE_SIZE=FINE_SIZE, LOAD_SIZE=LOAD_SIZE, data_dir=data_dir,
                                           transforms= train_transforms, split='train', exp_setup=exp_setup, noise_severity=noise_severity)
    
    val_dataset = SUNRGBD_Dataset(FINE_SIZE=FINE_SIZE, LOAD_SIZE=LOAD_SIZE, data_dir=data_dir,
                                          transforms = val_transforms, split='val', exp_setup=exp_setup, noise_severity=noise_severity)

    test_data = SUNRGBD_Dataset(FINE_SIZE=FINE_SIZE, LOAD_SIZE=LOAD_SIZE, data_dir=data_dir,
                                                transforms = val_transforms, split='test', exp_setup=exp_setup, noise_severity=noise_severity)
    # print(len(train_dataset), len(val_dataset), len(test_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = get_dataloader(data_dir="/home/pxt220000/Projects/datasets/sunrgbd_trainval", 
#                                                            FINE_SIZE=224, LOAD_SIZE=256, noise_severity=1, exp_setup = "both_modalities_one_twice_as_severe")
#     image, depth = None, None
#     samples =0
#     for (images, depths, labels), ind, noise in train_loader:
#         samples += images.size(0)
#     print(samples)