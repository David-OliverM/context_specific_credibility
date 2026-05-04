import random
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from torch.utils.data import Dataset, DataLoader, Subset
from packages.torchfsdd.lib.torchfsdd.dataset import TorchFSDDGenerator
from packages.torchfsdd.lib.torchfsdd.helpers import TrimSilence
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose, Resize
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle

def load_fsdd():
    # Set number of features and classes
    n_mfcc = 112
    n_digits = 10

    # Specify transformations to be applied to the raw audio
    transforms = Compose([
        # Trim silence from the start and end of the audio
        TrimSilence(threshold=1e-6),
        # Generate n_mfcc+1 MFCCs (and remove the first one since it is a constant offset)
        lambda audio: MFCC(sample_rate=8e3, n_mfcc=n_mfcc+1)(audio)[1:, :],
        # Standardize MFCCs for each frame
        lambda mfcc: (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-6),
        # Transpose from DxT to TxD
        lambda mfcc: mfcc.transpose(1, 0),
        # Resize into 28x28
        lambda mfcc: Resize(size=(112,112), antialias=True)(mfcc.unsqueeze(0)).squeeze(),
    ])
    # Initialize a generator for a local version of FSDD
    fsdd = TorchFSDDGenerator(version='local', path='/nas1-nfs1/home/pxt220000/projects/CS_Credibility/packages/torchfsdd/lib/test/data/v1.0.10', transforms=transforms, load_all=True)

    # Create two Torch datasets for a train-test split from the generator
    train_set, val_set, test_set = fsdd.train_val_test_split(test_size=0.1, val_size=0.1)
    return train_set, val_set, test_set


class AV_dataset(Dataset):
    def __init__(self, data_dir, mnist_dataset, audio_dataset, noise_severity = 0, split='train'):
        self.mnist_dataset = mnist_dataset
        self.audio_dataset = audio_dataset
        self.noise_severity = noise_severity
        self.class_visual = [5, 8, 1] 
        self.class_audio = [6, 0, 7]
        self.paired_samples = []
        if self.noise_severity:
            print(f"Replacing {noise_severity*100}% of the {split} images belonging to {self.class_visual} with that of {self.class_audio} respectively")
            print(f"Replacing {noise_severity*100}% of the {split} audios belonging to {self.class_audio} with that of {self.class_visual} respectively")
        else:
            print(f"Noise severity is set to None for {split}.")


        if os.path.exists(os.path.join(data_dir, f'{split}_samples.pkl')):
            with open(os.path.join(data_dir, f'{split}_samples.pkl'), 'rb') as f:
                self.paired_samples = pickle.load(f)
            print("loaded from folder")

        else:
            # Group MNIST samples by label
            self.visual_dict = {}
            for idx in range(len(self.mnist_dataset)):
                image, label = self.mnist_dataset[idx]
                if label not in self.visual_dict:
                    self.visual_dict[label] = []
                self.visual_dict[label].append((image, label))

            # Group spoken digit samples by label
            self.audio_dict = {}
            for idx in range(len(self.audio_dataset)):
                audio, label = self.audio_dataset[idx]
                audio = torch.unsqueeze(audio, 0)
                if label not in self.audio_dict:
                    self.audio_dict[label] = []
                self.audio_dict[label].append((audio, label))

            # Ensure that each label has at least one sample in both datasets
            # print(self.visual_dict.keys())
            # print(self.audio_dict.keys())
            common_labels = set(self.visual_dict.keys()) & set(self.audio_dict.keys())
            assert len(common_labels) > 0, "No common labels found between MNIST and spoken digit datasets"

            # Pair samples with the same label
            
            for label in common_labels:
                mnist_samples = self.visual_dict[label]
                spoken_digit_samples = self.audio_dict[label]
                for mnist_sample in mnist_samples:
                    # Randomly select a spoken digit sample with the same label
                    spoken_digit_sample = random.choice(spoken_digit_samples)
                    self.paired_samples.append((mnist_sample[0], spoken_digit_sample[0], label))

            # Save paired samples to a .pkl file
            with open(os.path.join(data_dir, f'{split}_samples.pkl'), 'wb') as f:
                pickle.dump(self.paired_samples, f)
                


    def __len__(self):
        return len(self.paired_samples)


    def __getitem__(self, idx):
        image, audio, label = self.paired_samples[idx]
        img_noise_mask = torch.zeros_like(image)
        aud_noise_mask = torch.zeros_like(audio)
        img_corr = 'none'
        audio_corr = 'none'
        corr_modalities = [False, False]
        if self.noise_severity:
            if label in self.class_visual:
                corrupt_img = np.random.rand()
                if corrupt_img < self.noise_severity:
                    tmp = image.clone()
                    donor_label = self.class_audio[self.class_visual.index(label)]
                    donor_idxs = [i for i, sample in enumerate(self.paired_samples) if sample[2] == donor_label]
                    corruption = random.choice(donor_idxs)
                    image = self.paired_samples[corruption][0]
                    img_noise_mask = image - tmp
                    img_corr = "swap"
                    corr_modalities[0] = True
            
            if label in self.class_audio:
                corrupt_aud = np.random.rand()
                if corrupt_aud < self.noise_severity:
                    tmp = audio.clone()
                    donor_label = self.class_visual[self.class_audio.index(label)]
                    donor_idxs = [i for i, sample in enumerate(self.paired_samples) if sample[2] == donor_label]
                    corruption = random.choice(donor_idxs)
                    audio = self.paired_samples[corruption][1]
                    aud_noise_mask = audio - tmp
                    audio_corr = "swap"
                    corr_modalities[1] = True

        return (image, audio, label), (idx, (img_corr, audio_corr), torch.tensor(corr_modalities)), (img_noise_mask, aud_noise_mask)
    




def get_dataloader(data_dir, batch_size=40, num_workers=8, noise_severity=None, test_noise=0.0, exp_setup = "both_modalities_one_twice_as_severe"):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    indices = list(range(60000))
    train_indices = indices[:55000]
    val_indices = indices[55000:]
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    v_train = Subset(train_dataset, train_indices)
    v_val = Subset(train_dataset, val_indices)
    v_test = datasets.MNIST('data', train=False, transform=transform)
    a_train, a_val, a_test = load_fsdd()

    # Create a multimodal dataset instance and its DataLoader
    if noise_severity:
        noise = 0.7
    else:
        noise = 0
    AV_trainset = AV_dataset(data_dir, v_train, a_train, noise, split='train')
    AV_valset = AV_dataset(data_dir, v_val, a_val, noise, split= 'val')
    test_noise = test_noise #Adjust this for setting test noise
    AV_testset = AV_dataset(data_dir, v_test, a_test, test_noise, split = 'test')
    # print(len(AV_trainset), " ", len(AV_valset), " ", len(AV_testset))
    AV_train = DataLoader(AV_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    AV_val = DataLoader(AV_valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    AV_test = DataLoader(AV_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("......Data loaded..........")

    
    return AV_train, AV_val, AV_test


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(data_dir="/nas1-nfs1/home/pxt220000/projects/datasets/clean_avmnist", noise_severity=1, test_noise=0.0)
    for (image, audio, label), _, _ in test_loader:
        print(image.shape)