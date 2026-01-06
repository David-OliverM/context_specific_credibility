import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import io
import soundfile
from typing import Optional, Union, IO, Tuple, List, Dict
from pathlib import Path
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
import tempfile
import torchvision.transforms as T
import torch
from packages.AVROBUSTBENCH.corruptions.corruptions import corruption_dict
from collections import Counter


def assign_noise(file_json, corruption_types, noise_severity_levels, num_modalities, setup, noise_severity=None, split = 'train'):
    if setup == "random_modality_random_noise_severity":
        noise_severity_levels = [2*noise for noise in noise_severity_levels]
        print(f"For each example, a random modality is corrupted with a random corruption from {corruption_types} of an intensity chosen from {noise_severity_levels}" )
        for i, example in enumerate(file_json):
            corruption = np.random.choice(corruption_types)
            corrupt_modality = np.random.randint(0, num_modalities)
            noise_level = np.random.choice(noise_severity_levels)
            if corrupt_modality == 0:
                file_json[i]['video'] = (corruption, noise_level)
                file_json[i]['audio'] = ('none', None)
            else:
                file_json[i]['video'] = ('none', None)
                file_json[i]['audio'] = (corruption, noise_level)

    elif setup == 'both_modalities_one_twice_as_severe':
        if not noise_severity:
            print("Noise severity is set to None. No noise is being applied")
        else:
            print(f"Noise applied on both the modalities. A random corruption from {corruption_types} is applied on the data point")
            print("Applied Noise intensity is as follows:")
            if split == 'train':
                print(f"Snow -- video : {noise_severity}, audio : {2*noise_severity}")
                print(f"Frost -- video : {2*noise_severity}, audio : {noise_severity}")
                print(f"Rain -- video : {2*noise_severity}, audio : {noise_severity}")
            else:
                print(f"Snow -- video : {2* noise_severity}, audio : {noise_severity}")
                print(f"Frost -- video : {noise_severity}, audio : {2*noise_severity}")
                print(f"Rain -- video : {noise_severity}, audio : {2*noise_severity}")

        for i, example in enumerate(file_json):
            if noise_severity is not None:
                corruption = np.random.choice(corruption_types)
                if corruption == 'snow':
                    rgb_noise = noise_severity if split == 'train' else 2*noise_severity
                    depth_noise = 2*noise_severity if split == 'train' else noise_severity

                elif corruption in ['frost', 'rain']:
                    rgb_noise = 2*noise_severity if split == 'train' else noise_severity
                    depth_noise = noise_severity if split == 'train' else 2*noise_severity
                
                else:
                    rgb_noise = depth_noise = None
            else:
                corruption = 'none'
                rgb_noise = depth_noise = None

            # data_noise_dict[index] = {'corruption': corruption, 'modality': [0, 1], 'noise_level': [rgb_noise, depth_noise]}
            file_json[i]['video'] = (corruption, rgb_noise)
            file_json[i]['audio'] = (corruption, depth_noise)

    all_corruptions = [v['video'][0] if v['video'][0] is not None else v['audio'][0] for v in file_json]
    counts = Counter(all_corruptions)
    print(f"Snow: {counts.get('snow', 0)}")
    print(f"Frost:    {counts.get('frost', 0)}")
    print(f"Rain:      {counts.get('rain', 0)}")
    print(f"No noise:      {counts.get('none', 0)}")
    return file_json



# Modified from https://github.com/YuanGongND/cav-mae/blob/master/src/dataloader.py
class AVRobustBench(Dataset):
    '''
    Create an instance of the AVRobustBench dataset.

    By default, no corruption is applied and frame 4 of all frames is chosen.

    Args:
        json_file (path-like json object or file-like json object): The json file containing the metadata.
        frame_num (int): The specific frame to use and to corrupt, default is 4.
        corruption (str, optional): The corruption to apply to the image and audio, default is none.
        severity (int): The severity level of the corruption, default is 5.
        all_frames (bool, optional): Returns a list of all frames if true, default is False.
    '''

    def __init__(self, 
                 json_file: Union[str, Path, IO], 
                 frame_num: int = 4, 
                 corruptions: Optional[List[str]] = None, 
                 severity: int = 5,
                 all_frames: Optional[bool] = False,
                 split="train",
                 exp_setup = "both_modalities_one_twice_as_severe") -> None:
        
        self.datapath = json_file
        self.frame_num = frame_num
        self.corruption = list(corruptions)
        self.severity = severity
        self.all_frames = all_frames
        self.split = split
        self.exp_setup = exp_setup
        

        self.preprocess = T.Compose([
            T.Resize(224, interpolation=Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])
        
        
        # self.audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
        #       'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}

        self.target_length = 1024
        self.freqm = 0
        self.timem = 0
        self.skip_norm = False
        self.norm_mean = -5.081
        self.norm_std = 4.4849
        self.num_mel_bins = 128

        if self.severity is not None:
            for corruption in self.corruption:
                assert corruption in corruption_dict, f"{corruption!r} is not a valid corruption"
            # self.visual_corruption, self.audio_corruption = corruption_dict[self.corruption]


        with open(self.datapath, 'r') as f:
            data_json = json.load(f)
        
        self.corruption.append('none')

        self.data = data_json['data']
        self.data = assign_noise(file_json=self.data, 
                                    corruption_types = self.corruption, 
                                    noise_severity_levels = [1, 2, 3], 
                                    num_modalities = 2, setup = self.exp_setup, noise_severity=self.severity, split=self.split)
        self.data = self.process_data(self.data)

    def process_data(self, data_json: List[Dict[str, str]]) -> np.ndarray:
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], 
                            data_json[i]['labels'], 
                            data_json[i]['video_id'], 
                            data_json[i]['video_path'],
                            data_json[i]['video'],
                            data_json[i]['audio']]
            
        # data_np = np.array(data_json, dtype=str)

        return data_json

    def decode_data(self, np_data: np.ndarray) -> dict[str, str]:
        datum = {}

        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        datum['video_corr'] = np_data[4][0]
        datum['video_noise_level'] = np_data[4][1]
        datum['audio_corr'] = np_data[5][0]
        datum['audio_noise_level'] = np_data[5][1]

        return datum

    def get_image(self, filename: str, visual_corruption, severity) -> Image.Image:
        image = Image.open(filename)
        
        if visual_corruption != 'none':
            image, corr_mask = visual_corruption(image, severity=severity)
            image_tensor = self.preprocess(image)
            if isinstance(corr_mask, np.ndarray):
                corr_mask = torch.from_numpy(corr_mask)

            # If mask has channels, average them
            if corr_mask.ndim == 3:
                corr_mask = corr_mask.mean(dim=-1)

            # Convert mask to float
            corr_mask = corr_mask.float()

            # Resize mask to match image (224x224)
            corr_mask = corr_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            corr_mask = torch.nn.functional.interpolate(corr_mask, size=(224, 224), mode='nearest')
            corr_mask = corr_mask.squeeze()
            corr_mask = corr_mask.unsqueeze(0).repeat(3, 1, 1)
        else: 
            image_tensor = self.preprocess(image)
            corr_mask = torch.zeros_like(image_tensor)
        return image_tensor, corr_mask
    
    

    def get_wav(self, filename, audio_corruption, severity):
        waveform, sr = torchaudio.load(filename)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        corr_mask = torch.zeros_like(waveform)

        if audio_corruption != 'none':
            waveform, corr_mask = audio_corruption(waveform=waveform, intensity=severity)

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10
        )

        # waveform mask → frame mask
        frame_shift_samples = int(sr * 0.01)
        num_frames = fbank.shape[0]

        mask_frames = []
        for i in range(num_frames):
            start = i * frame_shift_samples
            end = start + frame_shift_samples
            mask_frames.append(corr_mask[:, start:end].mean())

        mask_frames = torch.stack(mask_frames).squeeze()
        audio_corr_mask = mask_frames[:, None].repeat(1, fbank.shape[1])

        # pad / trim
        p = self.target_length - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
            audio_corr_mask = torch.nn.functional.pad(audio_corr_mask, (0, 0, 0, p))
        else:
            fbank = fbank[:self.target_length]
            audio_corr_mask = audio_corr_mask[:self.target_length]

        # SpecAugment
        fbank = fbank.T.unsqueeze(0)
        audio_corr_mask = audio_corr_mask.T.unsqueeze(0)

        if self.timem != 0:
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = timem(fbank)
            audio_corr_mask = timem(audio_corr_mask)

        if self.freqm != 0:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            fbank = freqm(fbank)

        fbank = fbank.squeeze(0).T
        audio_corr_mask = audio_corr_mask.squeeze(0).T

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / self.norm_std

        return fbank, audio_corr_mask

    

    def __len__(self) -> int:
        # return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[Image.Image], IO[bytes]]:
        datum = self.data[index]
        datum = self.decode_data(datum) 

        label = int(datum['labels'])
        audio_corr = datum['audio_corr']
        visual_corr = datum['video_corr']
        audio_path = str(datum['wav'])

        corruption = audio_corruption = visual_corruption = 'none'
        if audio_corr != 'none':
            audio_corruption = corruption_dict[audio_corr][1]
            corruption = audio_corr

        if visual_corr != 'none':
            visual_corruption = corruption_dict[visual_corr][0]
            corruption = visual_corr


        
        audio, audio_corr_mask = self.get_wav(audio_path, audio_corruption, datum['audio_noise_level'])

        video_path = str(datum['video_path'])
        video_id = str(datum['video_id'])

        frames = []
        vis_corr_frames = []

        if not self.all_frames:
            image_path = video_path + f'/frame_{self.frame_num}/' + video_id + '.jpg'
            image, image_corr = self.get_image(image_path, visual_corruption, datum['video_noise_level'])
            frames.append(image)
            vis_corr_frames.append(image_corr)
        else:
            for i, frame in enumerate(os.listdir(video_path)):    
                image_path = video_path + f'/{frame}/' + video_id + '.jpg'
                image, image_corr = self.get_image(image_path, visual_corruption, datum['video_noise_level'])
                frames.append(image)
                vis_corr_frames.append(image_corr)
            video = torch.stack(frames, dim=1)
            video_corr = torch.stack(vis_corr_frames, dim=1)

        return (video, audio, label), (index, corruption), (video_corr, audio_corr_mask)
    
    @staticmethod
    def create_video(video_path: str, 
                     corruption: str = 'gaussian', 
                     severity: int = 5, 
                     duration: Optional[float] = None, 
                     save_path: Optional[str] = None) -> io.BytesIO:
        """
        Apply visual/audio corruption to an MP4 and return it as a BytesIO with the option to save. 
        
        Default corruption is gaussian.

        Args:
            video_path (str): Path to the source .mp4 file.
            corruption (str): Name of the corruption.
            severity (int): Corruption severity.
            duration (int, optional): Seconds from start to process (None for full length).
            save_path (str, optional): If provided, also write the output to this path.

        Returns:
            A BytesIO containing the corrupted MP4 data.
        """

        assert corruption in corruption_dict, f"{corruption} is not a corruption"
        visual_corruption, audio_corruption = corruption_dict[corruption]


        clip = VideoFileClip(video_path)
        if duration is not None:
            clip = clip.subclip(0, duration)
        fps = clip.fps

        frames = []
        for frame in clip.iter_frames(fps=fps, dtype="uint8"):
            image = Image.fromarray(frame)
            image = visual_corruption(x=image, severity=severity)
            frames.append(np.array(image))

        corrupted_video = ImageSequenceClip(frames, fps=fps)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            clip.audio.write_audiofile(tmp_audio.name, logger=None, verbose=False)
            waveform, sr = torchaudio.load(tmp_audio.name)
            waveform = audio_corruption(waveform=waveform, intensity=severity)
            torchaudio.save(tmp_audio.name, waveform, sr)

        corrupted_audio = AudioFileClip(tmp_audio.name)
        corrupted_video = corrupted_video.set_audio(corrupted_audio)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            corrupted_video.write_videofile(tmp_vid.name, codec="libx264", audio_codec="aac", logger=None, verbose=False)
            tmp_vid_path = tmp_vid.name

        buf = io.BytesIO()
        with open(tmp_vid_path, 'rb') as f:
            buf.write(f.read())
        buf.seek(0)

        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())

        os.remove(tmp_audio.name)
        os.remove(tmp_vid_path)

        return buf
    



def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True, noise_severity=None, exp_setup="both_modalities_one_twice_as_severe"):
    train_file = data_dir+"/train_data.json"
    val_file = data_dir+"/val_data.json"
    test_file = data_dir+"/test_data.json"
    corruptions = ['snow', 'frost', 'rain']
    train_dataset = AVRobustBench(train_file, corruptions=corruptions, severity=noise_severity, frame_num=4, all_frames=True, split='train', exp_setup=exp_setup)
    val_dataset = AVRobustBench(val_file, corruptions=corruptions, severity=noise_severity, frame_num=4, all_frames=True, split='val', exp_setup=exp_setup)
    test_dataset = AVRobustBench(test_file, corruptions=corruptions, severity=noise_severity, frame_num=4, all_frames=True, split='test', exp_setup=exp_setup)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader(data_dir="/home/pxt220000/Projects/datasets/avrobustbench/kinetics", noise_severity=None)
    image, audio = None, None
    samples =0
    # for (images, depths, labels), ind, noise in test_loader:
    #     samples += images.size(0)
    # print(samples)
    cls=0
    for (frames, audio, label), (ind, corr), (vid_mask, aud_mask) in train_loader:
        cls = max(cls, max(label))
    print(cls)



