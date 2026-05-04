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
                 corruption: Optional[str] = None, 
                 severity: int = 5,
                 all_frames: Optional[bool] = False) -> None:
        
        self.datapath = json_file
        self.frame_num = frame_num
        self.corruption = corruption
        self.severity = severity
        self.all_frames = all_frames

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

        if self.corruption is not None:
            assert self.corruption in corruption_dict, f"{self.corruption!r} is not a valid corruption"
            self.visual_corruption, self.audio_corruption = corruption_dict[self.corruption]


        with open(self.datapath, 'r') as f:
            data_json = json.load(f)

        self.data = data_json['data']
        self.data = self.process_data(self.data)

    def process_data(self, data_json: List[Dict[str, str]]) -> np.ndarray:
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], 
                            data_json[i]['labels'], 
                            data_json[i]['video_id'], 
                            data_json[i]['video_path']]
            
        data_np = np.array(data_json, dtype=str)

        return data_np

    def decode_data(self, np_data: np.ndarray) -> dict[str, str]:
        datum = {}

        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]

        return datum

    def get_image(self, filename: str) -> Image.Image:
        image = Image.open(filename)

        if self.corruption is not None:
            image = self.visual_corruption(image, severity=self.severity)

        image_tensor = self.preprocess(image)
        return image_tensor
    
    # def get_image(self, filename, filename2=None, mix_lambda=1):
    #     if filename2 == None:
    #         img = Image.open(filename)
    #         image_tensor = self.preprocess(img)
    #         return image_tensor
    #     else:
    #         img1 = Image.open(filename)
    #         image_tensor1 = self.preprocess(img1)

    #         img2 = Image.open(filename2)
    #         image_tensor2 = self.preprocess(img2)

    #         image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
    #         return image_tensor

    # def get_wav(self, filename: str) -> IO[bytes]:
    #     waveform, sr = torchaudio.load(filename)

    #     if self.corruption is not None:
    #         waveform = self.audio_corruption(waveform=waveform, intensity=self.severity)

    #     waveform = waveform.numpy().T

    #     buffer = io.BytesIO()
    #     soundfile.write(buffer, waveform, sr, format='WAV')
    #     buffer.seek(0)

    #     return buffer
    

    def get_wav(self, filename: str) -> torch.Tensor:
        # Load the audio file as a waveform
        waveform, sr = torchaudio.load(filename)

        # Apply any corruption if necessary
        if self.corruption is not None:
            waveform = self.audio_corruption(waveform=waveform, intensity=self.severity)

        # Normalize the waveform
        waveform = waveform - waveform.mean()

        # Extract fbank (Mel-frequency cepstral coefficients) or other features
        try:
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
        except Exception as e:
            print(f"Error while extracting features: {e}")
            # Fallback in case of error
            fbank = torch.zeros([512, 128]) + 0.01
            print('There is a loading error when extracting features.')

        # Pad or trim fbank to match the target length
        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # Cut or pad to target length
        if p > 0:
            # Padding the fbank
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            # Trimming the fbank
            fbank = fbank[0:target_length, :]

        # Apply masking if required
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        
        # Apply transformations
        fbank = torch.transpose(fbank, 0, 1)  # Transpose to (time, features)
        fbank = fbank.unsqueeze(0)  # Add batch dimension

        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)

        fbank = fbank.squeeze(0)  # Remove batch dimension
        fbank = torch.transpose(fbank, 0, 1)  # Transpose back to (features, time)

        # Normalize the input if required
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / self.norm_std
        
        return fbank
    
    def get_wav2mel(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            n_mels=self.num_mel_bins,   # replaces num_mel_bins
            center=True,
            power=2.0
        )

        mel = mel_transform(waveform)          # (1, n_mels, time)
        mel = mel.squeeze(0)                   # (n_mels, time)
        mel = torch.log(mel + 1e-10)

        # transpose to (time, freq)
        mel = mel.transpose(0, 1)

        # pad / cut to target length
        n_frames = mel.shape[0]
        p = self.target_length - n_frames

        if p > 0:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, p))
        else:
            mel = mel[:self.target_length, :]

        return mel


    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[List[Image.Image], IO[bytes]]:
        datum = self.data[index]
        datum = self.decode_data(datum)

        audio_path = str(datum['wav'])
        audio = self.get_wav(audio_path)

        video_path = str(datum['video_path'])
        video_id = str(datum['video_id'])

        frames = []

        if not self.all_frames:
            image_path = video_path + f'/frame_{self.frame_num}/' + video_id + '.jpg'
            image = self.get_image(image_path)
            frames.append(image)
        else:
            for i, frame in enumerate(os.listdir(video_path)):    
                image_path = video_path + f'/{frame}/' + video_id + '.jpg'
                image = self.get_image(image_path)
                frames.append(image)

        fbank = audio
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        return frames, fbank
    
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
    