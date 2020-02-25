"""
author: Wira D K Putra
25 February 2020

See original repo at
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
"""

import os
import wave
import torch
import numpy as np

from glob import glob
from tqdm.auto import tqdm

import torchaudio
from torchaudio.transforms import MFCC, Resample
from torch.utils.data import Dataset, DataLoader


class BaseLoad:
    def __init__(self, sr, n_mfcc=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self._mfcc = MFCC(sr, n_mfcc=40, log_mels=True)
        
    def _load(self, path, mfcc=True):
        try:
            waveform, ori_sr = torchaudio.load(path)
            waveform = waveform.mean(0, keepdims=True)
        except RuntimeError:
            raise Exception(f"Error loading {path}")
        _resample = Resample(ori_sr, self.sr)
        audio = _resample(waveform)

        if mfcc:
            audio = self._mfcc(audio)
        return audio


class VCTKTripletDataset(Dataset, BaseLoad):
    def __init__(self, wav_path, txt_path, n_data, sr=16000, min_dur=2):
        self.wav_path = wav_path
        self.txt_path = txt_path
        BaseLoad.__init__(self, sr)
        
        self.min_dur = min_dur
        self.speakers = list(sorted(os.listdir(wav_path)))
        self.speaker_to_idx = {v: k for k, v in enumerate(self.speakers)}
        
        self.data = [self._random_sample() for _ in tqdm(range(n_data), desc="Sample Data")]
        self._remove_short_audio()
        
    def __getitem__(self, i):
        a, p, n, ya, yp, yn = self.data[i]
        mfcc_a = self._load(a)
        mfcc_p = self._load(p)
        mfcc_n = self._load(n)
        ya = self.speaker_to_idx[ya]
        yp = self.speaker_to_idx[yp]
        yn = self.speaker_to_idx[yn]
        return mfcc_a, mfcc_p, mfcc_n, ya, yp, yn
        
    def __len__(self):
        return len(self.data)
    
    def _random_sample(self):
        speaker_a, speaker_n = np.random.choice(self.speakers, 2, replace=False)
        a, p = np.random.choice(glob(f"{self.wav_path}/{speaker_a}/*.wav"), 2, replace=False)
        n = np.random.choice(glob(f"{self.wav_path}/{speaker_n}/*.wav"))
        return a, p, n, speaker_a, speaker_a, speaker_n
    
    def _remove_short_audio(self):
        def _dur(fname):
            with wave.open(fname, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            return duration
        
        new_data = [data for data in self.data if min(_dur(data[0]), _dur(data[0]), _dur(data[0])) >= self.min_dur]
        n_excluded = len(self.data) - len(new_data)
        
        if n_excluded > 0:
            print(f"Excluding {n_excluded} triplet containing audio shorter than {self.min_dur}s")
        self.data = new_data
    

class VCTKTripletDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=3):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, num_workers=num_workers)
        
    def collate(self, batch):
        a, p, n, ya, yp, yn = zip(*batch)
        X = a + p + n
        y = ya + yp + yn
        
        min_frame = min([i.shape[-1] for i in X])
        X = [i[:, :, :min_frame] for i in X]
        return torch.cat(X).unsqueeze(1), torch.LongTensor(y)
    
    
class VCTKSpeakerDataset(Dataset, BaseLoad):
    def __init__(self, wav_path, txt_path, n_speaker=20, n_each_speaker=10, sr=16000, min_dur=2):
        self.wav_path = wav_path
        self.txt_path = txt_path
        BaseLoad.__init__(self, sr)
        
        self.min_dur = min_dur
        self.speakers = list(sorted(os.listdir(wav_path)))
        self.speaker_to_idx = {v: k for k, v in enumerate(self.speakers)}
        
        random_speakers = np.random.choice(self.speakers, n_speaker, replace=False)
        self.data = [(path, speaker) for speaker in tqdm(random_speakers, desc="Sample Data")
                     for path in np.random.choice(glob(f"{self.wav_path}/{speaker}/*.wav"), n_each_speaker, replace=False)]
        self._remove_short_audio()
        
    def __getitem__(self, i):
        X, y = self.data[i]
        mfcc = self._load(X)
        y = self.speaker_to_idx[y]
        return mfcc, y
        
    def __len__(self):
        return len(self.data)
    
    def _remove_short_audio(self):
        def _dur(fname):
            with wave.open(fname, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            return duration
        
        new_data = [data for data in self.data if _dur(data[0]) >= self.min_dur]
        n_excluded = len(self.data) - len(new_data)
        
        if n_excluded > 0:
            print(f"Excluding {n_excluded} triplet containing audio shorter than {self.min_dur}s")
        self.data = new_data    

    
class VCTKSpeakerDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=3):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, num_workers=num_workers)
        
    def collate(self, batch):
        X, y = zip(*batch)
        
        min_frame = min([i.shape[-1] for i in X])
        X = [i[:, :, :min_frame] for i in X]
        return torch.cat(X).unsqueeze(1), torch.LongTensor(y)
