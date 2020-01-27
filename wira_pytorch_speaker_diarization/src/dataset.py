from glob import glob
from os.path import split, dirname

import numpy as np
from tqdm.auto import tqdm

import wave
from contextlib import closing
import librosa
from librosa.feature import melspectrogram

from src.exception import SliceError
from src.utils import quantile_normalize, zcr

import torch
from torch.utils.data import Dataset


class AudioFolder(Dataset):
    def __init__(self, folder_path, sr=16000, n_data=500, slice_dur=5, n_mel=128, n_fft=2048, 
                 win_length=2048, hop_length=1024, freq_range=(0, 8000)):
        self.sr = sr
        self.slice_dur = slice_dur
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.f_min, self.f_max = freq_range
        self.audio_params = {
            "sr": sr,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mel": n_mel,
            "n_fft": n_fft,
            "f_min": freq_range[0],
            "f_max": freq_range[1]
        }
        self.n_data = n_data
        
        files = sorted(glob(f"{folder_path}/*/*.wav"))
        min_duration = min(self._wav_duration(f) for f in files)
        
        if slice_dur > min_duration:
            raise SliceError(f"The shortest audio is {min_duration:.2f} s. slice_dur must be less than that.")
        
        self.files = files
        self.labels = [self._parent(f) for f in files]

        self.classes = ["<NONE>"] + list(set(self.labels))
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        
        X, y = zip(*[self._create_data() for _ in tqdm(range(n_data), desc="Create Data")])
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.n_data
    
    def _create_data(self):
        f = np.random.choice(self.files)
        label = self._parent(f)
        
        X, sr = librosa.load(f, sr=self.sr)
        X = self._random_slice(X)
        X = quantile_normalize(X)
        mask = zcr(X, frame_len=self.win_length, hop_len=self.hop_length)
        y = mask * self.class_to_idx[label]
        
        X = melspectrogram(X, sr=self.sr, center=False, n_mels=self.n_mel, n_fft=self.n_fft, win_length=self.win_length, 
                           hop_length=self.hop_length, fmin=self.f_min, fmax=self.f_max)
        X = np.log(X)
        return X, y
        
    def _random_slice(self, audio):
        slice_len = int(self.slice_dur * self.sr)
        diff = len(audio) - slice_len
        start = np.random.randint(diff)
        return audio[start: start+slice_len]
            
    @staticmethod
    def _parent(path):
        return split(dirname(path))[-1]   
    
    @staticmethod
    def _wav_duration(fname):
        with closing(wave.open(fname, 'r')) as f:
            duration = f.getnframes() / f.getframerate()
        return duration
