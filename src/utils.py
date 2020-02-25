"""
author: Wira D K Putra
25 February 2020

See original repo at
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
"""

import torch
import librosa
import numpy as np
from itertools import groupby
from scipy.ndimage import gaussian_filter1d


def zcr_vad(y, shift=0.025, win_len=2048, hop_len=1024, threshold=0.005):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if y.ndim == 2:
        y = y[0]
    zcr = librosa.feature.zero_crossing_rate(y + shift, win_len, hop_len)[0]
    activity = gaussian_filter1d(zcr, 1) > threshold
    activity = np.repeat(activity, len(y) // len(activity) + 1)
    activity = activity[:len(y)]
    return activity


def get_timestamp(activity):
    mask = [k for k, _ in groupby(activity)]
    change = np.argwhere(activity[:-1] != activity[1:]).flatten()
    span = np.concatenate([[0], change, [len(activity)]])
    span = list(zip(span[:-1], span[1:]))
    span = np.array(span)[mask]
    return span
