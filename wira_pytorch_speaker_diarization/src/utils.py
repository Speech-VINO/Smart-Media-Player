import numpy as np
from librosa.feature import zero_crossing_rate


def quantile_normalize(audio, quantile=0.999):
    return audio / np.quantile(abs(audio), quantile)

def zcr(audio, shift=0.05, frame_len=2048, hop_len=1024, zcr_threshold=0.005):
    zc_rate = zero_crossing_rate(audio+shift, frame_length=frame_len, hop_length=hop_len, center=False)[0]
    mask = np.where(zc_rate > zcr_threshold, 1, 0)
    return mask