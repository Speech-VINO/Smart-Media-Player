import os
import numpy as np
import matplotlib.pyplot as plt

import librosa
from librosa.feature import melspectrogram
from src.utils import quantile_normalize

import torch
from torch.onnx import export
from torch.nn.functional import log_softmax

from src.model import SpeakerDiarization

from itertools import groupby
from collections import defaultdict

from openvino.inference_engine import IECore, IENetwork


class BasePredictor:
    def __init__(self, config_path, chunk_dur):
        config = torch.load(config_path)
        
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sr = config.audio_params["sr"]
        self.win_length = config.audio_params["win_length"]
        self.hop_length = config.audio_params["hop_length"]
        self.n_mel = config.audio_params["n_mel"]
        self.n_fft = config.audio_params["n_fft"]
        self.f_min = config.audio_params["f_min"]
        self.f_max = config.audio_params["f_max"]
        self.classes = config.classes
        self.chunk_len = chunk_dur * self.sr
        
        self.n_layer = config.n_layer
        self.n_hidden = config.n_hidden        
        self.n_input = config.n_input
        self.n_output = config.n_output
        self.dropout = config.dropout
        
    def _to_melspec(self, audio):
        spec = melspectrogram(audio, sr=self.sr, center=False, n_mels=self.n_mel, n_fft=self.n_fft, 
                              win_length=self.win_length, hop_length=self.hop_length, fmin=self.f_min, 
                              fmax=self.f_max)
        spec = np.log(spec)
        spec = torch.FloatTensor([spec])
        return spec
        
    def _pad(self, audio):
        mod = (len(audio) % self.chunk_len)
        if mod != 0:
            pad = np.ones(self.chunk_len - mod)*1e-5
            audio = np.concatenate([audio, pad])
        return audio.reshape(-1, self.chunk_len)
    
    def _plot(self, X, result):
        lim = max(abs(X))
        plt.figure(figsize=(15, 2))
        plt.plot(X, 'k-')
        for i, c in enumerate(self.classes):
            if c != "<NONE>":
                plt.fill_between(range(len(X)), -lim, lim, where=(result==i), alpha=0.5, label=c)
        plt.legend(loc="upper center", ncol=i, bbox_to_anchor=(0.5, -0.25))        
        
    def _get_timestamp(self, X, result):
        speakers = [k for k, _ in groupby(result)]
        change = np.argwhere(result[:-1] != result[1:]).flatten()
        span = np.concatenate([[0], change, [len(X)]]) / self.sr
        
        timestamp = defaultdict(lambda: list())
        for speaker, start, end in zip(speakers, span[:-1], span[1:]):
            if speaker != 0:
                speaker_name = self.classes[speaker]
                timestamp[speaker_name].append((start, end))
        return timestamp

    
class PyTorchPredictor(BasePredictor):
    def __init__(self, weights_path, config_path, chunk_dur):
        super().__init__(config_path, chunk_dur)
        weights = torch.load(weights_path, map_location="cpu")
        
        self.model = SpeakerDiarization(self.n_input, self.n_output, self.n_hidden, self.n_layer, self.dropout)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model = self.model.to(self._device)
        
    def predict(self, wav_path, plot=False):
        X, sr = librosa.load(wav_path, sr=self.sr)
        X = quantile_normalize(X)
        X = self._pad(X)
        
        hidden = None
        result = []
        for segment in X:
            spec = self._to_melspec(segment).to(self._device)
            with torch.no_grad():
                out, hidden = self.model(spec, hidden)
                pred = log_softmax(out, dim=1).argmax(1).squeeze(0)
            n_repeat = self.chunk_len // len(pred) + 1
            pred = pred.repeat_interleave(n_repeat)
            result.append(pred[:self.chunk_len])
        result = np.hstack(result)
        X = X.flatten()

        if plot:
            self._plot(X, result)
        timestamp = self._get_timestamp(X, result)
        return timestamp
    
    def to_onnx(self, fname="speaker_diarization.onnx", outdir="model"):
        os.makedirs(outdir, exist_ok=True)
        audio = np.ones(self.chunk_len)
        audio = self._to_melspec(audio)
        hidden = torch.rand(self.n_layer, 1, self.n_hidden)
        export(self.model, (audio, hidden), f"{outdir}/{fname}", input_names=["input"], output_names=["output"])
        
        
class OpenVINOPredictor(BasePredictor):
    def __init__(self, model_xml, model_bin, config_path, chunk_dur, CPU_EXT):
        super().__init__(config_path, chunk_dur)
        net = IENetwork(model_xml, model_bin)
        self._h0_size = net.inputs["hidden"].shape
        
        plugin = IECore()
        plugin.add_extension(CPU_EXT, "CPU")
        self.exec_net = plugin.load_network(net, "CPU")
        
    def predict(self, wav_path, plot=False):
        X, sr = librosa.load(wav_path, sr=self.sr)
        X = quantile_normalize(X)
        X = self._pad(X)
        
        hidden = np.zeros(self._h0_size)
        result = []
        for segment in X:
            spec = self._to_melspec(segment)
            output = self.exec_net.infer({"input": spec, "hidden": hidden})
            out, hidden = output["output"], output["147"]
            pred = self._log_softmax(out, dim=1).argmax(1).squeeze(0)
            n_repeat = self.chunk_len // len(pred) + 1
            pred = pred.repeat(n_repeat)
            result.append(pred[:self.chunk_len])
        result = np.hstack(result)
        X = X.flatten()

        if plot:
            self._plot(X, result)
        timestamp = self._get_timestamp(X, result)
        return timestamp
    
    @staticmethod
    def _log_softmax(x, dim=1):        
        e_x = np.exp(x - np.max(x, axis=dim))
        return np.log(e_x / e_x.sum(axis=dim))
