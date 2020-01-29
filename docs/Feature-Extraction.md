### **Feature Extraction**

---------------------------------



**Actor Identity**

| **Sample File**      | **Actor** | **Emotion** | **Modality** | **Statement** | **Vocal Channel** |
| -------------------- | --------- | ----------- | ------------ | ------------- | ----------------- |
| 01-01-02-01-01-01-01 | 01        |             |              |               |                   |
| 01-01-02-01-01-01-02 | 02        |             |              |               |                   |
| 01-01-02-01-01-01-03 | 03        |             |              |               |                   |
| 01-01-02-01-01-01-04 | 04        |             |              |               |                   |
| 01-01-02-01-01-01-05 | 05        |             |              |               |                   |



**Mel filterbanks**

Mel filterbanks are computed using `librosa` library. For installation of `librosa`, please execute this command:

`pip install librosa`

Librosa converts the audio wav signal into a time series by sampling the `.wav`file using a signal rate. The default signal rate of the time series will be `22050`. 

`from librosa.feature import melspectrogram`



**Quantile Analysis**

Mel filterbanks generate a frequency model by applying Short Time Fourier Analysis. STFT is a windowed discrete Fourier transform that generates a spectogram from the input time series function. Using quantile analysis, we assume the time series is coming from a distribution and the time series function is normalised using a quantile value of `0.999`. 

Librosa transforms the Mel filterbanks into frequency bins by computing the up slopes and down slopes using a `Slaney` transform



$$
\begin{align*}
Slaney_t &= \frac{2.0}{(mel_f[2:mels+2] - mel_f[:mels])}
\end{align*}
$$


Slaney transform makes sure the output is mapped with the feature size of each `wav` file. 