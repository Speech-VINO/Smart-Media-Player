# Search-by-Voice Project for the Intel Edge AI Scholarship Challenge 2020 SpeechVINO study group

## Goal:
To develop a search-by-voice application at the edge to detect time period(s) during audio/video play where specific persons are speaking with Intel's OpenVINO Toolkit

## Plan of Attack:
![Search-by-Voice Workflow](https://github.com/Speech-VINO/Search-by-Voice/blob/master/searchbyvoiceapp.png)

## Datasets:
1. https://www.kaggle.com/wiradkp/mini-speech-diarization

## Kaldi for Dummies tutorial (Official):
https://kaldi-asr.org/doc/kaldi_for_dummies.html

## Notes from Wira
There are 2 approach to handle a raw audio data:
1. Using the raw signal
2. Converts the signal to “image”

It is common to choose the latter because it saves a lot of computation. Think about it, raw signal with 44.1 kHz would have 44100 data points each second. That is quiet overwhelming to compute. So there are different way to represent those raw audio signals. Instead of its amplitude, why not focus on its frequency? And there the representation of frequency vs time of an audio signal is what we called as spectrogram. We actually have a powerful tool to convert amplitude to frequency, it is called Fourier Transform. 

So when we have an audio signal, here is what we do:
* Set a certain window / segment length to compute Fourier Transform
* Convert those segment and we get its freq information
* Move the window to the right, and start compute the freq again until it reaches the end of the audio

This is called *STFT* (Short-term Fourier Transform) and Voila… STFT results in frequencies for each window plot it as frequency vs time, we’ll get the spectrogram. This is easy to do using python library named librosa.
Spectrogram may be the basic feature you need to know for now.

## Literature & Resources:
1. [Identifying speakers with voice recognition - Python Deep Learning Cookbook](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781787125193/9/ch09lvl1sec61/identifying-speakers-with-voice-recognition)
2. [New Scientist article: Speech Recognition AI Identifies You by Voice Wherever You Are](https://www.newscientist.com/article/mg22830423-100-speech-recognition-ai-identifies-you-by-voice-wherever-you-are/)
3. [Medium article: Speaker Diarization with Kaldi](https://towardsdatascience.com/speaker-diarization-with-kaldi-e30301b05cc8)
4. [Arxiv article: Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719)
5. [Google AI Blog: Accurate Online Speaker Diarization with Supervised Learning](https://ai.googleblog.com/2018/11/accurate-online-speaker-diarization.html)
6. [Medium article: AEI: Artificial 'Emotional' Intelligence](https://towardsdatascience.com/aei-artificial-emotional-intelligence-ea3667d8ece)
7. [Medium article: Speaker Diarization with Kaldi](https://towardsdatascience.com/speaker-diarization-with-kaldi-e30301b05cc8)
8. [How to start Kaldi and speech recognition](https://towardsdatascience.com/how-to-start-with-kaldi-and-speech-recognition-a9b7670ffff6)
