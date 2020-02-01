# Search-by-Voice Project for the Intel Edge AI Scholarship Challenge 2020 SpeechVINO study group

## Goal:
To develop a search-by-voice application at the edge to detect time period(s) during audio/video play where specific persons are speaking with Intel's OpenVINO Toolkit

## Plan of Attack:
![Search-by-Voice Workflow](https://github.com/Speech-VINO/Search-by-Voice/blob/master/searchbyvoiceapp.png)

## Official PyTorch torchaudio tutorial
https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

## Datasets:
1. https://www.kaggle.com/wiradkp/mini-speech-diarization

## Kaldi for Dummies tutorial (Official):
https://kaldi-asr.org/doc/kaldi_for_dummies.html

## Code contribution by Wira (source code adapted here) 
1. https://github.com/WiraDKP/pytorch_speaker_diarization

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

Now, we have a spectrogram in freq vs time. When we check the STFT results, turns out that the Fourier Transform results are all near zero. That’s why we’ll measure it in the log scale, or more commonly, in decibels (dB). I prefer log though, but it is related to the math, so I’ll not explain here for now.

Here is how significant the log transformation is: 
![Log Transformation Image](https://github.com/Speech-VINO/Search-by-Voice/blob/master/log_transformation.png)

We would call these kind of audio representation as audio features. So now you have a feature, you can perform classification using CNN or RNN.

Well of course, knowledge evolve through time.

If you look at a spectrogram, you would almost always saw that in higher frequency, the color is black (almost zero). And that make sense, you would most likely hear an audio in frequency 0 - 8,000 Hz, rather than 50,000 Hz. So somehow, we will need to focus more on lower frequency rather that the higher one.

Then a group of people start wondering if there is a better way to represent spectrogram? Can we somehow “weigh” the lower frequency better than the higher one? Those people are Stevens, Volkmann, and Newmann. They used a natural frequency in our melody and created this formula: 

![Formula](https://github.com/Speech-VINO/Search-by-Voice/blob/master/formula.png)

Looks intimidating, huh? Don’t worry, it is simply just to “weigh” the frequency. Because it comes from melody, people would call it the Mel Frequency, or sometimes the Mel Scale. In fact, you could somehow create your own scale based on what you think the importance of the frequency and call it KS Scale, Aiad Scale and so on. So I’ll stop the scale right there because there are a lot of them. I’ll simply mention the most famous one - the Mel Scale.

So now, let’s weigh our previous feature. From the spectrogram, we can scale its frequency using Mel Scale, and now it is called as Mel Spectrogram.

For some practical note
```
stft = librosa.stft
mel scale = librosa.filters.mel
melspectrogram = librosa.feature.melspectrogram
```
So now it is simply the same, just with different feature representation; and, you can use CNN, RNN, or Transformer to classify the mel spectrogram. In fact, that is what I did in speaker diarization project: I featurized the audio into log melspectrogram, and then classified them using RNN (GRU). That is the main idea.

You can also use this for different use cases such as SER. Simply featurize the audio, then classify them into emotions. So my code in speaker diarization could actually be modified for SER too. 

## Literature & Resources:
1. [Identifying speakers with voice recognition - Python Deep Learning Cookbook](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781787125193/9/ch09lvl1sec61/identifying-speakers-with-voice-recognition)
2. [New Scientist article: Speech Recognition AI Identifies You by Voice Wherever You Are](https://www.newscientist.com/article/mg22830423-100-speech-recognition-ai-identifies-you-by-voice-wherever-you-are/)
3. [Medium article: Speaker Diarization with Kaldi](https://towardsdatascience.com/speaker-diarization-with-kaldi-e30301b05cc8)
4. [Arxiv article: Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719)
5. [Google AI Blog: Accurate Online Speaker Diarization with Supervised Learning](https://ai.googleblog.com/2018/11/accurate-online-speaker-diarization.html)
6. [Medium article: AEI: Artificial 'Emotional' Intelligence](https://towardsdatascience.com/aei-artificial-emotional-intelligence-ea3667d8ece)
7. [Medium article: Speaker Diarization with Kaldi](https://towardsdatascience.com/speaker-diarization-with-kaldi-e30301b05cc8)
8. [How to start Kaldi and speech recognition](https://towardsdatascience.com/how-to-start-with-kaldi-and-speech-recognition-a9b7670ffff6)
9. https://www.kaggle.com/caesarlupum/speech-recognition-timealignedspectrogramsn
10. [Medium article: How to Start with Kalki and Speech Recognition](https://towardsdatascience.com/how-to-start-with-kaldi-and-speech-recognition-a9b7670ffff6)
