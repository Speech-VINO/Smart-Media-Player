# Smart Media Player - Segment Video/Audio by Speaker 
*Smart Media Player Project for the Intel Edge AI Scholarship Challenge 2020 #SpeechVINO study group*

## Goal
To develop a ***Smart Media Player Application*** at the Edge using Intel's OpenVINO Toolkit where, given an audio or video as input, the application will segment it in terms of timestamps with respect to the individual speakers.

## Idea and Motivation
As meantioned above, the goal of ***Smart Media Player Application*** is to develop a search-by-voice application at the edge to detect time period(s) during audio/video play where the specific person is speaking, using Intel's OpenVINO Toolkit.

**Why did we choose to work on this project?**

Often, there are times when people are looking for a certain speaker and would want to listen to that part of the audio or video clip! Maybe its a webinar with many speakers, or just an audio clip where you want to listen to only what Mr. X spoke of. At times like these, it's cumbersome to search through the entire audio or video clip just for a few seconds long clip, and would rather look for an application that could help find Mr. X's video or audio segments with just one click.

**So, How does the Smart Media Player Application help such users?**

Once the audio or video clip is submitted, the search-by-voice segregates or makes a list of speakers and their timestamps in the given clip and presents it to the user. Now, the user may just look for Mr. X's part from within the audio or video clip with just one click!

**What could this be used for?**

Normal day to day users searching webinar clips, Forensics - searching video for certain speakers, etc.

**What happens under the hood - Our Approach?**

Under the hood, once the user submits an audio or video clip, the audio part of it is taken and pre-processed to the desired input format, followed by Speaker diarisation (or diarization), which is the process of partitioning an input audio stream into homogeneous segments according to the speaker identity, resulting with the list of Identified speakers and their audio clip timestamps.

For which, the team chose to try out two options - (1) Create a model from scratch and train it, (2) Use Kaldi, an open-source speech recognition toolkit, which is currently in progress!


## Smart Media Player Demo (Desired Output)

## Player for one video
User can have list of speakers inside the media and can know/play specific one
![Project Workflow](https://github.com/Speech-VINO/Smart-Media-Player/blob/master/Demo/smartplayer-video.gif)

## Player for Playlist
User can have list of speakers inside the media as well as know how many minutes inside each video in playlist specific speaker is speaking and how many speakers are inside the each video in playlist
![Project Workflow](https://github.com/Speech-VINO/Smart-Media-Player/blob/master/Demo/smartplayer-playlist.gif)


## Concept Workflow
![Project Workflow](https://github.com/Speech-VINO/Smart-Media-Player/blob/master/Approach/ConceptWorkflow.png)

## Concept Implementation (AI Magic)
@wira please add speaker embedding and diaraization concept behinde this project

## Datasets
1. https://www.kaggle.com/wiradkp/mini-speech-diarization

# Approach 1: From Scratch

### Code contribution by Wira (source code adapted here) 
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

### Current status and understanding of the results: 
We have a trained speaker embedding that could represent an audio segment into vector. The speaker embedding is used in the speaker diarization to find timestamp for each speaker in the audio/video. We have come up with a mockup of the application that demonstrates how the smart media player should work. **Desired outputs:** Timestamp of each speaker in an audio/video including unknown person.

![Scattered graph depictions the speaker diaraization](https://github.com/Speech-VINO/Search-by-Voice/blob/master/image1.jpg)

# Approach 2: Using Kaldi

## Why Choose Kaldi?
Kaldi is an open-source speech recognition toolkit written in C++ for speech recognition and signal processing, freely available under the Apache License v2.0.It aims to provide software that is flexible and extensible, and is intended for use by automatic speech recognition (ASR) researchers for building a recognition system.

It supports linear transforms, MMI, boosted MMI and MCE discriminative training, feature-space discriminative training, and deep neural networks and is capable of generating features like mfcc, fbank, fMLLR, etc. Hence in recent deep neural network research, a popular usage of Kaldi is to pre-process raw waveform into acoustic feature for end-to-end neural models. Acoustic models are the statistical representations of a phoneme’s acoustic information. A phoneme here represents a member of the set of speech sounds in a language. 

The acoustic models are created by training the models on acoustic features from labeled data, such as the Wall Street Journal Corpus, TIMIT, or any other transcribed speech corpus. There are many ways these can be trained, and the tutorial will try to cover some of the more standard methods. Acoustic models are necessary not only for automatic speech recognition, but also for forced alignment.

*Kaldi provides tremendous flexibility and power in training your own acoustic models and forced alignment system.*

### Current Status
Converting Kaldi for deployment in OpenVINO, followed by Model Optimization. 

# Team Members (Slack handles)

@Aiad Taha 

@K.S.

@MCB 

@sago 

@Shaistha

@Wira

@Zarreen Reza 


## Desired Outputs
![Video ouput](https://github.com/Speech-VINO/Search-by-Voice/blob/master/output1.png)

![Marked Output](https://github.com/Speech-VINO/Search-by-Voice/blob/master/output2.png)

## Future Updates


## Social Impact 

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
11. [Official PyTorch torchaudio tutorial](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html)
12. [Kaldi for Dummies tutorial (Official)](https://kaldi-asr.org/doc/kaldi_for_dummies.html)
