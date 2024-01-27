# Project Title: ASR Model for Torah Shiurim

## Goal
The goal of this project is to develop an Automatic Speech Recognition (ASR) model specifically designed for transcribing Torah shiurim. Torah shiurim are educational lectures or classes on Jewish religious texts, teachings, and interpretations. By creating an ASR model tailored to this domain, we aim to improve the accuracy and efficiency of transcribing these valuable teachings.

#### The project has multiple scopes

1) General training of a SOTA model for hebrew language ASR
  - Model training can be done in several ways:
    a) Finetune Whisper model
    b) Finetune Wav2Vec + n-gram

  - Datasets to be used can be found on huggingface from imvladikon, as well as Fleurs.

2) ASR for Torah Shiurim.
    a) Wav2Vec + n-gram
    b) Wav2Vec + AED (don't know how to do)
    c) Whisper (Need Dicta for this for synchronized data)

The SOTA Hebrew model is a goal in-and-of itself. Possibility for this to be used for the Torah shiurim later.
if the 

First task to be done is to prepare the KenLM model, and use the finetuned wav2vec model as the base
and the Torah based KenLm as the decoder.

If this works, then we can try and sync the transcriptions for whisper after.

Immediate steps:
1) Finetune wave2vec to get best hebrew model possible (compare to whisper regular for baseline)
2) Create Torah Kenml based model with the finetuned wav2vec as base