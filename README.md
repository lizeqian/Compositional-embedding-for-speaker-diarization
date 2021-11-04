Repository for paper *Compositional embedding models for speaker identification and diarization with simultaneous speech from 2+ speakers*  
https://arxiv.org/abs/2010.11803

## Requirement
Install pytorch using instructions from https://pytorch.org.
Install pyannote-audio using instructions from https://github.com/pyannote/pyannote-audio.

# Evaluation

## Prepare test data 
Download AMI Headset-mix dataset using the script from https://github.com/pyannote/pyannote-audio/tree/master/tutorials/data_preparation.

## Run test script for all experiments in the paper
Run the command  
`python diarization_pipeline.py [YOUR_AMI_DATA_PATH]`  
will generate rttm formated diarization results for all experiments.

## Diarization for a single wav file (for ISAT)
Use command  
`python isat_diarization.py [WAV_PATH] [OUTPUT_DIR]`  
to generate both rttm formated VAD and diarization results.

# Training

## Prepare trainin data

## Train the model
