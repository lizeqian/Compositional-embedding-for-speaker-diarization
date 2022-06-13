Repository for paper *Compositional embedding models for speaker identification and diarization with simultaneous speech from 2+ speakers*  
https://arxiv.org/abs/2010.11803

## Requirement
Install pytorch using instructions from https://pytorch.org.  
Install pyannote-audio 1.1 using instructions from https://github.com/pyannote/pyannote-audio.
```
pip install pyannote.audio==1.1.1
```

# Evaluation

## Prepare test data 
Download AMI Headset-mix dataset using the script from https://github.com/pyannote/pyannote-audio/tree/master/tutorials/data_preparation.

## Run test script for all experiments in the paper
Run the command  
```
python diarization_pipeline.py [YOUR_AMI_DATA_PATH]
```
will generate rttm formated diarization results for all experiments.

## Diarization for a single wav file (for ISAT)
Use command  
```
python isat_diarization.py [WAV_PATH] [OUTPUT_DIR]
```  
to generate both rttm formated VAD and diarization results for a 16k Hz wav file.

In config.yml, smaller speech_turn_assignment.threshold results in more singletons not assigned to any speaker, larger speech_turn_clustering.preference (closer to 0) results in more clusters (speakers).

# Training

## Prepare trainin data
Prepare 3 files that includes paths to VoxCeleb (or any other dataset) and noise file.

`voxceleb_train.json` and `voxceleb_test.json` are the train split and validation split of VoxCeleb. 

Each is a dictionary where the keys are the speaker ids and the values are lists of paths of wav files belong to the corresponding speaker id.

`musan_noise_files_list.txt` contains the paths of noise files used for data augmentation. Each line is the path to a noise wav file.

## Train the model
```
python train_fg.py
```
A pretrained speaker embedding model that is trained on VoxCeleb will be loaded before the training of compositional embedding.

There is an updated training strategy in `train_fg_arcface_triplet.py` which uses arcface loss and triplet loss in an interleaving way.
