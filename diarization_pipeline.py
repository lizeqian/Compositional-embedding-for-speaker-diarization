import os
import torch
from pyannote.audio.models import SincTDNN
from pyannote.audio.train.task import Task, TaskOutput, TaskType
from pyannote.core import Annotation
from pyannote.core import Segment, Annotation
from pyannote.audio.utils.signal import Binarize
from speaker_diarization_overlap import SpeakerDiarizationOverlap
from g_net import GNet
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def baseline(pipeline, test_file):
    hypothesis = Annotation()
    diarization = pipeline(test_file)
    for seg_cnt, seg in enumerate(diarization._tracks._list):
        if seg.duration <= 0.1:
            continue
        label = diarization._tracks[seg]
        label = label[list(label.keys())[0]]
        hypothesis[seg] = label
    return hypothesis

def baseline_ovl(pipeline, ovl, test_file):
    hypothesis = Annotation()
    ovl_scores = ovl(test_file)
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, min_duration_off=0.1, min_duration_on=0.1)
    overlap = binarize.apply(ovl_scores, dimension=1)
    test_file['overlap'] = overlap
    diarization, diarization1 = pipeline(test_file)
    for seg_cnt, seg in enumerate(diarization._tracks._list):
        if seg.duration <= 0.1:
            continue
        label = diarization._tracks[seg]
        label1 = diarization1._tracks[seg]
        label = label[list(label.keys())[0]]
        label1 = label1[list(label1.keys())[0]]

        hypothesis[seg] = label
        for ref_seg in overlap:
            if ref_seg.intersects(seg):
                hypothesis[Segment(max(seg.start, ref_seg.start), min(seg.end, ref_seg.end))] = label
                hypothesis[Segment(max(seg.start, ref_seg.start), min(seg.end, ref_seg.end)+1e-7)] = label1
    return hypothesis

def comp(pipeline, test_file):
    hypothesis = Annotation()
    diarization = pipeline(test_file)
    for seg_cnt, seg in enumerate(diarization._tracks._list):
        if seg.duration <= 0.1:
            continue
        label = diarization._tracks[seg]
        label = label[list(label.keys())[0]]
        if '_' in str(label):
            a,b = label.split('_')
            hypothesis[seg] = a
            hypothesis[Segment(seg.start, seg.end+1e-7)] = b
        else:
            hypothesis[seg] = label
    return hypothesis

def comp_ovl(pipeline, ovl, test_file):
    hypothesis = Annotation()
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, min_duration_off=0.1, min_duration_on=0.1)
    ovl_scores = ovl(test_file)
    overlap = binarize.apply(ovl_scores, dimension=1)
    test_file['overlap'] = overlap
    diarization, diarization1 = pipeline(test_file)
    for seg_cnt, seg in enumerate(diarization._tracks._list):
        if seg.duration <= 0.1:
            continue
        label = diarization._tracks[seg]
        label1 = diarization1._tracks[seg]
        label = label[list(label.keys())[0]]
        label1 = label1[list(label1.keys())[0]]
        hypothesis[seg] = label
        for ref_seg in overlap:
            if ref_seg.intersects(seg):
                hypothesis[Segment(max(seg.start, ref_seg.start), min(seg.end, ref_seg.end))] = label1.split('_')[0]
                hypothesis[Segment(max(seg.start, ref_seg.start), min(seg.end, ref_seg.end)+1e-7)] = label1.split('_')[1]
    return hypothesis

if __name__ == "__main__":
    ami_path = sys.argv[1]

    # create result folders for all experiments
    os.makedirs('results/baseline', exist_ok=True)
    os.makedirs('results/baseline_ovl', exist_ok=True)
    os.makedirs('results/comp', exist_ok=True)
    os.makedirs('results/comp_ovl', exist_ok=True)

    # get the name of audios
    audios = []
    with open(f'{ami_path}/AMI/MixHeadset.test.rttm') as f: 
        lines = f.readlines()
        for line in lines:
            audio_name = line.strip().split()[1].split('.')[0]
            if audio_name not in audios:
                audios.append(audio_name)

    # baseline experiment
    task = Task(TaskType.REPRESENTATION_LEARNING,TaskOutput.VECTOR)
    specifications = {'X':{'dimension': 1} ,'task': task}
    sincnet = {'instance_normalize': True, 'stride': [5, 1, 1], 'waveform_normalize': True}
    tdnn = {'embedding_dim': 512}
    embedding = {'batch_normalize': False, 'unit_normalize': False}
    f_net = SincTDNN(specifications=specifications, sincnet=sincnet, tdnn=tdnn, embedding=embedding).to(device) 
    f_net.load_state_dict(torch.load("checkpoints/f_vxc.pt"))
    pipeline = SpeakerDiarizationOverlap('baseline', None, device, sad_scores='sad_ami', scd_scores='scd_ami', embedding='emb_ami', method = 'affinity_propagation')
    pipeline.load_params('config.yml')
    pipeline._pipelines['speech_turn_clustering']._embedding.scorer_.model_ = f_net
    pipeline._pipelines['speech_turn_assignment']._embedding.scorer_.model_ = f_net

    for cnt, audio in enumerate(audios):
        print(audio)
        test_file = {'uri': f'{audio}.Mix-Headset', 'audio': f'{ami_path}/amicorpus/{audio}/audio/{audio}.Mix-Headset.wav'}
        hypothesis = baseline(pipeline, test_file)
        hypothesis.uri = audio+'.Mix-Headset'
        with open(f'results/baseline/{audio}.rttm', 'w') as f:
            hypothesis.write_rttm(f)

    # baseline with overlap detector
    ovl = torch.load('checkpoints/ovl.pt')
    pipeline = SpeakerDiarizationOverlap('baseline_ovl', None, device, sad_scores='sad_ami', scd_scores='scd_ami', embedding='emb_ami', method = 'affinity_propagation')
    pipeline.load_params('config.yml')
    pipeline._pipelines['speech_turn_clustering']._embedding.scorer_.model_ = f_net
    pipeline._pipelines['speech_turn_assignment']._embedding.scorer_.model_ = f_net
    for cnt, audio in enumerate(audios):
        print(audio)
        test_file = {'uri': f'{audio}.Mix-Headset', 'audio': f'{ami_path}/amicorpus/{audio}/audio/{audio}.Mix-Headset.wav'}
        hypothesis = baseline_ovl(pipeline, ovl, test_file)
        hypothesis.uri = audio+'.Mix-Headset'
        with open(f'results/baseline_ovl/{audio}.rttm', 'w') as f:
            hypothesis.write_rttm(f)
    
    # compositional embedding
    f_net.load_state_dict(torch.load("checkpoints/best_f.pt"))
    g_net = GNet().to(device)
    g_net.load_state_dict(torch.load("checkpoints/best_g.pt"))
    pipeline = SpeakerDiarizationOverlap('comp', g_net, device, sad_scores='sad_ami', scd_scores='scd_ami', embedding='emb_ami', method = 'affinity_propagation')
    pipeline.load_params('config.yml')
    pipeline._pipelines['speech_turn_clustering']._embedding.scorer_.model_ = f_net
    pipeline._pipelines['speech_turn_assignment']._embedding.scorer_.model_ = f_net
    for cnt, audio in enumerate(audios):
        print(audio)
        test_file = {'uri': f'{audio}.Mix-Headset', 'audio': f'{ami_path}/amicorpus/{audio}/audio/{audio}.Mix-Headset.wav'}
        hypothesis = comp(pipeline, test_file)
        hypothesis.uri = audio+'.Mix-Headset'
        with open(f'results/comp/{audio}.rttm', 'w') as f:
            hypothesis.write_rttm(f)
    
    # compositional embedding with overlap detector
    pipeline = SpeakerDiarizationOverlap('comp_ovl', g_net, device, sad_scores='sad_ami', scd_scores='scd_ami', embedding='emb_ami', method = 'affinity_propagation')
    pipeline.load_params('config.yml')
    pipeline._pipelines['speech_turn_clustering']._embedding.scorer_.model_ = f_net
    pipeline._pipelines['speech_turn_assignment']._embedding.scorer_.model_ = f_net
    for cnt, audio in enumerate(audios):
        print(audio)
        test_file = {'uri': f'{audio}.Mix-Headset', 'audio': f'{ami_path}/amicorpus/{audio}/audio/{audio}.Mix-Headset.wav'}
        hypothesis = comp_ovl(pipeline, ovl, test_file)
        hypothesis.uri = audio+'.Mix-Headset'
        with open(f'results/comp_ovl/{audio}.rttm', 'w') as f:
            hypothesis.write_rttm(f)
    