import os, sys
import torch
from pyannote.audio.models import SincTDNN
from pyannote.audio.train.task import Task, TaskOutput, TaskType
from pyannote.core import Annotation
from pyannote.core import Segment, Annotation
from pyannote.audio.utils.signal import Binarize
from speaker_diarization_overlap import SpeakerDiarizationOverlap
from diarization_pipeline import baseline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    audio, target = sys.argv[1], sys.argv[2]
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

    test_file = {'uri': audio, 'audio': audio}
    vad = pipeline._get_vad_segments(test_file)
    with open(os.path.join(target, 'vad.rttm'), 'w') as f:
        vad.write_rttm(f)
    hypothesis = baseline(pipeline, test_file)
    hypothesis.uri = audio
    with open(os.path.join(target, 'diarization.rttm'), 'w') as f:
        hypothesis.write_rttm(f)