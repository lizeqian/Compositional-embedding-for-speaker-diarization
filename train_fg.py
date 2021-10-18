from pyannote.audio.models.models import Embedding
from pyannote.audio.models import SincTDNN
from pyannote.audio.models.scaling import Scaling
from pyannote.audio.train.task import Task, TaskOutput, TaskType
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os, random
from torch import optim
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm
import pickle
import math
import itertools
import torch.nn as nn
import torch.nn.functional as F
from initialization import weight_init
from g_net import GNet
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 512

class LibriReadeload(Dataset):
    def __init__(self, mode, step_size, speaker_dict, noise_list):
        with open(speaker_dict, 'rb') as f:
            self.speaker_dict = pickle.load(f)
        self.spkers = list(self.speaker_dict.keys())
        if mode == 'train':
            self.spkers = self.spkers[:-100]
        else:
            self.spkers = self.spkers[-100:]

        with open(noise_list) as f: 
            lines = f.readlines()

        self.noise_list = []
        for line in lines:
            self.noise_list.append(line.strip())

        self.combinations = torch.tensor(list(itertools.combinations(list(range(5)),2)))
        self.step_size = step_size

    def addNoise(self, data):
        noise_path = random.choice(self.noise_list)
        samplerate, noise = wavfile.read(noise_path)
        noise = noise.astype(np.int32)
        if noise.shape[0] >= samplerate*2:
            st = random.randint(0, noise.shape[0] - samplerate*2)
            noise = noise[st:st + samplerate*2]
        else:
            noise = noise.repeat(math.ceil((samplerate*2)/noise.shape[0]))[:samplerate*2]
        coeff = random.uniform(0, 0.4)
        noise_energy = np.sqrt(np.sum(noise**2))
        data_energy = np.sqrt(np.sum(data**2))
        res = data + coeff * noise * data_energy / noise_energy
        return res
    
    def getAudio(self, spker):
        audio_path = random.choice(self.speaker_dict[spker])
        samplerate, data = wavfile.read(audio_path)
        data = data.astype(np.int32)
        res = np.zeros(samplerate * 2)
        if data.shape[0] < samplerate * 2:
            res[:data.shape[0]] = data
        else:
            st = random.randint(0, data.shape[0] - samplerate * 2)
            res = data[st:st + samplerate * 2]
        return res

    def getMixture(self, A, B):
        sampleA = self.getAudio(A)
        sampleB = self.getAudio(B)
        res = sampleA + sampleB
        return res

    def __getitem__(self, index):
        sel_speakers = random.sample(self.spkers, 5)
        test_samples = []
        ref_samples = []
        for spker in sel_speakers:
            sample = self.getAudio(spker)
            sample = self.addNoise(sample)
            ref_samples.append(sample)
        
        for spker in sel_speakers:
            sample = self.getAudio(spker)
            sample = self.addNoise(sample)
            test_samples.append(sample)
        
        for comb in self.combinations:
            sample = self.getMixture(sel_speakers[comb[0]], sel_speakers[comb[1]])
            sample = self.addNoise(sample)
            test_samples.append(sample)
        
        data = np.stack(test_samples + ref_samples)

        return torch.tensor(data)

    def __len__(self):
        return self.step_size

def getDataloader(mode, batch_size, step_size, speaker_dict, noise_list):
    dataset = LibriReadeload(mode, step_size, speaker_dict, noise_list)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=12)
    return dataloader

def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    return torch.norm(D, p=2, dim=2)

class CompostionalEmbedding(pl.LightningModule):
    def __init__(self, f_net, g_net):
        super(CompostionalEmbedding, self).__init__()
        self.f_net = f_net
        self.g_net = g_net
        self.criterion = nn.MarginRankingLoss(0.1)
        self.combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size * 20, 32000, 1)
        x = f_net(x).contiguous().view(batch_size, 20, -1)
        return x

    def training_step(self, batch, batch_idx):
        data = batch.float()
        embeddings = self.forward(data)
        loss = 0
        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[self.combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[self.combinations2.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)

            truth = F.normalize(torch.cat([centroids, merged2]))
            preds = F.normalize(embedding[:-5])
            dists = pairwiseDists(preds, truth)
            loss = 0
            for i in range(15):
                if i < 5:
                    weight = 1
                else:
                    weight = 0.5
                dist = dists[i]
                loss += weight * self.criterion(dist[[e for e in range(15) if e != i]], dist[[i]*14], torch.ones(14, device=device))
            self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch.float()
        embeddings = self.forward(data)
        all_cnt_1, all_cnt_2 = 0, 0
        hit_cnt_1, hit_cnt_2 = 0, 0
        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[self.combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[self.combinations2.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)

            truth = F.normalize(torch.cat([centroids, merged2]))
            preds = F.normalize(embedding[:-5])
            dists = pairwiseDists(preds, truth)
            _, res = torch.topk(dists, 1, largest=False)
            label = torch.tensor(list(range(15)), device=device)
            all_cnt_1 += 5
            all_cnt_2 += 10
            hit_cnt_1 += torch.sum(label[:5] == res.squeeze()[:5]).item()
            hit_cnt_2 += torch.sum(label[5:] == res.squeeze()[5:]).item()
        self.log('step_val_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('step_val_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('step_val_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))
        return all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2
    
    def validation_epoch_end(self, outputs):
        all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2 = 0, 0, 0, 0
        for output in outputs:
            all_cnt_1 += output[0]
            all_cnt_2 += output[1]
            hit_cnt_1 += output[2]
            hit_cnt_2 += output[3]
        self.log('val_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('val_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('val_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([{'params': self.f_net.parameters(), 'lr': 1e-7}, {'params': self.g_net.parameters(), 'lr': 1e-5}], lr=3e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

class ResetLearningRate(Callback):
    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        if pl_module.current_epoch // 30 == 0:
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                for p_idx, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = 1e-5
                

if __name__ == "__main__":
    epochs = 10000

    task = Task(TaskType.REPRESENTATION_LEARNING,TaskOutput.VECTOR)
    specifications = {'X':{'dimension': 1} ,'task': task, 'y': {'classes': ['a', 'b']}}
    sincnet = {'instance_normalize': True, 'stride': [5, 1, 1], 'waveform_normalize': True}
    tdnn = {'embedding_dim': 512}
    embedding = {'batch_normalize': False, 'unit_normalize': False}#, 'scale': "unit"}
    f_net = SincTDNN(specifications, sincnet=sincnet, tdnn=tdnn, embedding=embedding).to(device)
    
    g_net = GNet().to(device)
    g_net.apply(weight_init)
    f_net.apply(weight_init)
    f_net.load_state_dict(torch.load("/home/lizeqian/pyannote-audio/checkpoint_pretrained/f.pt"))
    # f_net.load_state_dict(torch.load("checkpoint_diarization_3/best_f.pt"))
    # g_net.load_state_dict(torch.load("checkpoint_diarization_3/best_g.pt"))

    train_loader = getDataloader('train', 16, 10000, '/home/lizeqian/voxceleb/speaker_raw_dict.pkl', '/home/lizeqian/voxceleb/musan.txt')
    test_loader = getDataloader('test', 16, 1000, '/home/lizeqian/voxceleb/speaker_raw_dict.pkl', '/home/lizeqian/voxceleb/musan.txt')

    model = CompostionalEmbedding(f_net, g_net)
    
    tb_logger = pl_loggers.TensorBoardLogger('logs_fg_1018/')
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', filename='{epoch:05d}-{val_acc:.5f}', save_top_k=20, mode='max')
    reset_callback = ResetLearningRate()
    trainer = pl.Trainer(logger=tb_logger,callbacks=[checkpoint_callback, reset_callback],max_epochs=5000, gpus=1, resume_from_checkpoint='logs_fg/default/version_0/checkpoints/epoch=00309-val_acc=0.46293.ckpt')
    trainer.fit(model, train_loader, test_loader)