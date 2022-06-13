from pyannote.audio.models import SincTDNN
from pyannote.audio.train.task import Task, TaskOutput, TaskType
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os, random
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm
import pickle, json
import math
import itertools
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from loss_functions import AngularPenaltySMLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 512

class GNet (nn.Module):
    def __init__ (self):
        super(GNet, self).__init__()
        self.linear1a = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.linear1b = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward (self, X1, X2):
        linear = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear

class LibriReadeload(Dataset):
    def __init__(self, step_size, speaker_dict, noise_list, mode='train'):
        with open(speaker_dict, 'r') as f:
            self.speaker_dict = json.load(f)
        self.spkers = list(self.speaker_dict.keys())

        with open(noise_list) as f: 
            lines = f.readlines()

        self.noise_list = []
        for line in lines:
            self.noise_list.append(line.strip())

        self.combinations = torch.tensor(list(itertools.combinations(list(range(5)),2)))
        self.step_size = step_size
        self.samplerate = 16000

    def addNoise(self, data, noise):
        noise = noise.astype(np.int32)

        if noise.shape[0] >= self.samplerate*2:
            st = random.randint(0, noise.shape[0] - self.samplerate*2)
            noise = noise[st:st + self.samplerate*2]
        else:
            noise = noise.repeat(math.ceil((self.samplerate*2)/noise.shape[0]))[:self.samplerate*2]
        coeff = random.uniform(0, 0.4)
        noise_energy = np.sqrt(np.sum((noise/32767)**2))
        data_energy = np.sqrt(np.sum((data/32767)**2))
        res = (data + coeff * noise * data_energy / noise_energy)*0.5
        return res.astype(np.int16)
    
    def getAudio(self, spker):
        audio_path = random.choice(self.speaker_dict[spker])
        samplerate, data = wavfile.read(audio_path)
        data = data.astype(np.int32)
        res = np.zeros(samplerate * 2)
        if data.shape[0] < samplerate * 2:
            res = data.repeat(math.ceil((samplerate*2)/data.shape[0]))[:samplerate*2]
        else:
            st = random.randint(0, data.shape[0] - samplerate * 2)
            res = data[st:st + samplerate * 2]
        return res.astype(np.int16)

    def getMixture(self, A, B):
        sampleA = self.getAudio(A)
        sampleB = self.getAudio(B)
        res = (sampleA + sampleB) * 0.5
        return res.astype(np.int16)

    def __getitem__(self, index):
        sel_speaker_ids = random.sample(range(len(self.spkers)), 5)
        test_samples = []
        ref_samples = []

        noise_path = random.choice(self.noise_list)
        samplerate, noise = wavfile.read(noise_path)
        
        for spker_id in sel_speaker_ids:
            spker = self.spkers[spker_id]
            sample = self.getAudio(spker)
            sample = self.addNoise(sample, noise)
            test_samples.append(sample)

        for spker_id in sel_speaker_ids:
            spker = self.spkers[spker_id]
            sample = self.getAudio(spker)
            sample = self.addNoise(sample, noise)
            ref_samples.append(sample)
        
        for comb in self.combinations:
            sample = self.getMixture(self.spkers[sel_speaker_ids[comb[0]]], self.spkers[sel_speaker_ids[comb[1]]])
            sample = self.addNoise(sample, noise)
            test_samples.append(sample)
        
        data = np.stack(test_samples)
        ref_samples = np.stack(ref_samples)
        return torch.tensor(data).unsqueeze(-1).float(), torch.tensor(sel_speaker_ids).long(), torch.tensor(ref_samples).unsqueeze(-1).float()

    def __len__(self):
        return self.step_size

def getDataloader(mode, batch_size, step_size, speaker_dict, noise_list):
    dataset = LibriReadeload(step_size, speaker_dict, noise_list, mode)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    return dataloader

def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    return torch.norm(D, p=2, dim=2)

class CompostionalEmbedding(pl.LightningModule):
    def __init__(self, mode):
        super(CompostionalEmbedding, self).__init__()
        task = Task(TaskType.REPRESENTATION_LEARNING,TaskOutput.VECTOR)
        specifications = {'X':{'dimension': 1} ,'task': task, 'y': {'classes': ['a', 'b']}}
        sincnet = {'instance_normalize': True, 'stride': [5, 1, 1], 'waveform_normalize': True}
        tdnn = {'embedding_dim': 512}
        embedding = {'batch_normalize': False, 'unit_normalize': False}
        self.f_net = SincTDNN(specifications, sincnet=sincnet, tdnn=tdnn, embedding=embedding)
        self.f_net.load_state_dict(torch.load('checkpoints/f_vxc.pt'))
        self.g_net = GNet().to(device)
        self.criterion2 = AngularPenaltySMLoss(512, 7263,self.g_net, s=10, m=0.05) 
        self.criterion = nn.MarginRankingLoss(0.1)
        self.combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))
        self.mode = mode

    def forward(self, x):
        batch_num, batch_size, _, _ = x.size()
        x = x.contiguous().view(batch_size * batch_num, 32000, 1)
        x = self.f_net(x).contiguous().view(batch_num, batch_size, -1)
        return x

    def training_step(self, batch, batch_idx):
        data, label, sup = batch
        data = torch.cat([data, sup], 1)
        data = data.float()
        embeddings = self.forward(data)
        running_loss = 0
        for emb_cnt, embedding in enumerate(embeddings):
            if self.mode == 'arcface':
                truth = torch.cat([label[emb_cnt], label[emb_cnt]], 0)
                validEmb = torch.cat([embedding[:5], embedding[-5:]], 0)
                running_loss += self.criterion2(validEmb, truth)
            elif self.mode == 'triplet':
                centroids = embedding[-5:]
                comb2_a = centroids[self.combinations2.transpose(-2, -1)[0]]
                comb2_b = centroids[self.combinations2.transpose(-2, -1)[1]]
                merged2 = self.g_net(comb2_a, comb2_b)
                truth = F.normalize(torch.cat([centroids, merged2]))
                preds = F.normalize(embedding[:15])
                dists = pairwiseDists(preds, truth)
                for d_cnt, dist in enumerate(dists):
                    if d_cnt < 5:
                        weight = 1
                    else:
                        weight = 0.5
                    running_loss += weight * self.criterion(dist[[e for e in range(15) if e != d_cnt]], dist[[d_cnt]*14], torch.ones(14, device=device))
        
        running_loss = running_loss/data.size(0)
        self.log('train_loss', running_loss)
        return running_loss

    def validation_step(self, batch, batch_idx):
        data, label, ref_data = batch
        data = data.float()
        ref_data = ref_data.float()
        embeddings = self.forward(torch.cat([data, ref_data], 1))

        all_cnt_1, all_cnt_2 = 0, 0
        hit_cnt_1, hit_cnt_2 = 0, 0
        all_cnt_only1, hit_cnt_only1 = 0, 0
        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[self.combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[self.combinations2.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)

            truth = F.normalize(torch.cat([centroids, merged2]))
            preds = F.normalize(embedding[:-5])
            dists = pairwiseDists(preds, truth)
            _, res = torch.topk(dists, 1, largest=False)
            _, res_only1 = torch.topk(dists[:5, :5], 1, largest=False)
            label = torch.tensor(list(range(15)), device=device)
            all_cnt_1 += 5
            all_cnt_2 += 10
            all_cnt_only1 += 5
            hit_cnt_only1 += torch.sum(label[:5] == res_only1.squeeze()[:5]).item()
            hit_cnt_1 += torch.sum(label[:5] == res.squeeze()[:5]).item()
            hit_cnt_2 += torch.sum(label[5:] == res.squeeze()[5:]).item()
        self.log('step_val_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('step_val_acc_only1set', hit_cnt_only1/all_cnt_only1)
        self.log('step_val_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('step_val_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))
        return all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2, hit_cnt_only1, all_cnt_only1
    
    def validation_epoch_end(self, outputs):
        all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2 = 0, 0, 0, 0
        all_cnt_only1, hit_cnt_only1 = 0, 0
        for output in outputs:
            all_cnt_1 += output[0]
            all_cnt_2 += output[1]
            hit_cnt_1 += output[2]
            hit_cnt_2 += output[3]
            hit_cnt_only1, all_cnt_only1 = output[4], output[5]
        self.log('val_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('val_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('val_acc_only1set', hit_cnt_only1/all_cnt_only1)
        self.log('val_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))

    def test_step(self, batch, batch_idx):
        data, label, ref_data = batch
        data = data.float()
        ref_data = ref_data.float()
        embeddings = self.forward(torch.cat([data, ref_data], 1))

        all_cnt_1, all_cnt_2 = 0, 0
        hit_cnt_1, hit_cnt_2 = 0, 0
        all_cnt_only1, hit_cnt_only1 = 0, 0
        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[self.combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[self.combinations2.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)

            truth = F.normalize(torch.cat([centroids, merged2]))
            preds = F.normalize(embedding[:-5])
            dists = pairwiseDists(preds, truth)
            _, res = torch.topk(dists, 1, largest=False)
            _, res_only1 = torch.topk(dists[:5, :5], 1, largest=False)
            label = torch.tensor(list(range(15)), device=device)
            all_cnt_1 += 5
            all_cnt_2 += 10
            all_cnt_only1 += 5
            hit_cnt_only1 += torch.sum(label[:5] == res_only1.squeeze()[:5]).item()
            hit_cnt_1 += torch.sum(label[:5] == res.squeeze()[:5]).item()
            hit_cnt_2 += torch.sum(label[5:] == res.squeeze()[5:]).item()
        self.log('step_test_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('step_test_acc_only1set', hit_cnt_only1/all_cnt_only1)
        self.log('step_test_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('step_test_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))
        return all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2, hit_cnt_only1, all_cnt_only1
    
    def test_epoch_end(self, outputs):
        all_cnt_1, all_cnt_2, hit_cnt_1, hit_cnt_2 = 0, 0, 0, 0
        all_cnt_only1, hit_cnt_only1 = 0, 0
        for output in outputs:
            all_cnt_1 += output[0]
            all_cnt_2 += output[1]
            hit_cnt_1 += output[2]
            hit_cnt_2 += output[3]
            hit_cnt_only1, all_cnt_only1 = output[4], output[5]
        self.log('val_acc_1set', hit_cnt_1/all_cnt_1)
        self.log('val_acc_2set', hit_cnt_2/all_cnt_2)
        self.log('val_acc_only1set', hit_cnt_only1/all_cnt_only1)
        self.log('val_acc', (hit_cnt_1 + hit_cnt_2)/(all_cnt_1+all_cnt_2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class ResetLearningRate(Callback):
    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        if pl_module.current_epoch // 30 == 0:
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                for p_idx, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = 1e-5
                

if __name__ == "__main__":

    train_loader = getDataloader('train', 32, 10000, 'voxceleb_train.json', 'musan_noise_files_list.txt')
    test_loader = getDataloader('test', 16, 1000, 'voxceleb_test.json', 'musan_noise_files_list.txt')

    model = CompostionalEmbedding(mode='triplet')
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dirpath = 'logs_new'

    for iteration in range(100):
        tb_logger = pl_loggers.TensorBoardLogger(f'{dirpath}/')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', filename='{epoch:05d}-{val_acc:.5f}', save_top_k=100, mode='max')
        if iteration > 0:
            model = model.load_from_checkpoint(f'{dirpath}/saved_model.ckpt', mode="triplet")
        trainer = pl.Trainer(logger=tb_logger,callbacks=[checkpoint_callback] , max_epochs=50, gpus=1)
        trainer.fit(model, train_loader, test_loader)
        trainer.save_checkpoint(f'{dirpath}/saved_model.ckpt')

        tb_logger = pl_loggers.TensorBoardLogger(f'{dirpath}/')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', filename='{epoch:05d}-{val_acc:.5f}', save_top_k=20, mode='max')
        model = model.load_from_checkpoint(f'{dirpath}/saved_model.ckpt', mode="arcface")
        trainer = pl.Trainer(logger=tb_logger,callbacks=[checkpoint_callback] , max_epochs=30, gpus=1)
        trainer.fit(model, train_loader, test_loader)
        trainer.save_checkpoint(f'{dirpath}/saved_model.ckpt')
