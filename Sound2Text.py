#%%
import os
import sys
import time
import math
import random
import tqdm
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple
from IPython.display import Audio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import optim

import torchvision
from torchvision import models

import torchaudio
import torchaudio.transforms

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torchmetrics as tm
from torchmetrics.aggregation import MeanMetric
from torchmetrics.text import WordErrorRate as WER

import csv

# Arguments
device = 'cuda' if torch.cuda.is_available() else 'cpu'

root = 'M:/Git/Sound2Text/Dataset-LJSpeech'
seed = 42
batch_size = 64

# Load Dataset
df = pd.read_csv(f'{root}/metadata.csv', 
                 delimiter='|', 
                 names=['id', 'transcript', 'normalized_transcript'],
                 quoting=csv.QUOTE_NONE)
df.describe()
df.head()
df.shape # (13100, 3)

# check Nan
df.isna().sum()

# check Duplicated
duplicates = df[df.transcript.duplicated(keep=False)].reset_index()

df['path'] = df['id'].apply(lambda x: f'{root}/wavs/{x}.wav') # (13100, 4)

# test a record
wav_id = random.randint(0, df.shape[0]-1)
wav_id, txt, norm_txt, wav_path = df.iloc[wav_id]
print(wav_id, txt, norm_txt, sep='\n')

waveform, sample_rate = torchaudio.load(wav_path)
print(waveform.shape, 
      waveform.dtype, 
      sample_rate, sep='\n', end='\n\n') # [1, 222621] , torch.float32, 22050
Audio(data=waveform, rate=sample_rate)

# Create train, validation, and test subsets from the dataset
generator = torch.Generator().manual_seed(seed)
train, valid, test = random_split(df,               # from torch.utils.data
                                  lengths=[0.75, 0.10, 0.15], 
                                  generator=generator)
# train.dataset, train.indices
len(train) # 9825
len(valid) # 1310
len(test)  # 1965

df_train = df.iloc[train.indices]
print(df_train.shape) # (9825, 4)
df_train.to_csv(f'{root}/train-subset.csv', index=False)

df_valid = df.iloc[valid.indices]
print(df_valid.shape) # (1310, 4)
df_valid.to_csv(f'{root}/valid-subset.csv', index=False)

df_test = df.iloc[test.indices]
print(df_test.shape) # (1965, 4)
df_test.to_csv(f'{root}/test-subset.csv', index=False)


################################   Main to run ################################
# Load csv and test & visualize it and Build a vocab
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

# Load train_csv
df_train = pd.read_csv(f'{root}/train-subset.csv')

# Test after load df_train
df_train.head()
idx = random.randint(0, len(df_train))
sample = df_train.iloc[idx]
waveform, sample_rate = torchaudio.load(sample['path'])
# waveform: [1, 194717]  ,  sample_rate: 22050
print(sample['transcript'])           # Accuracy of Weapon
print(sample['normalized_transcript'])# Accuracy of Weapon
Audio(data=waveform, rate=sample_rate)# from IPython.display

plt.plot(waveform[0])

plot_waveform(waveform, sample_rate)

plot_specgram(waveform, sample_rate)


# Build a vocab
from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(
    df_train.normalized_transcript.apply(lambda x: x.lower()),
    min_freq=10,
    specials=['=', '#', '<', '>'], special_first=True)
    # = padding                 ,   # Unknone, 
    # < start_of_sentence(sos)  ,   > end_of_sentence(eos)
vocab.set_default_index(1)
print(vocab.get_itos()) # ['=', '<', ' ', 'e', 't', 'a', ...
len(sorted(vocab.get_itos())) # 43
vocab(['<', 'a', 'c', '>'])   # [2, 7, 16, 3]
torch.save(vocab, 'vocab.pt')

################################################################################
# Compare the MelSpectrogram on CPU & GPU
sample = df_train.iloc[0]
waveform, sample_rate = torchaudio.load(sample['path'])#[1, 92829]

waveform = waveform.repeat(100, 1, 1)#.to(device) # [100, 1, 92829]

transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate).requires_grad_(False)#.to(device)

n = 100
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
s = time.time()

for i in range(n):
  with torch.no_grad():
    mel_specgram = transform(waveform)

end.record()
torch.cuda.synchronize()

print(start.elapsed_time(end)/n)# on GPU: 6.56ms / on CPU: 31.2

print(1e3*(time.time()-s)/n)    # on GPU: 6.57ms / on CPU: 31.1

print(mel_specgram.shape) # [100, 1, 128, 465]
################################################################################

# Calculate the time spent loading an audio file
start = time.time()

for path in df_train['path']:
  waveform, sample_rate = torchaudio.load(path)

end = time.time()

print(f'{end-start:.2f}s', f"{1e3*(end-start)/len(df_train['path']):.2f} ms")

# Calculate the total RAM occupation by dataset
total_nbytes = 0

for path in df_train['path']:
  waveform, _ = torchaudio.load(path)
  total_nbytes += waveform.nbytes

print(f'{total_nbytes/1e6} GB')
################################# From Here to run #####################################

# Custom dataset
class LJSpeechDataset(Dataset):
  def __init__(self, root: str, csv_file: str,
               input_transform: Optional[Callable] = None, # .MelSpectrogram(Voices)
               target_transform: Optional[Callable] = None,# vocab(text)
               memory: Optional[bool] = False):

    self.data = pd.read_csv(os.path.join(root, csv_file))
    self.phase = csv_file.split('-')[0].capitalize()
    self.input_transform = input_transform # .MelSpectrogram shift in Model
    self.target_transform = target_transform 
    self.memory = memory

    self.sos = target_transform(['<']) # 2
    self.eos = target_transform(['>']) # 3

    if memory:
      self._save_memory()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.memory:
      waveform = self.audios[idx].clone().squeeze()
    else:
      waveform = torchaudio.load(self.data.iloc[idx, 3])[0].squeeze()

    transcript = self.data.iloc[idx, 1]

    #self.target_transform is equall vocab(['a', 'c']) -> [7, 16]
    transcript = self.target_transform(list(transcript.lower()))
    transcript = self.sos + transcript + self.eos
    transcript = torch.LongTensor(transcript)

    return waveform, transcript

  def _save_memory(self):
    self.audios = []
    for path in self.data['path']:
      self.audios.append(self._load_audio(path))

  def _load_audio(self, path):
    return torchaudio.load(path)[0] # 0: waveform, 1: sample_rate

  def __repr__(self): # if print(train_set) will be displayed this
    return f"""Number of Datapoints: {len(self.data)}\nPhase: {self.phase}"""


vocab = torch.load('vocab.pt')
train_set = LJSpeechDataset(root=root, csv_file='train-subset.csv',
                            target_transform=vocab, memory=True)
print(train_set) # Number of Datapoints: 9825  ,  Phase: Train

valid_set = LJSpeechDataset(root=root, csv_file='valid-subset.csv', 
                            target_transform=vocab, memory=True)

test_set = LJSpeechDataset(root=root, csv_file='test-subset.csv', 
                           target_transform=vocab, memory=False)

wave1, text1 = train_set[1] # [129693] , [32413]
wave2, text2 =train_set[2]  # [98]     , [26]

# Dataloader
# for add pad to waves & texts
def collate_fn(batch):
  x, y = zip(*batch)
  x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0).unsqueeze(1)
  y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
  return x, y

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

waveforms, targets = next(iter(train_loader))
waveforms.shape, targets.shape # [64, 1, 220061] , [64, 176]
targets[0]

















































