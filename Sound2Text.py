#%%
import os
import sys
import time
import math
import random
from pyparsing import C
from sympy import Float
import tqdm
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple
from IPython.display import Audio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa

import torch
from torch import nn, tensor
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

from torchmetrics.text import WordErrorRate

import csv
import wandb

# Arguments
device = 'cuda' if torch.cuda.is_available() else 'cpu'


seed = 8
wandb_enable = False
root = 'data/LJSpeech-1.1'
batch_size = 32
sample_rate = 22050
clip = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1
momentum = 0.9
wd = 1e-4
num_epochs = 20
root = 'M:/Git/Sound2Text/Dataset-LJSpeech'
root_wandb_key = 'M:/Git/Sound2Text/WandB_KeyFiles/key.txt'
seed = 42
batch_size = 16

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


################################   ################################
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

generator = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)

valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator)

waveforms, targets = next(iter(train_loader))
waveforms.shape, targets.shape # [64, 1, 220061] , [64, 176]
targets[0]


##################################   MODEL   ##########################################
class CNN2DFeatureExtractor(nn.Module):
  def __init__(self, input_channel=1, out_channels=[32, 64, 64]):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(input_channel, out_channels[0], kernel_size=11, stride=1, padding=5),
      nn.BatchNorm2d(out_channels[0]),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(out_channels[0], out_channels[1], kernel_size=11, stride=1, padding=5),
      nn.BatchNorm2d(out_channels[1])
    )
    self.layer3 = nn.Sequential(
      nn.Conv2d(out_channels[1], out_channels[2], kernel_size=11, stride=1, padding=5),
      nn.BatchNorm2d(out_channels[2]),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)
    )
  def forward(self, x): # [B, 1, 128, T] , channels is n_kernels = 1
    #print("1", x.shape) # [16, 1, 80, 808]
    x = self.layer1(x)  # [B, out_channels[0], 128/2, 804/2]
    #print("2", x.shape) # [16, 32, 40, 404]

    x = self.layer2(x)  # [B, out_channels[1], 64, 402]
    #print("3", x.shape) # [16, 64, 40, 404]

    x = self.layer3(x) # [B, out_channels[2], 64/2, 402/1]
    #print("4", x.shape)# [16, 64, 20, 404]

    return x


class ResNetFeatureExtractor(nn.Module):

  def __init__(self, ):
    super().__init__()

    self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    module_list = [nn.Conv2d(1, 64, 7, stride=(2, 1), padding=3, bias=False)]
    module_list += list(self.model.children())[1:-5]
    module_list += [nn.Conv2d(64, 32, 1, bias=False), 
                    nn.BatchNorm2d(32), 
                    nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*module_list)

  def forward(self, x):
    # [64, 1, 128, 804] -> [64, 32, 32, 402]
    x = self.model(x)
    return x
  
# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html  
class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

# test  
pos_encoding = PositionalEncoding(d_model=128, )
print(pos_encoding.pe.shape)
pos_encoding(torch.randn(200, 2, 128)).shape  
  
  
class TransformerModel(nn.Module):

  def __init__(self, embed_dim, n_head, num_encoders, num_decoders, dim_feedforward,
               len_vocab=43, dropout=0.1, activation=F.relu):
    super().__init__()

    self.embed_dim = embed_dim
    # Embedding
    #len_vocab = len(vocab)
    self.embedding = nn.Embedding(len_vocab, embedding_dim=embed_dim, padding_idx=0)

    # Position Encoding
    self.pos_encoder = PositionalEncoding(d_model=embed_dim)

    # Transformer
    self.transformer = nn.Transformer(
        d_model=embed_dim, nhead=n_head,
        num_encoder_layers=num_encoders, num_decoder_layers=num_decoders,
        dim_feedforward=dim_feedforward,
        dropout=dropout, 
        activation=activation
        )

    self.init_weights()

  def init_weights(self) -> None:
      initrange = 0.1
      self.embedding.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, tgt): # src: Audio features, tgt: text token ids
    # src: [S, B, D] = [404, 16, 1280]
    # tgt_before:[B ,  L]  -> tgt_after:[B , L  , D]     
    #            [16, 154] ->           [16, 154, 1280]
    tgt = self.embedding(tgt) * math.sqrt(self.embed_dim)
    
    
    tgt = tgt.permute(1, 0, 2) #[L(seq_len), B(batch_size), D(embed_dim)]
    # or tgt = tgt.permute(1, 0, 2) tgt.transpose(0, 1) or tgt = tgt.T
    tgt = self.pos_encoder(tgt)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt)).to(device)
    #inputs: src:(S, B, D), tgt:(L, B, D), tgt_mask:(L, L)-> output:(L, B, D)
    #src:[404,16,1280],tgt:[154,16,1280],tgt_mask:[154,154]->out:[154,16,1280]
    out = self.transformer(src, tgt, tgt_mask=tgt_mask)
    return out  
  
  
class Speech2Text(nn.Module):
  def __init__(self, len_vocab, d_model, n_head, num_encoders, num_decoders,
                dim_feedforward, dropout, activation, sample_rate, 
                inplanes, planes, n_mels=128, n_fft=1024, cnn_mode='resnet'):
    super().__init__()
    # PreProcessing
    self.voice_transform = nn.Sequential(
      torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000),
      torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft),
      torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    )
    
    # Feature embedding
    self.cnn_mode = cnn_mode
    if cnn_mode == 'simple':
      self.cnn = CNN2DFeatureExtractor(input_channel=1, 
                                       out_channels=[inplanes, planes, planes])
    elif cnn_mode == 'resnet':
      self.cnn = ResNetFeatureExtractor()
    else:
      raise NotImplementedError("Please select one of the simple or resnet model")
  
      # Transformer
    self.transformers = TransformerModel(
        embed_dim=d_model, n_head=n_head,
        num_encoders=num_encoders, num_decoders= num_decoders,
        dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

    # Classifier
    self.cls = nn.Linear(d_model, len_vocab)

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.cls.bias.data.zero_()
    self.cls.weight.data.uniform_(-initrange, initrange)
    
  # src: Voices[B, 1, T] , tgt: Texts[B, L] -> [B, S, len_vocab]
  def forward(self, src, tgt):  
    with torch.no_grad():
      # in_src: [B, 1, T=(List of air_pressure in time)]
      # out_src after voice_transform: [B, 1, n_mels, time_frames]
      # in: [16, 1, 222621] -> out: [16, 1, 80, 808]
      src = self.voice_transform(src)
    
    #if self.cnn_mode == 'resnet':
    # [16, 1, 80, 808] -> [16, 3, 80, 808]
    #  x = x.repeat(1, 3, 1, 1) 
    
    # in: [B , C,  F , T]   ----------> out: [B,  C', F',  T']
    # C' = planes, F'= F/H_stride=80/4=20, T'= T/W_stride=808/2=404
    # in: [16, 1, 80, 808] --CNN2D---> out: [16, 64, 80/4, 808/2]
    # in: [16, 1, 80, 808] -ResNet18-> out: [16, 32, 80/4, 808/2]
    src = self.cnn(src)       

    # B(batch_size), C(num_channels), 
    # F(freq_bins) , S(seq_len) = embedding dimension
    B, C, F, S = src.shape # [16, 64, 20, 404]
    # or src = src.permute(3, 0, 1, 2).contiguous().view(b, s, c*d)
    src = src.reshape(B, -1, S) # [B, C, F, S] -> [B, C*F, S]
    src = src.permute(2, 0, 1)  # [B, C*F, S]  -> [S, B, D(C*F)]
    
    # D=C*F = 64*20=1280(for CNN2D) or 32*20=640(for ResNet18)
    # in: (src: [S, B, D], tgt: [B, L]) -> out: [S, B, D]
    # src: [404, 16, 1280], tgt: [16, 154]-> out: [154, 16, 1280]
    #print("SRC shape before Transformer:", src.shape)
    #print("TGT shape before Transformer:", tgt.shape)
    out = self.transformers(src, tgt) 
    
    # [S, B, D] -> [B, S, D]
    # [154, 16, 1280] -> [16, 154, 1280]
    out = out.permute(1, 0, 2)

    # [B, S, D] -> [B, S, len_vocab]
    # [16, 154, 1280] -> [16, 154, 43]
    out = self.cls(out)
    return out





def postprocess(outputs, targets):
  generates, transcripts = [], []
  for output, target in zip(outputs, targets):
    # Generates
    g = ''.join(vocab.lookup_tokens(output.argmax(dim=-1).tolist()))
    generates.append(g)
    # Transcripts
    t = ''.join(vocab.lookup_tokens(target.tolist()))
    transcripts.append(t)
  return generates, transcripts


### Evaluation ###
def evaluate(model, test_loader, loss_fn, metric):
  model.eval()
  loss_eval = MeanMetric()
  metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets[:, :-1])

      loss = loss_fn(outputs.permute(0, 2, 1), targets[:, 1:])
      loss_eval.update(loss.item(), weight=len(targets))

      outputs, targets = postprocess(outputs, targets)
      metric.update(outputs, targets)

  return loss_eval.compute().item(), metric.compute().item()



def train_one_epoch(model, train_loader, loss_fn, optimizer, 
                    metric, clip, scheduler=None, epoch=None):
  model.train()
  loss_train = MeanMetric()
  metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets[:, :-1])

      loss = loss_fn(outputs.permute(0, 2, 1), targets[:, 1:])

      loss.backward()
      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=clip)

      optimizer.step()
      optimizer.zero_grad()

      if scheduler:
        scheduler.step()

      loss_train.update(loss.item(), weight=len(targets))

      outputs, targets = postprocess(outputs, targets)
      metric.update(outputs, targets)

      tepoch.set_postfix(loss=loss_train.compute().item(),
                         metric=metric.compute().item())

  return model, loss_train.compute().item(), metric.compute().item()






################## my Test##################



"""
batch = next(iter(train_loader))
batch[0].shape
batch[1].shape
"""




### 🟡 Step 2: Try to train and overfit the model on a small subset of the dataset.###
mini_train_size = 20
_, mini_train_dataset = random_split(train_set, 
                                     (len(train_set)-mini_train_size,
                                      mini_train_size))

mini_train_loader = DataLoader(mini_train_dataset, 
                               batch_size=batch_size, 
                               shuffle=True, collate_fn=collate_fn)

model = Speech2Text(len_vocab=43, d_model=640 , n_head=2, 
                    num_encoders=4, num_decoders=1,
                    dim_feedforward=512, dropout=0.1, 
                    activation=F.relu, cnn_mode='simple',
                    sample_rate=22050, inplanes=32, 
                    planes=32, n_mels=80, n_fft=400).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
metric = WordErrorRate().to(device)
num_epochs = 200
for epoch in range(num_epochs):
  model, loss_train, metric_value = train_one_epoch(model, mini_train_loader, 
                                                    loss_fn, optimizer, metric, 
                                                    1.0, None, epoch+1)
  if (epoch+1) % 10 == 0:
    print(
      f'Epoch {epoch+1}/{num_epochs}, Loss: {loss_train:.4f}, WER: {metric_value:.4f}')



### 🟡  Train the model for a limited number of epochs, 
#               experimenting with various learning rates. ###

num_epochs = 2

for lr in [1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]:
  print(f'LR={lr}')

  model = Speech2Text(len_vocab=43, d_model=640 , n_head=2, 
                      num_encoders=4, num_decoders=1,
                      dim_feedforward=512, dropout=0.1, 
                      activation=F.relu, cnn_mode='simple',
                      sample_rate=22050, inplanes=32, 
                      planes=32, n_mels=80, n_fft=400).to(device)

  optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
  loss_fn = nn.CrossEntropyLoss(ignore_index=0)
  metric = WordErrorRate().to(device)

  for epoch in range(num_epochs):
    model, loss_train, metric_value = train_one_epoch(model, valid_loader, 
                                                      loss_fn, optimizer, 
                                                      metric, 1.0, None, epoch+1)
    if (epoch+1) % 10 == 0:
      print(f'Epoch {epoch+1}/{num_epochs}, \
            Loss: {loss_train:.4f}, \
            WER: {metric_value:.4f}')

loss, metric_value = evaluate(model, mini_train_loader, loss_fn, metric)
waveforms, targets = next(iter(mini_train_loader))
with torch.no_grad():
  outputs = model(waveforms.to(device), targets[:, :-1].to(device))
outputs, targets = postprocess(outputs, targets)




### train + WANDB + eval + save best Model ###
wandb_enable = True
if wandb_enable:
  #wandb_arg_name = input('Please input  WandB argument name:')
  #print(wandb_arg_name)
  
  if os.path.exists(root_wandb_key):
    with open(root_wandb_key) as f:
        key = f.readline().strip()
    wandb.login(key=key)
  else:
      print("Key file does not exist. \
        Please create the key file with your wandb API key.")

  wandb.init(
  project="pytorch-wandb-Sound2Text",
  name="Sound2Text" + str(random.randint(1000,9999)),
  config={
      'lr': lr,
      'momentum': momentum,
      'batch_size': batch_size,
      #'seq_len': seq_len,
      #'hidden_dim': hidden_dim,
      #'embedding_dim': embedding_dim,
      #'num_layers': num_layers,
      #'dropout_embed': dropoute,
      #'dropout_in_lstm': dropouti,
      #'dropout_h_lstm': dropouth,
      #'dropout_out_lstm': dropouto,
      'clip': clip,
      }
  )
num_epochs = 30
loss_train_hist = []
loss_valid_hist = []

metric_train_hist = []
metric_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

model = Speech2Text(len_vocab=43, d_model=640 , n_head=2, 
                    num_encoders=4, num_decoders=1,
                    dim_feedforward=512, dropout=0.1, 
                    activation=F.relu, cnn_mode='simple',
                    sample_rate=22050, inplanes=32, 
                    planes=32, n_mels=80, n_fft=400).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
metric = WordErrorRate().to(device)

for epoch in range(num_epochs):
  # Train
  model, loss_train, metric_train = train_one_epoch(model,
                                                    train_loader,
                                                    loss_fn,
                                                    optimizer,
                                                    metric,
                                                    clip,
                                                    None,
                                                    epoch+1)
  # Validation
  loss_valid, metric_valid = evaluate(model,
                                      valid_loader,
                                      loss_fn,
                                      metric)

  loss_train_hist.append(loss_train)
  loss_valid_hist.append(loss_valid)

  metric_train_hist.append(metric_train)
  metric_valid_hist.append(metric_valid)

  if loss_valid < best_loss_valid:
    torch.save(model, f'model.pt')
    best_loss_valid = loss_valid
    print('Model Saved!')
    
    if wandb_enable:
      wandb.log({"metric_train": metric_train, "loss_train": loss_train,
                  "metric_valid": metric_valid, "loss_valid": loss_valid})

  print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}')
  epoch_counter += 1
wandb.finish()





### 🟡 Test model using a batch and calculate loss and WER .###
torch.cuda.empty_cache()
waveforms, targets = next(iter(test_loader))
len(waveforms), len(targets) # 16, 16
"""
model = Speech2Text(len_vocab=43, d_model=1280 , n_head=4, 
                    num_encoders=3, num_decoders=3,
                    dim_feedforward=512, dropout=0.1, 
                    activation=F.relu, cnn_mode='simple',
                    sample_rate=22050, inplanes=32, 
                    planes=64, n_mels=80, n_fft=400).to(device)
"""
# or
model = torch.load('model.pt').to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
with torch.no_grad():
  outputs = model(waveforms.to(device), targets.to(device))
  loss = loss_fn(outputs.permute(0, 2, 1), targets.to(device))

print(loss) # tensor(5.4840, device='cuda:0')
outputs.shape # [16, 154, 43]
generates, transcripts = postprocess(outputs, targets)

metric = WordErrorRate().to(device)
metric.reset()
metric.update(generates, transcripts)
final_wer = metric.compute()
print("Final WER =", final_wer)#tensor(1.9628, device='cuda:0')

metric.plot()
fig, ax = metric.plot()
import matplotlib.pyplot as plt
plt.show()




"""
from torchmetrics.text import WordErrorRate
metric = WordErrorRate().to(device)
metric.reset()

# batch اول
preds1 = ["hello word"]
target1 = ["hello world"]
metric.update(preds1, target1)

# batch دوم
preds2 = ["my name is Mani"]
target2 = ["my name is Mani"]
metric.update(preds2, target2)

final_wer = metric.compute()
print("Final WER =", final_wer)

metric.plot()

"""




x = tensor(np.arange(1280), dtype=torch.float32).view(1,1,128,10) # [4, 1, 128, 10]
x.shape

model = nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=5)
y = model(x)
y.shape # [1, 64, 128, 10]

model = CNN2DFeatureExtractor()
y = model(x)




model = ResNetFeatureExtractor().to(device)
with torch.no_grad():
  out = model(torch.randn((2, 3, 80, 796), device=device))
out.shape


######################################


print([1,2,3,4,5][:-1]) # [1,2,3,4]




























