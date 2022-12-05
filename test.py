import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import easydict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import time
import math

import seaborn as sns
from pylab import rcParams
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
import copy

class Encoder(nn.Module):
  def __init__(self, config, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = int(seq_len), n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.config = config
    self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
    )
    self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
    )

  def forward(self, x):
    batch_size = x.shape[0]
    # print(f'ENCODER input dim: {x.shape}')
    # x = x.reshape((batch_size, self.seq_len, self.n_features)) # [batch_size, 1000, 1] # 원래
    x = self._sliding_window(x.shape[1], x, self.config.small_window, self.config.small_stride)
    # print(f'ENCODER reshaped dim: {x.shape}')

    x, (_, _) = self.rnn1(x)
    # print(f'ENCODER output rnn1 dim: {x.shape}')
    x, (hidden_n, _) = self.rnn2(x)
    # print(f'ENCODER output rnn2 dim: {x.shape}')
    # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
    # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
    return hidden_n.reshape((batch_size, self.embedding_dim))
  
  def _sliding_window(self, original_window, arr, window_size, stride):
    total_data = []

    # [16, 1000] -> [16, 330, 10]
    count = math.ceil((original_window - window_size) / stride)
    batch_size = arr.shape[0]
    for b in range(batch_size):
      slice_data = arr[b, :] # [1000]
      start_pt = 0
      datas = []
      for i in range(count):
        data = slice_data[start_pt : start_pt + window_size]
        start_pt = start_pt + stride
        datas.append(data)
      if start_pt < original_window:
        data = slice_data[-window_size:]
        datas.append(data)
      
      datas = torch.stack(datas, 0)
      total_data.append(datas)
    total_data = torch.stack(total_data, 0)  
    return total_data


class Decoder(nn.Module):
  def __init__(self, config, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = int(seq_len), input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True
    )
    self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    batch_size = x.shape[0]
    # print(f'DECODER input dim: {x.shape}')
    # x = x.repeat(self.seq_len, self.n_features) # todo testare se funziona con più feature
    x = x.repeat(self.seq_len, 1)
    # print(f'DECODER repeat dim: {x.shape}')
    x = x.reshape((batch_size, self.seq_len, self.input_dim))
    # print(f'DECODER reshaped dim: {x.shape}')
    x, (hidden_n, cell_n) = self.rnn1(x)
    # print(f'DECODER output rnn1 dim:/ {x.shape}')
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, config, seq_len, n_features, embedding_dim=64, device='cuda', batch_size=32):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(config, seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(config, seq_len, embedding_dim, n_features).to(device)
    # 추가
    self.seq_len = int(seq_len)
    self.n_features = n_features
    self.config = config

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    x = self._return_window(x)
    return x
  
  def _return_window(self, x):
    batch_size = x.shape[0]
    total_data = []
    for b in range(batch_size):
      data = x[b,:] # [331, 10]
      flatten_data = data[:self.seq_len,0:self.config.small_stride].flatten()
      
      length_fit = self.config.window_size - self.config.small_window
      if len(flatten_data) > length_fit:
        flatten_data  = flatten_data[:length_fit]
      flatten_data = torch.cat((flatten_data, data[-1,:].flatten()),dim=0)
      total_data.append(flatten_data)
    total_data = torch.stack(total_data, 0)
    total_data = total_data.reshape(batch_size, -1, 1)

    return total_data