import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#tensorboard 機動コマンド
#tensorboard --logdir runs/
import shutil
#shutil.copyfile("C:\\src\\src.txt", "C:\\dst\\dst.txt")

if __name__ == "__main__":

  cantkeep_path = '../dataset/dry_signals/cantKeep'
  cantraise_path = '../dataset/dry_signals/cantRaise'
  cantslow_path = '../dataset/dry_signals/cantSlow'
  cantstop_path = '../dataset/dry_signals/cantStop'
  cantstretch_path = '../dataset/dry_signals/cantStretch'

  #dir = Path(path)
  cantkeep_file_list = list(Path(cantkeep_path).iterdir())
  cantraise_file_list = list(Path(cantraise_path).iterdir())
  cantslow_file_list = list(Path(cantslow_path).iterdir())
  cantstop_file_list = list(Path(cantstop_path).iterdir())
  cantstretch_file_list = list(Path(cantstretch_path).iterdir())


  file_name = 1

  for idx,i in enumerate(cantkeep_file_list):
    if idx < 25:
      #print(i)
      shutil.copyfile(i,'../dataset/dry_signals/patience/'+ str(file_name) +'.csv')
      file_name +=1

  for idx,i in enumerate(cantraise_file_list):
    if idx < 25:
      #print(i)
      shutil.copyfile(i,'../dataset/dry_signals/patience/'+ str(file_name) +'.csv')
      file_name +=1

  for idx,i in enumerate(cantslow_file_list):
    if idx < 25:
      #print(i)
      shutil.copyfile(i,'../dataset/dry_signals/patience/'+ str(file_name) +'.csv')
      file_name +=1

  for idx,i in enumerate(cantstop_file_list):
    if idx < 25:
      #print(i)
      shutil.copyfile(i,'../dataset/dry_signals/patience/'+ str(file_name) +'.csv')
      file_name +=1

  for idx,i in enumerate(cantstretch_file_list):
    if idx < 25:
      #print(i)
      shutil.copyfile(i,'../dataset/dry_signals/patience/'+ str(file_name) +'.csv')
      file_name +=1


