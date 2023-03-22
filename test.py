# import torch
# import tqdm, gc, time
# from sklearn.metrics import roc_auc_score, log_loss
# from torch.utils.data import DataLoader
# from models.emb_MLPs import *

# from dataset import AvazuDataset, Movielens1MDataset, CriteoDataset


# def get_dataset(name, path):
#     if name == 'movielens1M' or name == 'movielens1M_inter':
#         return Movielens1MDataset(path)
#     elif name == 'avazu':
#         return AvazuDataset(path)
#     elif name == 'criteo':
#         return CriteoDataset(path)

# import pandas as pd
# aa = pd.read_csv('/root/AutoField/avazu/train')
# print(aa)
import numpy as np
import pandas as pd
import lmdb
import torch.utils.data
from tqdm import tqdm
from torch.utils.data import Dataset
import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

path = '/root/AutoField/avazu/train.csv'
with open(path) as f:
    f.readline()
    pbar = tqdm(f, mininterval=1, smoothing=0.1)
    pbar.set_description('Create avazu dataset cache: setup lmdb')
    for line in pbar:
        values = line.rstrip('\n').split(',')
        if len(values) != self.NUM_FEATS + 2:
            continue
        np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
        np_array[0] = int(float(values[1]))
        for i in range(1, self.NUM_FEATS + 1):
            np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])
        buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
        item_idx += 1
        # if item_idx % buffer_size == 0:
            # yield buffer
            # buffer.clear()
    # yield buffer


#test