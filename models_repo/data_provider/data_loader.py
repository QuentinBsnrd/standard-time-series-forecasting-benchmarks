import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class PiezoSeqDatasetScaled(Dataset):
    def __init__(self, series, input_len=336, output_len=360, split='train', split_ratio=(0.5, 0.1, 0.4), scale=True):
        assert split in ['train', 'val', 'test']
        self.input_len = input_len
        self.output_len = output_len
        self.scale = scale
        self.split = split

        total_len = len(series)
        n_train = int(total_len * split_ratio[0])
        n_val = int(total_len * split_ratio[1])

        self.scaler = StandardScaler()
        if scale:
            self.scaler.fit(series[:n_train])
            self.series = self.scaler.transform(series)
        else:
            self.series = series

        if split == 'train':
            start, end = 0, n_train
        elif split == 'val':
            start, end = n_train - input_len - output_len, n_train + n_val
        else:  # 'test'
            start, end = n_train + n_val - input_len - output_len, total_len

        self.series = torch.tensor(self.series[start:end], dtype=torch.float32)
        self.samples = []
        self.start_indices = []  # To find the original series position

        for i in range(len(self.series) - input_len - output_len):
            x = self.series[i:i+input_len]
            y = self.series[i+input_len:i+input_len+output_len]
            self.samples.append((x, y))
            self.start_indices.append(i + start)  # Absolute index in the original series

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        start_idx = self.start_indices[idx]
        return x, y, start_idx

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def load_data_train_val_test(args, file):
    df = pd.read_csv("data/"+file)
    df["date"] = pd.to_datetime(df["date"])
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    series = df.values.astype('float32')

    split_ratio = (args.split_ratio, 0.1, 1 - args.split_ratio - 0.1)

    train_dataset = PiezoSeqDatasetScaled(series, input_len=args.input_len, output_len=args.pred_len, split='train', split_ratio=split_ratio)
    val_dataset = PiezoSeqDatasetScaled(series, input_len=args.input_len, output_len=args.pred_len, split='val', split_ratio=split_ratio)
    test_dataset = PiezoSeqDatasetScaled(series, input_len=args.input_len, output_len=args.pred_len, split='test', split_ratio=split_ratio)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader, train_dataset.scaler