# Copyright (c) Facebook, Inc. and its affiliates.

import re
import numpy as np
import torch
import json
import pandas as pd

class SpatioTemporalDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train):

        self.S_mean, self.S_std = self._standardize(train_set)

        S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean, torch.zeros(1, 1).to(self.S_mean)], dim=1)
        S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std, torch.ones(1, 1).to(self.S_std)], dim=1)

        self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_ for seq in (train_set if train else test_set)]

    def __len__(self):
        return len(self.dataset)

    def _standardize(self, dataset):
        dataset = [torch.tensor(seq) for seq in dataset]
        full = torch.cat(dataset, dim=0)
        S = full[:, 1:3]
        S_mean = S.mean(0, keepdims=True)
        S_std = S.std(0, keepdims=True)
        return S_mean, S_std

    def unstandardize(self, spatial_locations):
        return spatial_locations * self.S_std + self.S_mean

    def ordered_indices(self):
        lengths = np.array([seq.shape[0] for seq in self.dataset])
        indices = np.argsort(lengths)
        return indices, lengths[indices]

    def batch_by_size(self, max_events):
        try:
            from data_utils_fast import batch_by_size_fast
        except ImportError:
            raise ImportError('Please run `python setup.py build_ext --inplace`')

        indices, num_tokens = self.ordered_indices()

        if not isinstance(indices, np.ndarray):
            indices = np.fromiter(indices, dtype=np.int64, count=-1)
        num_tokens_fn = lambda i: num_tokens[i]

        return batch_by_size_fast(
            indices, num_tokens_fn, max_tokens=max_events, max_sentences=-1, bsz_mult=1,
        )

    def __getitem__(self, index):
        return self.dataset[index]


class Foursquare(SpatioTemporalDataset):
    def __init__(self,split="train",num=1000, args=None):
        p2id = {'Business and Professional Services':0,'Sports and Recreation':1, 'Travel and Transportation':2, 'Retail':3, 'Dining and Drinking':4, 'Landmarks and Outdoors':5, 'Arts and Entertainment':6, 'Community and Government':7, 'Health and Medicine':8}
        assert split in ["train", "val", "test"]
        with open('../data/Foursquare/data.json', 'r') as f:
            dataset = json.load(f)

        self.split = split

        dataset = [np.array([(i[0],i[2][0],i[2][1],p2id[i[1]]) for i in u]) for u in dataset]
        args.num_event = len(p2id)

        train_set = dataset[:int(len(dataset)*0.6)]
        if self.split == 'val':
            split_set = dataset[int(len(dataset)*0.6):int(len(dataset)*0.8)]
        elif  self.split == 'test':
            split_set = dataset[int(len(dataset)*0.8):]
        else:
            split_set = dataset[:int(len(dataset)*0.6)]

        if self.split == 'train':
            args.S_mean, args.S_std = self._standardize(train_set)
            
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class Mobile(SpatioTemporalDataset):
    def __init__(self,split="train",num=1000, args=None):
        

        p2id = {'购物': 0, '生活服务': 1, '公司': 2, '旅游度假': 3, '工厂农林': 4, '政府团体': 5, '教育培训': 6, '餐饮': 7, '医疗服务': 8, '酒店住宿': 9, '休闲运动': 10, '汽车服务': 11, '交通': 12, '住宅': 13}
        args.num_event = len(p2id)
        with open('../data/Mobile/data.json', 'r') as f:
            dataset = json.load(f)


        assert split in ["train", "val", "test"]

        self.split = split
        
        dataset = [np.array([(i[0],i[3][0],i[3][1],p2id[i[-1]]) for i in u]) for u in dataset]
        args.num_event = len(p2id)
    
        train_set = dataset[:int(len(dataset)*0.6)]
        if self.split == 'val':
            split_set = dataset[int(len(dataset)*0.6):int(len(dataset)*0.8)]
        elif  self.split == 'test':
            split_set = dataset[int(len(dataset)*0.8):]
        else:
            split_set = dataset[:int(len(dataset)*0.6)]


        if self.split == 'train':
            args.S_mean, args.S_std = self._standardize(train_set)

        
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"



def spatiotemporal_events_collate_fn(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)
        spatial_locations: (N, max_T, D)
        event_types: (N, max_T)
        mask: (N, max_T)
    """
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:3]
    event_types = data[:, :, 3]
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
    return event_times, spatial_locations, event_types, mask
