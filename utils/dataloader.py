from itertools import chain

import numpy as np
import torch
import time
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_unique_max_len(datas):
    return max([len(set(data)) for data in datas])


class DataSet(Dataset):
    def __init__(self, rawData):
        self.data = rawData[0]
        self.target = rawData[1]
        self.max_length = compute_max_len(self.data)
        self.unique_max_length = compute_unique_max_len(self.data)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # padding
        data = torch.cat(
            [torch.tensor(self.data[index]), torch.zeros(self.max_length - len(self.data[index]), dtype=torch.int64)])
        label = torch.tensor(self.target[index]) - 1
        mask = torch.cat([torch.ones([len(self.data[index])], dtype=torch.float32),
                          torch.zeros(self.max_length - len(self.data[index]), dtype=torch.float32)], dim=0)
        # build graph
        unique_node, alias_index = torch.unique(data, return_inverse=True)
        unique_node_mask = torch.concat(
            [unique_node != 0,
             torch.zeros(self.unique_max_length - unique_node.shape[0], dtype=torch.bool)],
            dim=-1)
        item = torch.concat(
            [unique_node, torch.zeros(self.unique_max_length - unique_node.shape[0], dtype=torch.int64)], dim=-1)
        A = torch.zeros((self.unique_max_length, self.unique_max_length), dtype=torch.float32)
        for i in range(data.shape[0] - 1):
            if data[i + 1] == 0:
                break
            u = torch.where(item == data[i])[0][0]
            v = torch.where(item == data[i + 1])[0][0]
            A[u][v] = 1  # 可以赋权值试试
        A_in_sum = torch.sum(A, 0)
        A_in_clip = torch.clip(A_in_sum, 1)
        A_in = torch.divide(A, A_in_clip)
        A_out_sum = torch.sum(A, 1)
        A_out_clip = torch.clip(A_out_sum, 1)
        A_out = torch.divide(A.transpose(1, 0), A_out_clip)
        A = torch.concat([A_in, A_out], 0).transpose(1, 0)
        return alias_index, A, item, label, mask, unique_node_mask


def create_relation_graph(batch):
    alias_index = torch.stack([batch[i][0] for i in range(len(batch))], dim=0)
    A = torch.stack([batch[i][1] for i in range(len(batch))], dim=0)
    item = torch.stack([batch[i][2] for i in range(len(batch))], dim=0)
    label = torch.stack([batch[i][3] for i in range(len(batch))], dim=0)
    mask = torch.stack([batch[i][4] for i in range(len(batch))], dim=0)
    unique_node_mask = torch.stack([batch[i][5] for i in range(len(batch))], dim=0)

    # unique_node_len = torch.sum(unique_node_mask, dim=-1)
    batch_size = mask.shape[0]
    # matrix = torch.zeros([batch_size, batch_size], dtype=torch.float32)
    # item = item.to('cuda')
    # unique_node_mask = unique_node_mask.to('cuda')
    matrix = np.zeros([batch_size, batch_size], dtype=np.float32)
    # start = time.time()
    # for i in range(batch_size - 1):
    #     target_item = item[i]
    #     test_item = item[i + 1:]
    #     union_len = torch.tensor([torch.unique(torch.concat([test_item[j], target_item], dim=-1)).shape[0] - 1 for j in
    #                              range(len(test_item))], device='cuda')
    #     intersection = torch.sum(torch.logical_and(torch.isin(test_item, target_item), unique_node_mask[i + 1:]),
    #                              dim=-1)
    #     relation_weights = intersection / union_len
    #     matrix[i, i + 1:] = relation_weights
    #     matrix[i + 1:, i] = relation_weights
    #
    # matrix = matrix + torch.eye(batch_size, dtype=torch.float32)
    # degree = torch.diag(1.0 / torch.sum(matrix, dim=-1))
    # for i in range(batch_size - 1): 13.1
    #     target_item = item[i]
    #     for j in range(i + 1, batch_size):
    #         unique_len = torch.unique(torch.concat([item[j], target_item], dim=-1)).size(0) - 1
    #         intersection = torch.sum(torch.logical_and(torch.isin(item[j], target_item), unique_node_mask[j]), dim=-1)
    #         relation_weights = intersection / unique_len
    #         matrix[i][j] = relation_weights
    #         matrix[j][i] = relation_weights
    # for i in range(batch_size - 1): 3.3
    #     seq_a = set(np.asarray(item[i]))
    #     seq_a.discard(0)
    #     for j in range(i + 1, batch_size):
    #         seq_b = set(np.asarray(item[j]))
    #         seq_b.discard(0)
    #         overlap = seq_a.intersection(seq_b)
    #         ab_set = seq_a | seq_b
    #         matrix[i][j] = float(len(overlap)) / float(len(ab_set))
    #         matrix[j][i] = matrix[i][j]
    # for i in range(batch_size - 1): 6.9
    #     target_item = np.asarray(item[i])
    #     for j in range(i + 1, batch_size):
    #         test_item = np.asarray(item[j])
    #         unique_len = len(np.union1d(test_item, target_item)) - 1
    #         intersection = len(np.intersect1d(test_item, target_item)) - 1
    #         relation_weights = intersection / unique_len
    #         matrix[i][j] = relation_weights
    #         matrix[j][i] = relation_weights
    unique_node_mask = np.asarray(unique_node_mask)  # 0.99
    for i in range(batch_size - 1):
        target_item = np.asarray(item[i])
        test_item = np.asarray(item[i + 1:])
        union_len = np.asarray([np.unique(np.concatenate((test_item[j], target_item), axis=-1)).shape[0] - 1 for j in
                                range(len(test_item))])
        intersection = np.sum(np.logical_and(np.isin(test_item, target_item), unique_node_mask[i + 1:]), axis=-1)
        relation_weights = intersection / union_len
        matrix[i, i + 1:] = relation_weights
        matrix[i + 1:, i] = relation_weights
    # end_time = time.time()
    matrix = matrix + np.eye(batch_size, dtype=np.float32)
    degree = np.diag(1.0 / np.sum(matrix, axis=-1))
    # matrix = matrix + torch.eye(batch_size, dtype=torch.float32)
    # degree = torch.diag(1.0 / torch.sum(matrix, dim=-1))
    matrix = torch.from_numpy(matrix)
    degree = torch.from_numpy(degree)
    # print('total time:', end_time - start)
    return alias_index, A, item, label, mask, matrix, degree


def compute_item_num(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    items_num = len(np.unique(seq_in_1D))
    return items_num


def compute_max_len(sequence):
    len_list = [len(seq) for seq in sequence]
    return np.max(len_list)


def split_train_val(train_data, split_rate=0.1):
    session_total = len(train_data[0])
    split_num = int(session_total * split_rate)

    val_index = np.random.choice(a=np.arange(0, session_total), size=split_num, replace=False)
    np.random.shuffle(val_index)
    val_data = ([train_data[0][index] for index in val_index], [train_data[1][index] for index in val_index])

    train_index = np.setdiff1d(np.arange(0, session_total), val_index)
    train_data_new = ([train_data[0][index] for index in train_index], [train_data[1][index] for index in train_index])

    return train_data_new, val_data
