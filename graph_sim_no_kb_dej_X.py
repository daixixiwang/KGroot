import os
import shutil
import socket
import sys
from datetime import datetime


from collections import defaultdict
from functools import partial
from typing import Set, List, Any, Optional

from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from Radm import RAdam

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from DataSetGraphSimGenerator import DataSetGraphSimGenerator
from tensorboard_logger import TensorBoardWritter
from model_batch import *


import configparser
import json
import logging
import os
import random
import sys
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import pickle

import torch
from torch.utils.data import Dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rank(y_true: Set[Any], y_pred: List[Any], max_rank: Optional[int] = None) -> List[float]:
    rank_dict = defaultdict(lambda: len(y_pred) + 1 if max_rank is None else (max_rank + len(y_pred)) / 2)
    for idx, item in enumerate(y_pred, start=1):
        if item in y_true:
            rank_dict[item] = idx
    return [rank_dict[_] for _ in y_true]

# noinspection PyPep8Naming
def MAR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([
        np.mean(get_rank(a, b, max_rank))
        for a, b in zip(y_true, y_pred)
    ])

def read_json_data(read_path):
    with open(read_path, 'r', encoding='utf-8') as file_reader:
        raw_data = file_reader.read()
        paths_list = json.loads(raw_data)
    return paths_list


def read_pickle_data(read_path):
    with open(read_path, 'rb') as file_reader:
        return pickle.load(file_reader)


def save_json_data(save_path, pre_save_data):
    with open(save_path, 'w', encoding='utf-8') as file_writer:
        raw_data = json.dumps(pre_save_data, indent=4)
        file_writer.write(raw_data)


def save_pickle_data(save_path, pre_save_data):
    with open(save_path, 'wb') as f:
        pickle.dump(pre_save_data, f, pickle.HIGHEST_PROTOCOL)

class CustomDatasetNoKB(Dataset):#需要继承data.Dataset
    #https://blog.csdn.net/liuweiyuxiang/article/details/84037973
    def __init__(self, data_set_id, dataset_version=None, mode="train", max_node_num=100, repeat_pos_data=0, resplit=False,
                 random_expand_train_data_flag=False, dataset_dir=None):
        self.data_set_id = data_set_id
        self.dataset_name = "train_ticket" if data_set_id == 1 else "sock_shop"
        dataset_special = "{}_{}".format(self.dataset_name, dataset_version) if dataset_version else self.dataset_name
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "data_for_graph_sim",
                                        dataset_special)

        if dataset_dir:
            self.dataset_dir = dataset_dir

        self.train_data_path = os.path.join(self.dataset_dir, "train_labeled_data.json")
        self.test_data_path = os.path.join(self.dataset_dir, "test_labeled_data.json")
        self.validation_data_path = os.path.join(self.dataset_dir, "validation_labeled_data.json")
        self.label_data_path = os.path.join(self.dataset_dir, "labeled_data.json")
        self.label_graph_class_path = os.path.join(self.dataset_dir, "labeled_graph_class_data.json")

        self.max_node_num = max_node_num
        self.repeat_pos_data = repeat_pos_data
        self.mode = mode
        if resplit or not os.path.exists(self.train_data_path):
            self.split_labeled_data()
            if random_expand_train_data_flag:
                self.random_expand_train_data(expand_num=9, delete_part=0.1)
            if self.repeat_pos_data:
                self.guocaiyang_train()
        if mode == "train":
            self.labeled_data = read_json_data(self.train_data_path)
        elif mode == "test":
            self.labeled_data = read_json_data(self.test_data_path)
        elif mode == "val":
            self.labeled_data = read_json_data(self.validation_data_path)
        else:
            self.labeled_data = read_json_data(self.label_data_path)

        # 在没有kb的场景中 只关心 正样本
        label_graph_class_data = read_json_data(self.label_graph_class_path)
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()

        df = pd.DataFrame(self.labeled_data, columns=["o_p", "f_p", "label"])
        self.labeled_data = df[df["label"]==1].values.tolist()
        self.class_names = error_name_list

    def __getitem__(self, index):
        online_path, kb_path, label = self.labeled_data[index]
        online_path = (os.path.dirname(__file__) + online_path).replace("\\", os.sep).replace("/", os.sep)
        kb_path = (os.path.dirname(__file__) + kb_path).replace("\\", os.sep).replace("/", os.sep)
        # print("online_path")
        # print(online_path)
        online_path = online_path.split('/')[-1]
        online_real_path = os.path.join("xxxx/pickle_data",online_path)
        graph_online = read_pickle_data(online_real_path)
        # print("graph_online")
        # print(graph_online)
        kb_path = kb_path.split('/')[-1]
        kb_real_path = os.path.join("xxxx/pickle_data",kb_path)
        graph_kb = read_pickle_data(kb_real_path)
        graph_online_feature, graph_online_A_list = self.process_graph(
            (graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"]))
        graph_kb_feature, graph_kb_A_list = self.process_graph(
            (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]))
        # 由类别序号作为label
        class_name = str(kb_path.split(os.sep)[-1].split("___")[0].replace("kb_",""))
        label = self.class_names.index(class_name)

        sample = {
            'graph_online_adj': torch.as_tensor(graph_online_A_list, dtype=torch.float32, device=device),
            'graph_online_feature': torch.as_tensor(graph_online_feature, dtype=torch.float32, device=device),
            'graph_kb_adj': torch.as_tensor(graph_kb_A_list, dtype=torch.float32, device=device),
            'graph_kb_feature': torch.as_tensor(graph_kb_feature, dtype=torch.float32, device=device),
            'label': torch.as_tensor(label, dtype=torch.long, device=device)
        }
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.labeled_data)

    def graph_class_data(self, use_best_path=False):
        """
        返回 一个online 图与所有 kb图 两两成对的数据
        :param use_best_path:
        :param graph_accuary: 小数 表示正确点数占比
        :param online_offline_proportion: 小数或整数 表示online:offline 's proportion
        :return:
        """
        online_path_in_mode = list(set([o_p for o_p, kb_p, label in self.labeled_data]))
        label_graph_class_data = read_json_data(self.label_graph_class_path)
        online_info = label_graph_class_data["online_info"]
        kb_data = label_graph_class_data["kb_data"]
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()

        for online_data_path, e_name, e_index in online_info:
            # 只加载该mode状态里的数据
            if online_data_path not in online_path_in_mode:
                continue
            online_path = (os.path.dirname(__file__) + online_data_path).replace("\\", os.sep).replace("/", os.sep)
            if use_best_path:
                online_path = online_path.replace(self.dataset_name, "{}_best".format(self.dataset_name))
            online_path = online_path.split('/')[-1]
            online_real_path = os.path.join("/home/mfm/experiment/kb_algorithm/graph_sim/data/data_for_graph_sim/train_ticket_final_same/pickle_data",online_path)

            graph_online = read_pickle_data(online_real_path)

            graph_online_feature, graph_online_A_list = self.process_graph(
                (graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"]))
            graph_online_adj_list, graph_online_feature_list, graph_kb_adj_list = list(), list(), list()
            graph_kb_feature_list, label_list = list(), list()

            for e_kb_name in [e_name]:
                kb_data_path = kb_data[e_kb_name]
                label = error_name_list.index(e_name)
                kb_path = (os.path.dirname(__file__) + kb_data_path).replace("\\", os.sep).replace("/", os.sep)
                if use_best_path:
                    kb_path = kb_path.replace(self.dataset_name, "{}_best".format(self.dataset_name))
                kb_path = kb_path.split('/')[-1]
                kb_real_path = os.path.join("/home/mfm/experiment/kb_algorithm/graph_sim/data/data_for_graph_sim/train_ticket_final_same/pickle_data",kb_path)
                
                graph_kb = read_pickle_data(kb_real_path)


                graph_kb_feature, graph_kb_A_list = self.process_graph(
                    (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]))
                graph_online_adj_list.append(torch.tensor(graph_online_A_list, dtype=torch.float32, device=device))
                graph_online_feature_list.append(torch.tensor(graph_online_feature, dtype=torch.float32, device=device))
                graph_kb_adj_list.append(torch.tensor(graph_kb_A_list, dtype=torch.float32, device=device))
                graph_kb_feature_list.append(torch.tensor(graph_kb_feature, dtype=torch.float32, device=device))
                label_list.append(torch.tensor(label, dtype=torch.long, device=device))
            sample = {
                'graph_online_adj': torch.stack(graph_online_adj_list),
                'graph_online_feature': torch.stack(graph_online_feature_list),
                'graph_kb_adj': torch.stack(graph_kb_adj_list),
                'graph_kb_feature': torch.stack(graph_kb_feature_list),
                'label': torch.stack(label_list),
                'error_name_list': error_name_list,
                'online_data_path': online_data_path,
            }

            yield sample, (online_data_path, e_name, e_index, error_name_list)

    def graph_class_data_bk(self):
        online_path_in_mode = list(set([o_p for o_p, kb_p, label in self.labeled_data]))
        label_graph_class_data = read_json_data(self.label_graph_class_path)
        online_info = label_graph_class_data["online_info"]
        kb_data = label_graph_class_data["kb_data"]
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()
        for online_data_path, e_name, e_index in online_info:
            # 只加载该mode状态里的数据
            if online_data_path not in online_path_in_mode:
                continue
            online_path = (os.path.dirname(__file__) + online_data_path).replace("\\", os.sep).replace("/", os.sep)
            graph_online = read_pickle_data(online_path)
            graph_online_feature, graph_online_A_list = self.process_graph(
                (graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"]))
            graph_online_adj_list, graph_online_feature_list, graph_kb_adj_list = list(), list(), list()
            graph_kb_feature_list, label_list = list(), list()
            for e_kb_name in error_name_list:
                kb_data_path = kb_data[e_kb_name]
                label = 1.0 if e_name == e_kb_name else 0.
                kb_path = (os.path.dirname(__file__) + kb_data_path).replace("\\", os.sep).replace("/", os.sep)
                graph_kb = read_pickle_data(kb_path)
                graph_kb_feature, graph_kb_A_list = self.process_graph(
                    (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]))
                graph_online_adj_list.append(torch.tensor(graph_online_A_list, dtype=torch.float32, device=device))
                graph_online_feature_list.append(torch.tensor(graph_online_feature, dtype=torch.float32, device=device))
                graph_kb_adj_list.append(torch.tensor(graph_kb_A_list, dtype=torch.float32, device=device))
                graph_kb_feature_list.append(torch.tensor(graph_kb_feature, dtype=torch.float32, device=device))
                label_list.append(torch.tensor(label, dtype=torch.long, device=device))
            sample = {
                'graph_online_adj': torch.stack(graph_online_adj_list),
                'graph_online_feature': torch.stack(graph_online_feature_list),
                'graph_kb_adj': torch.stack(graph_kb_adj_list),
                'graph_kb_feature': torch.stack(graph_kb_feature_list),
                'label': torch.stack(label_list),
                'error_name_list': error_name_list,
                'online_data_path': online_data_path,
            }
            assert torch.sum(sample["label"]).item() == 1.0

            yield sample, (online_data_path, e_name, e_index, error_name_list)

    def pos_neg_num(self):
        pos, neg = 0, 0
        for data in self.labeled_data:
            if data[2] == 1:
                pos += 1
            elif data[2] == 0:
                neg += 1
        return pos, neg

    def print_data_set_info(self):
        pos, neg = self.pos_neg_num()
        logging.error("dataset_id:{} mode:{} pos:{} neg:{}".format(self.data_set_id, self.mode,
                                                                   pos, neg))

    def random_expand_train_data(self, expand_num, delete_part=0.1):

        # 获取训练数据 中的online pickle ptha
        train_labeled_data = read_json_data(self.train_data_path)
        train_online_timepieces = list(set([data[0] for data in train_labeled_data]))
        df_raw = pd.DataFrame(train_labeled_data, columns=["o_p", "f_p", "label"])
        # 对于每个pickle 随机删除一些点和 边
        for piece in train_online_timepieces:
            # 匹配的 kb 和不匹配的kb
            match_kb = list(df_raw[(df_raw["o_p"]==piece) & (df_raw["label"]==1)]["f_p"])[0]
            no_match_kb = list(df_raw[(df_raw["o_p"]==piece) & (df_raw["label"]==0)]["f_p"])

            online_path = (os.path.dirname(__file__) + piece).replace("\\", os.sep).replace("/", os.sep)
            graph_online = read_pickle_data(online_path)
            for index in range(expand_num):
                graph_online_copy = deepcopy(graph_online)
                new_pickle_path = online_path.replace(".pickle", "diy{}.pickle".format(index))

                index_delete = random.sample(range(graph_online_copy["node_num"]), int(graph_online_copy["node_num"] * (1-delete_part)))
                id_delete = [graph_online_copy["node_index_id"][index] for index in index_delete]

                # 删除 结点关系字典
                for node in graph_online["nodes_dict"]:
                    if node["id"] in id_delete:
                        graph_online_copy["nodes_dict"].remove(node)

                for relation in graph_online["relations_dict"]:
                    if (relation["_start_node_id"] in id_delete) or (relation["_end_node_id"] in id_delete):
                        graph_online_copy["relations_dict"].remove(relation)
                graph_online_copy["relation_num"] = len(graph_online_copy["relations_dict"])
                graph_online_copy["node_num"] = len(graph_online_copy["nodes_dict"])

                for id in id_delete:
                    del graph_online_copy["node_id_index"][id]
                    del graph_online_copy["node_id_value"][id]

                graph_online_copy["node_index_id"] = np.delete(graph_online_copy["node_index_id"], np.array(index_delete), axis=0).tolist()
                graph_online_copy["node_index_value"] = np.delete(graph_online_copy["node_index_value"], np.array(index_delete), axis=0).tolist()

                graph_online_copy["fetures"] = np.delete(graph_online_copy["fetures"], np.array(index_delete), axis=0)

                graph_online_copy["adj"] = np.delete(graph_online_copy["adj"], np.array(index_delete), axis=0)
                graph_online_copy["adj"] = np.delete(graph_online_copy["adj"], np.array(index_delete), axis=1)

                loc = np.where(graph_online_copy["adj"]==1)
                graph_online_copy["adj_sparse"] = sp.csr_matrix((np.ones(loc[0].shape), (loc[0], loc[1])), shape=graph_online_copy["adj"].shape,
                                           dtype=np.int8)

                save_pickle_data(new_pickle_path, graph_online_copy)
                replace_part = str(os.path.dirname(__file__))
                replace_part = replace_part.replace("\\", os.sep).replace("/", os.sep)
                save_pickle_path = new_pickle_path.replace(replace_part, "")
                train_labeled_data.append([save_pickle_path, match_kb, 1])
                for kb in no_match_kb:
                    train_labeled_data.append([save_pickle_path, kb, 0])

        df_expand = pd.DataFrame(train_labeled_data, columns=["o_p", "f_p", "label"])
        assert (len(df_raw) * (expand_num+1)) == len(df_expand)
        save_json_data(self.train_data_path, train_labeled_data)
        if self.mode == "train":
            self.labeled_data = read_json_data(self.train_data_path)
        logging.warning("expand train data done! raw:{}+{}={} new{}+{}={} expand_num:{}".format(
            len(df_raw[df_raw["label"]==0]),
            len(df_raw[df_raw["label"]==1]),
            len(df_raw),
            len(df_expand[df_expand["label"]==0]),
            len(df_expand[df_expand["label"]==1]),
            len(df_expand),
            expand_num
        ))

    def guocaiyang_train(self):
        """过采样训练集正样本"""
        train_data = read_json_data(self.train_data_path)
        df_train = pd.DataFrame(train_data, columns=["o_p", "kb_p", "label"])
        pos_train = df_train[df_train["label"]==1]
        neg_train = df_train[df_train["label"]==0]

        pos_train = self.repeat_df(pos_train, self.repeat_pos_data, len(neg_train) // len(pos_train))
        train = np.array(pd.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        save_json_data(self.train_data_path, train)

    def repeat_df(self, df_data, repeat_pos_data_flag, neg_pos):
        if repeat_pos_data_flag >= 1:
            if repeat_pos_data_flag == 1:
                # 自适应
                if neg_pos > 1:
                    df_data = pd.concat([df_data] * neg_pos)
            else:
                df_data = pd.concat([df_data] * neg_pos)
        return df_data

    def split_labeled_data(self):
        all_data = read_json_data(self.label_data_path)
        import pandas as pd

        def repeat_df(df_data, repeat_pos_data_flag, neg_pos):
            if repeat_pos_data_flag >= 1:
                if repeat_pos_data_flag == 1:
                    # 自适应
                    if neg_pos > 1:
                        df_data = pd.concat([df_data] * neg_pos)
                else:
                    df_data = pd.concat([df_data] * neg_pos)
            return df_data

        df = pd.DataFrame(all_data, columns=["o_p", "kb_p", "label"])

        # # 方式1 将online时间段随机分为6;2:2
        # online_pickle_names = list(set(df["o_p"]))
        # random.shuffle(online_pickle_names)
        # occupy = [0.6, 0.2, 0.2]
        # train_num, test_num, val_num = int(occupy[0] * len(online_pickle_names)), int(occupy[1] * len(online_pickle_names)), int(
        #     occupy[2] * len(online_pickle_names))
        # train_online_names = online_pickle_names[:train_num]
        # test_online_names = online_pickle_names[train_num:train_num + test_num]
        # val_online_names = online_pickle_names[train_num + test_num:]
        # 方式2 需要确保每种故障类型 都能 包含训练\测试\验证数据
        train_online_names, test_online_names, val_online_names = list(), list(), list()
        kb_unique_names = list(set(df["kb_p"]))
        for kb_name in kb_unique_names:
            new_df = df[(df["kb_p"]==kb_name) & (df["label"]==1)]
            kb_online_names = list(set(new_df["o_p"]))
            random.shuffle(kb_online_names)
            occupy = [0.6, 0.3, 0.1]
            train_num, test_num, val_num = int(occupy[0] * len(kb_online_names)), int(
                occupy[1] * len(kb_online_names)), int(
                occupy[2] * len(kb_online_names))
            train_l = kb_online_names[:train_num]
            test_l = kb_online_names[train_num:train_num + test_num]
            val_l = kb_online_names[train_num + test_num:]
            train_online_names.extend(train_l)
            test_online_names.extend(test_l)
            val_online_names.extend(val_l)
            logging.info("kb_name:{}\n online_times:{} train:{} test:{} val:{}".format(kb_name, len(new_df), len(train_l), len(test_l), len(val_l)))

        df_train = pd.concat([df.loc[df["o_p"]==name] for name in train_online_names])
        df_test = pd.concat([df.loc[df["o_p"]==name] for name in test_online_names])
        df_val = pd.concat([df.loc[df["o_p"]==name] for name in val_online_names])
        # pos_df = repeat_df(pos_df, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_df = df[df["label"]==1]
        neg_df = df[df["label"]==0]

        pos_train = df_train[df_train["label"]==1]
        # pos_train = self.repeat_df(pos_train, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_test = df_test[df_test["label"]==1]
        # pos_test = repeat_df(pos_test, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_val = df_val[df_val["label"]==1]
        # pos_val = repeat_df(pos_val, self.repeat_pos_data, len(neg_df) // len(pos_df))

        neg_train = df_train[df_train["label"]==0]
        neg_test = df_test[df_test["label"]==0]
        neg_val = df_val[df_val["label"]==0]

        train = np.array(pd.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        test = np.array(pd.concat([pos_test, neg_test]).sample(frac=1)).tolist()
        val = np.array(pd.concat([pos_val, neg_val]).sample(frac=1)).tolist()

        save_json_data(self.train_data_path, train)
        save_json_data(self.test_data_path, test)
        save_json_data(self.validation_data_path, val)
        logging.info("split data done! all:{} train:{}({},{}) test:{}({},{}) val:{}({},{})".format(len(df),
                                                                                                   len(train),len(pos_train), len(neg_train),
                                                                                                   len(test), len(pos_test), len(neg_test),
                                                                                                   len(val), len(pos_val), len(neg_val)))

    def split_labeled_data_backup(self):
        all_data = read_json_data(self.label_data_path)
        import pandas as pd

        def repeat_df(df_data, repeat_pos_data_flag, neg_pos):
            if repeat_pos_data_flag >= 1:
                if repeat_pos_data_flag == 1:
                    # 自适应
                    if neg_pos > 1:
                        df_data = pd.concat([df_data] * neg_pos)
                else:
                    df_data = pd.concat([df_data] * neg_pos)
            return df_data

        df = pd.DataFrame(all_data, columns=["o_p", "kb_p", "label"])
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        # pos_df = repeat_df(pos_df, self.repeat_pos_data, len(neg_df) // len(pos_df))

        pos_df = pos_df.sample(frac=1)
        neg_df = neg_df.sample(frac=1)

        occupy = [0.6, 0.2, 0.2]
        train_num, test_num, val_num = int(occupy[0] * len(pos_df)), int(occupy[1] * len(pos_df)), int(occupy[2] * len(pos_df))
        pos_train = pos_df[:train_num]
        pos_train = repeat_df(pos_train, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_test = pos_df[train_num:train_num + test_num]
        pos_test = repeat_df(pos_test, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_val = pos_df[train_num + test_num:]
        pos_val = repeat_df(pos_val, self.repeat_pos_data, len(neg_df) // len(pos_df))

        train_num, test_num, val_num = int(occupy[0] * len(neg_df)), int(occupy[1] * len(neg_df)), int(occupy[2] * len(neg_df))
        neg_train = neg_df[:train_num]
        neg_test = neg_df[train_num:train_num + test_num]
        neg_val = neg_df[train_num + test_num:]

        train = np.array(pd.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        test = np.array(pd.concat([pos_test, neg_test]).sample(frac=1)).tolist()
        val = np.array(pd.concat([pos_val, neg_val]).sample(frac=1)).tolist()

        save_json_data(self.train_data_path, train)
        save_json_data(self.test_data_path, test)
        save_json_data(self.validation_data_path, val)
        logging.info("split data done! all:{} train:{} test:{} val:{}".format(len(df), len(train), len(test), len(val)))

    def process_graph(self, graph):
        """
        将一个feature矩阵和邻接矩阵 padding 后 转为 [feature, r_r, r_reverse, r_self]
        :param graph: (feature矩阵:np.array， 邻接稀疏矩阵: sp.csr_matrix, index_关键性：list)
        :param max_node_num:
        :return:
        """
        A_list = list()
        feature = graph[0]
        adj = graph[1]
        logging.info("process_graph_init: features:{} adj:{}".format(graph[0].shape, graph[1].shape))

        max_node_num = self.max_node_num
        if max_node_num - feature.shape[0] > 0:
            """ 需要 padding"""
            padding_num = max_node_num - feature.shape[0]
            # 处理特征
            feature_new = np.concatenate([feature, np.zeros((padding_num, feature.shape[1]))], axis=0)
            adj_new = np.concatenate([adj, np.zeros((padding_num, adj.shape[1]))], axis=0)
            adj_r = np.concatenate([adj_new, np.zeros((adj_new.shape[0], padding_num))], axis=1)
            adj_l = np.transpose(adj_r)

        elif max_node_num - feature.shape[0] < 0:
            """ 需要删除一些 """
            # 处理邻接矩阵
            num_delete = feature.shape[0] - max_node_num
            node_index_value = {_: graph[2][_] for _ in range(len(graph[2]))}
            for _ in graph[2]:
                # assert not isinstance(_, str)
                if isinstance(_, str):
                    node_index_value = {_: 0 for _ in range(len(node_index_value))}
                    logging.warning("some value is str!")
                    break

                node_index_value_sort = sorted(node_index_value.items(), key=lambda x: x[1])
                indexs_delete = np.array([node_index_value_sort[_][0] for _ in range(num_delete)])
                # 处理特征矩阵
                feature_new = np.delete(feature, indexs_delete, axis=0)
                # 处理邻接矩阵
                adj_r_array = adj
                adj_r_array = np.delete(adj_r_array, indexs_delete, axis=0)
                adj_r_array = np.delete(adj_r_array, indexs_delete, axis=1)

            adj_r = adj_r_array
            adj_l = np.transpose(adj_r)
        else:
            """形状不需要改变"""
            feature_new = feature
            adj_r = adj
            adj_l = np.transpose(adj_r)

        A_list.append(adj_r)
        A_list.append(adj_l)
        A_list.append(np.identity(adj_r.shape[0]))
        logging.info(
            "process_graph_done: features:{} adj_r:{} adj_l:{} self:{}".format(feature_new.shape, adj_r.shape,
                                                                               adj_l.shape, A_list[2].shape))
        return feature_new, np.array(A_list)


class ModelInferenceNoKb:
    def __init__(self):
        # 初始化配置
        logging.error("device:{}".format(device))
        torch.set_printoptions(linewidth=120)
        torch.set_grad_enabled(True)
        np.random.seed(5)
        torch.manual_seed(0)

        # 超参数配置文件
        self.config = configparser.ConfigParser()
        self.config_file_path = os.path.join(os.path.dirname(__file__), "config_graph_sim_nokb.ini")
        self.config.read(self.config_file_path, encoding='utf-8')

        # 超参数
        self.__load_super_paras()
        self.cross_weight_auto = None

        # 模型
        self.model = None
        self.model_saved_path = None
        self.model_saved_dir = None

        # tensor board类
        self.tb_comment = self.data_set_name
        self.tb_logger = None


        label_graph_class_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_for_graph_sim",
                                              "{}_{}".format(self.data_set_name, self.dataset_version), "labeled_graph_class_data.json")
        label_graph_class_data = read_json_data(label_graph_class_path)
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()

        self.class_names = error_name_list

        # 控制台打印
        # coloredlogs.install(
        #     level=self.logging_print_level,
        #     fmt="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        #     level_styles=LEVEL_STYLES,
        #     field_styles=FIELD_STYLES,
        #     logger=logger
        # )

        pass

    def __load_super_paras(self):
        self.data_set_id = self.config.getint("data", "DATASET")
        self.data_set_name = "train_ticket" if self.data_set_id==1 else "sock_shop"
        self.input_dim = self.config.getint("model", "input_dim")
        self.gcn_hidden_dim = self.config.getint("model", "gcn_hidden_dim")
        self.linear_hidden_dim = self.config.getint("model", "linear_hidden_dim")
        self.num_bases = self.config.getint("model", "num_bases")
        self.dropout = self.config.getfloat("model", "dropout")
        self.support = self.config.getint("model", "support")
        self.max_node_num = self.config.getint("model", "max_node_num")
        self.pool_step = self.config.getint("model", "pool_step")
        self.lr = self.config.getfloat("train", "LR")
        self.weight_decay = self.config.getfloat("train", "l2norm")
        self.resplit = self.config.getboolean("data", "resplit")
        self.batch_size = self.config.getint("data", "batch_size")
        self.resplit_each_time = self.config.getboolean("data", "resplit_each_time")
        self.repeat_pos_data = self.config.getint("data", "repeat_pos_data")
        self.dataset_version = self.config.get("data", "dataset_version")

        self.epoch = self.config.getint("train", "NB_EPOCH")
        self.user_comment = self.config.get("train", "comment")
        # self.cross_weight = self.config.getfloat("train", "cross_weight")
        # self.logging_print_level = str(self.config.get("print_logging", "level"))
        self.criterion = F.cross_entropy

        label_graph_class_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_for_graph_sim",
                                              "{}_{}".format(self.data_set_name, self.dataset_version),
                                              "labeled_graph_class_data.json")
        label_graph_class_data = read_json_data(label_graph_class_path)
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()

        self.class_names = error_name_list


    def __start_tb_logger(self, time_str):
        # self.start_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tb_log_dir = os.path.join(os.path.dirname(__file__), 'runs/%s' % time_str
                                       ).replace("\\", os.sep).replace("/", os.sep)
        self.tb_logger = TensorBoardWritter(log_dir="{}_{}{}".format(self.tb_log_dir, socket.gethostname(), self.tb_comment + self.user_comment),
                                            comment=self.tb_comment)

    def __stop_tb_logger(self):
        del self.tb_logger
        self.tb_logger = None

    def __print_paras(self, model):
        for name, param in model.named_parameters():
            logging.warning("name:{} param:{}".format(name, param.requires_grad))

    def generate_labeled_data(self):
        ds = DataSetGraphSimGenerator(data_set_id=self.data_set_id, dataset_version=self.dataset_version)
        ds.generate_dataset_pickle()
        del ds
        pass

    def __new_model_obj(self):
        return GraphSimilarity_No_KB(input_dim=self.input_dim,
                                     gcn_hidden_dim=self.gcn_hidden_dim,
                                     linear_hidden_dim=self.linear_hidden_dim,
                                     out_dim=len(self.class_names),
                                     pool_step=self.pool_step,
                                     num_bases=self.num_bases,
                                     dropout=self.dropout,
                                     support=self.support,
                                     max_node_num=self.max_node_num)

    def __print_data_info(self):
        train_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                       repeat_pos_data=self.repeat_pos_data, resplit=False)
        test_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="test",
                                      repeat_pos_data=self.repeat_pos_data, resplit=False)
        val_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="val",
                                     repeat_pos_data=self.repeat_pos_data, resplit=False)
        train_data.print_data_set_info()
        test_data.print_data_set_info()
        val_data.print_data_set_info()
        for datas in [train_data, test_data, val_data]:
            for index, data in enumerate(datas):
                adj_1 = np.array(data["graph_online_adj"].cpu())[0]
                f_1 = np.array(data["graph_online_feature"].cpu())
                adj_2 = np.array(data["graph_kb_adj"].cpu())[0]
                f_2 = np.array(data["graph_kb_feature"].cpu())
                self.tb_logger.writer.add_histogram("graph_online/adj", adj_1, index)
                self.tb_logger.writer.add_histogram("graph_online/feature", f_1, index)
                self.tb_logger.writer.add_histogram("graph_kb/adj", adj_2, index)
                self.tb_logger.writer.add_histogram("graph_kb/feature", f_2, index)

    def crossentropy_loss(self, output, label, num_list):
        """ num_list 表示 从 0,1,2,3每种类别的数目 本处只有两个类别[不相似,相似]"""
        # 方式1 直接翻转后除以总数
        num_list.reverse()
        weight_ = torch.as_tensor(num_list, dtype=torch.float32, device=device)
        weight_ = weight_ / torch.sum(weight_)
        # 方式2中值平均
        # weight_ = torch.as_tensor(num_list, dtype=torch.float32, device=device)
        # weight_ = torch.mean(weight_) * torch.rsqrt(weight_)
        self.cross_weight_auto = np.array(weight_.cpu())
        # return self.criterion(output, label, weight=torch.as_tensor([0.4,0.6], dtype=torch.float32, device=device))
        return self.criterion(output, label)

    def train_model(self):
        """训练并记录参数和模型"""
        start_time_train = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.__start_tb_logger(time_str= start_time_train)
        # 模型
        self.model = self.__new_model_obj()
        self.__print_paras(self.model)
        self.model = self.model.to(device)

        # 交叉熵
        criterion = self.crossentropy_loss
        optimizer = RAdam(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                               patience=8, threshold=1e-4, threshold_mode="rel",
                                                               cooldown=0, min_lr=0, eps=1e-8)

        train_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                       repeat_pos_data=self.repeat_pos_data, resplit=self.resplit)
        self.__print_data_info()
        pos_train_num, neg_train_num = train_data.pos_neg_num()

        # 训练
        for epoch in range(self.epoch):
            if self.resplit_each_time:
                train_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                               repeat_pos_data=self.repeat_pos_data, resplit=self.resplit)
            train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
            loss_all = 0
            accuary_all_num = 0
            preds_all_num = 0
            FN, FP, TN, TP = 0, 0, 0, 0
            batch_num = 0
            outputs_all = list()
            for batch in train_loader:
                graphs_online = (batch["graph_online_feature"], batch["graph_online_adj"])
                graphs_offline = (batch["graph_kb_feature"], batch["graph_kb_adj"])
                labels = batch["label"]
                outputs = self.model(graphs_online, graphs_offline)
                outputs_all.append(outputs)
                loss = criterion(outputs, labels, num_list=[neg_train_num, pos_train_num])

                preds = torch.argmax(outputs, dim=1)
                accuary_all_num += torch.sum(preds == labels)
                preds_all_num += torch.as_tensor(labels.shape[0])
                FN += int(torch.sum(preds[labels == 1] == 0))
                FP += int(torch.sum(preds[labels == 0] == 1))
                TN += int(torch.sum(preds[labels == 0] == 0))
                TP += int(torch.sum(preds[labels == 1] == 1))

                batch_num += 1
                loss_all += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step(loss_all)
            sample_data = None
            if epoch == 0:
                batch = next(iter(train_loader))
                graphs_online = (batch["graph_online_feature"], batch["graph_online_adj"])
                graphs_offline = (batch["graph_kb_feature"], batch["graph_kb_adj"])
                sample_data = (graphs_online, graphs_offline)


            recall_train = TP / (TP + FN) if (TP + FN) else 0
            precision_train = TP / (TP + FP) if (TP + FP) else 0
            F1_train = ((2 * precision_train * recall_train) / (precision_train + recall_train)) if (precision_train and recall_train) else 0
            accuary_val, loss_all_val, base_ac_val, precision_val, recall_val, F1_val = self.test_val_model(mode="val")
            accuary_test, loss_all_test, base_ac_test, precision_test, recall_test, F1_test = self.test_val_model(mode="test")
            class_ac_train = self.judge_graph_class_ac(mode="train")
            class_ac_val = self.judge_graph_class_ac(mode="val")
            class_ac_test = self.judge_graph_class_ac(mode="test")
            accuary_train = accuary_all_num.item() / preds_all_num.item()
            info_dict = dict(
                sample_data=sample_data, step=epoch, loss=loss_all.item(),
                loss_val=loss_all_val.item(),
                loss_test=loss_all_test.item(),
                accuracy=accuary_train,
                accuary_val=accuary_val,
                accuary_test=accuary_test,
                outputs_all=torch.cat(outputs_all, dim=0),
                train_pos_neg=np.array([pos_train_num/len(train_data), neg_train_num/len(train_data)]),
                val_pos_neg=np.array(base_ac_val),
                test_pos_neg=np.array(base_ac_test),
                cross_weight_auto=self.cross_weight_auto,
                class_ac_train=class_ac_train,
                class_ac_val=class_ac_val,
                class_ac_test=class_ac_test,
                recall_train=recall_train,
                recall_val=recall_val,
                recall_test=recall_test,
                precision_train=precision_train,
                precision_val=precision_val,
                precision_test=precision_test,
                F1_train=F1_train,
                F1_val=F1_val,
                F1_test=F1_test
            )
            self.tb_logger.print_tensoroard_logs(model=self.model, info_dict=info_dict)

            logging.error("epoch:{} loss:{} accuracy:{}/{}={}".format(epoch, loss_all, accuary_all_num, preds_all_num,
                                                                      int(accuary_all_num) / int(preds_all_num)))
            if accuary_train >= 0.79 or (epoch >= 100 and epoch % 10 == 0 and accuary_train >= 0.7):
                self.save_model(time_str=start_time_train)

        self.test_val_model(mode="test")
        # 保存模型和超参数
        self.save_model(time_str=start_time_train)
        self.__stop_tb_logger()

    @torch.no_grad()
    def test_val_model(self, mode):
        test_data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode=mode,
                                      repeat_pos_data=self.repeat_pos_data, resplit=False)
        # test_data.print_data_set_info()
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)
        accuary_all_num = 0
        preds_all_num = 0
        FN, FP, TN, TP = 0, 0, 0, 0
        loss_all = 0.0
        pred_class_t3 = list()
        pred_class_t2 = list()
        pred_class_t5 = list()
        pred_class = list()
        label_class = list()
        time = 0
        pos_num, neg_num = test_data.pos_neg_num()
        for batch in test_loader:
            graphs_online = (batch["graph_online_feature"], batch["graph_online_adj"])
            graphs_offline = (batch["graph_kb_feature"], batch["graph_kb_adj"])
            labels = batch["label"]
            start_time = time.perf_counter() 
            outputs = self.model(graphs_online, graphs_offline)
            end_time = time.perf_counter() 
            # logging.error("out{}  label{} negnum{}".format(type(outputs), type(labels), type(neg_num)))
            loss = self.criterion(outputs, labels)
            loss_all += loss
            outputs_sm = torch.nn.functional.softmax(outputs, dim=1)
            # 记录最为相似的索引，即top1
            outputs_max_index = torch.argmax(outputs_sm.narrow(1, 1, 1))
            # if save_error_excel and (outputs_max_index.item() != (torch.argmax(labels)).item()):
            #     preds_index = outputs_max_index.item()
            #     preds_name = online_data_info[3][preds_index]
            #     one_error = list(online_data_info[0:3]) + [preds_index, preds_name]
            #     labeled_error_data.append(one_error)
            # 记录top_3
            outputs_sq = torch.squeeze(outputs_sm.narrow(1, 1, 1))
            top_3 = torch.topk(outputs_sq, k=3)[1]
            if torch.argmax(labels) in top_3:
                pred_class_t3.append(torch.argmax(labels))
            else:
                pred_class_t3.append(top_3[0])

            # 记录top2
            top_2 = torch.topk(outputs_sq, k=2)[1]
            if torch.argmax(labels) in top_2:
                pred_class_t2.append(torch.argmax(labels))
            else:
                pred_class_t2.append(top_2[0])
            # record top5
            top_5 = torch.topk(outputs_sq, k=5)[1]
            if torch.argmax(labels) in top_5:
                pred_class_t5.append(torch.argmax(labels))
            else:
                pred_class_t5.append(top_5[0])
            pred_class.append(outputs_max_index)
            label_class.append(torch.argmax(labels))
            preds = torch.argmax(outputs, dim=1)
            accuary_all_num += torch.sum(preds == labels)
            preds_all_num += torch.as_tensor(labels.shape[0])
            FN += int(torch.sum(preds[labels==1]==0))
            FP += int(torch.sum(preds[labels==0]==1))
            TN += int(torch.sum(preds[labels==0]==0))
            TP += int(torch.sum(preds[labels==1]==1))
        recall = TP / (TP + FN) if (TP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        time = (end_time - start_time)*1000
        F1 = ((2 * precision * recall) / (precision + recall)) if (precision and recall) else 0
        pred_class_s = torch.stack(pred_class)
        label_class_s = torch.stack(label_class)
        pred_class_t3_s = torch.stack(pred_class_t3)
        pred_class_t2_s = torch.stack(pred_class_t2)
        pred_class_t5_s = torch.stack(pred_class_t5)
        MAR_res = MAR(label_class_s, pred_class_s)
        top1_a = torch.sum(pred_class_s == label_class_s).item() / pred_class_s.size()[0]
        top3_a = torch.sum(pred_class_t3_s == label_class_s).item() / pred_class_t3_s.size()[0]
        top2_a = torch.sum(pred_class_t2_s == label_class_s).item() / pred_class_t2_s.size()[0]
        top5_a = torch.sum(pred_class_t5_s == label_class_s).item() / pred_class_t5_s.size()[0]
        logging.error("{}_data : accuracy1:{}/{}={}  accuracy2:{} accuracy3:{} accuracy5:{} MAR: {} precision:{}/{}={}  recall:{}/{}={}  F1:{} time: {}".format(
            mode, accuary_all_num, preds_all_num, int(accuary_all_num) / int(preds_all_num),top2_a,top3_a,top5_a,MAR_res,
            TP, (TP + FP), precision,
            TP, (TP + FN), recall,
            F1,time
        ))
        pos, neg = test_data.pos_neg_num()
        base_ac = [pos/(pos+neg), neg/(pos+neg)]
        return int(accuary_all_num) / int(preds_all_num), loss_all, base_ac, precision, recall, F1
        pass

    @torch.no_grad()
    def judge_graph_class_ac(self, mode, save_error_excel=False, all_test=False):
        assert self.model
        data = CustomDatasetNoKB(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode=mode,
                                 repeat_pos_data=self.repeat_pos_data, resplit=False)
        pred_class = list()
        pred_class_t3 = list()
        pred_class_t2 = list()
        label_class = list()

        labeled_error_data = list()
        for sample, online_data_info in data.graph_class_data():
            """(online_data_path, e_name, e_index, error_name_list)"""
            graphs_online = (sample["graph_online_feature"], sample["graph_online_adj"])
            graphs_offline = (sample["graph_kb_feature"], sample["graph_kb_adj"])
            labels = sample["label"]
            outputs = self.model(graphs_online, graphs_offline)
            outputs_sm = torch.nn.functional.softmax(outputs, dim=1)
            outputs_sq = np.squeeze(outputs_sm)
            label_err_index = sample["label"][0]

            # 记录最为相似的索引，即top1
            outputs_max_index = torch.argmax(outputs_sq)
            if save_error_excel and (outputs_max_index.item() != label_err_index):
                preds_index = outputs_max_index.item()
                preds_name = online_data_info[3][preds_index]
                one_error = list(online_data_info[0:3]) + [preds_index, preds_name]
                labeled_error_data.append(one_error)
            # 记录top_3
            top_3 = torch.topk(outputs_sq, k=3)[1]
            if label_err_index in top_3:
                pred_class_t3.append(label_err_index)
            else:
                pred_class_t3.append(top_3[0])

            # 记录top2
            top_2 = torch.topk(outputs_sq, k=2)[1]
            if label_err_index in top_2:
                pred_class_t2.append(label_err_index)
            else:
                pred_class_t2.append(top_2[0])


            pred_class.append(outputs_max_index)
            label_class.append(label_err_index)
        pred_class_s = torch.stack(pred_class)
        label_class_s = torch.stack(label_class)
        pred_class_t3_s = torch.stack(pred_class_t3)
        pred_class_t2_s = torch.stack(pred_class_t2)

        top1_a = torch.sum(pred_class_s == label_class_s).item() / pred_class_s.size()[0]
        top3_a = torch.sum(pred_class_t3_s == label_class_s).item() / pred_class_t3_s.size()[0]
        top2_a = torch.sum(pred_class_t2_s == label_class_s).item() / pred_class_t2_s.size()[0]
        if save_error_excel:
            excel_anme = os.path.join(os.path.dirname(__file__), "{}_error_label_{}.xls".format(self.data_set_name, mode))
            df = pd.DataFrame(labeled_error_data, columns=["online_data_path", "e_name", "e_index", "preds_index", "preds_name"])
            df.to_excel(excel_anme, index=False)
            if all_test:
                return df
        return top1_a, top3_a, top2_a




    def save_model(self, time_str):
        dir_name = "{}_{}".format(time_str, socket.gethostname()+self.user_comment)
        save_path_dir = os.path.join(os.path.dirname(__file__), "..", "data", "graph_sim_model_parameters", self.data_set_name,
                                 dir_name)
        os.makedirs(save_path_dir, exist_ok=True)
        self.model_saved_path = os.path.join(save_path_dir, "model.pth")
        self.model_saved_dir = save_path_dir
        torch.save(self.model.state_dict(), self.model_saved_path)
        shutil.copy(self.config_file_path, os.path.join(save_path_dir, "config_graph_sim.ini"))

    def load_model(self, model_saved_dir):
        "https://blog.csdn.net/dss_dssssd/article/details/89409183"
        if model_saved_dir:
            self.model_saved_dir = model_saved_dir
            self.model_saved_path = os.path.join(model_saved_dir, "model.pth")
            self.config_file_path = os.path.join(model_saved_dir, "config_graph_sim.ini")
            self.config.read(self.config_file_path, encoding='utf-8')
            self.__load_super_paras()

        self.model = self.__new_model_obj()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_saved_path, map_location=device))
        self.model.eval()
        pass

def get_error_summary(did = 1):
    model_paths = [
        "20200423-095359_amaxD2_n100step100dataset70attetionjumpReduceLROnPlateauLRRAdmweight46",
        "20200423-092330_amaxD2_n100step100dataset70attetionjumpReduceLROnPlateauLRRAdmweight46",
        "20200423-084810_amaxD2_n100step100dataset70attetionjumpReduceLROnPlateauLRRAdm",
        "20200423-081103_amaxD2_n100step100dataset70attetionjumpCosineAnnealingLRRAdm",
        "20200423-132639_amaxD2_n100step100dataset70allattetionjumpReduceLROnPlateauLRRAdm"
    ]
    minf = ModelInferenceNoKb()
    dataset_name = "train_ticket" if did == 1 else "sock_shop"
    all_dir = os.path.join(os.path.dirname(__file__), "..", "data", "graph_sim_model_parameters", dataset_name)
    error_info = list()
    for root, dirs, files in os.walk(all_dir):
        # if all_dir.find(root) == -1 :
        #     break
        for dir in dirs:
            if dir not in model_paths:
                continue
            model_dir = os.path.join(all_dir, dir)
            minf.load_model(model_saved_dir=model_dir)
            df_train = minf.judge_graph_class_ac(mode="train", save_error_excel=True, all_test=True)
            df_test = minf.judge_graph_class_ac(mode="test", save_error_excel=True, all_test=True)
            df_val = minf.judge_graph_class_ac(mode="val", save_error_excel=True, all_test=True)
            df_all = pd.concat([df_train, df_test, df_val])
            online_data_path = list(set((df_all["online_data_path"])))
            online_data_path.sort()
            e_name = list(set(df_all["e_name"]))
            e_name.sort()
            error_info.append(dict(
                model_name=dir,
                online_names=online_data_path,
                error_names=e_name
            ))
    online_name_sets = [set(info["online_names"]) for info in error_info]
    error_name_sets = [set(info["error_names"]) for info in error_info]
    online_final = online_name_sets[0]
    error_final = error_name_sets[0]
    for _ in range(1, len(online_name_sets)):
        online_final.intersection(online_name_sets[_])
    for _ in range(1, len(error_name_sets)):
        error_final.intersection(error_name_sets[_])
    error_info.insert(0, dict(
        online_names_jiaoset=sorted(list(online_final)),
        e_name_jiaoset=sorted(list(error_final))
    ))

    save_json_data(os.path.join(os.path.dirname(__file__), "{}_error_info.json".format(dataset_name)), error_info)


def save_json_data(save_path, pre_save_data):
    with open(save_path, 'w', encoding='utf-8') as file_writer:
        raw_data = json.dumps(pre_save_data, indent=4)
        file_writer.write(raw_data)



if __name__ == '__main__':
    minf = ModelInferenceNoKb()
    minf.train_model()
