import configparser
import json
import os
import sys

from collections import defaultdict
from functools import partial
from typing import Set, List, Any, Optional

from datetime import datetime
import socket
import time
import shutil
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from Radm import RAdam

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from DataSetGraphSimGenerator import DataSetGraphSimGenerator, CustomDataset
from tensorboard_logger import TensorBoardWritter
from model_batch import *

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
class ModelInferenceNoGcn:
    def __init__(self):
        # 初始化配置
        logging.error("device:{}".format(device))
        torch.set_printoptions(linewidth=120)
        torch.set_grad_enabled(True)
        np.random.seed(5)
        torch.manual_seed(0)

        # 超参数配置文件
        self.config = configparser.ConfigParser()
        self.config_file_path = os.path.join(os.path.dirname(__file__), "config_graph_sim_nogcn.ini")
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
        self.criterion = F.cross_entropy

    def __start_tb_logger(self, time_str):
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
        return GraphSimilarity_No_Gcn(input_dim=self.input_dim,
                                     gcn_hidden_dim=self.gcn_hidden_dim,
                                     linear_hidden_dim=self.linear_hidden_dim,
                                     pool_step=self.pool_step,
                                     num_bases=self.num_bases,
                                     dropout=self.dropout,
                                     support=self.support,
                                     max_node_num=self.max_node_num)

    def __print_data_info(self):
        train_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                   repeat_pos_data=self.repeat_pos_data, resplit=False)
        test_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="test",
                                  repeat_pos_data=self.repeat_pos_data, resplit=False)
        val_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="val",
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
        num_list.reverse()
        weight_ = torch.as_tensor(num_list, dtype=torch.float32, device=device)
        weight_ = weight_ / torch.sum(weight_)
        self.cross_weight_auto = np.array(weight_.cpu())
        return self.criterion(output, label, weight=weight_)

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
        # 优化器
        optimizer = RAdam(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        # 学习律
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                               patience=8, threshold=1e-4, threshold_mode="rel",
                                                               cooldown=0, min_lr=0, eps=1e-8)
        # 数据

        train_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                   repeat_pos_data=self.repeat_pos_data, resplit=self.resplit)
        self.__print_data_info()
        pos_train_num, neg_train_num = train_data.pos_neg_num()

        # 训练
        for epoch in range(self.epoch):
            if self.resplit_each_time:
                train_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode="train",
                                           repeat_pos_data=self.repeat_pos_data,resplit=self.resplit)
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
            if accuary_train >= 0.77 or (epoch >= 100 and epoch % 10 == 0 and accuary_train >= 0.98):
                self.save_model(time_str=start_time_train)

        self.test_val_model(mode="test")
        self.save_model(time_str=start_time_train)
        self.__stop_tb_logger()

    @torch.no_grad()
    def test_val_model(self, mode):
        test_data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode=mode,
                                   repeat_pos_data=self.repeat_pos_data, resplit=False)
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)
        accuary_all_num = 0
        preds_all_num = 0
        FN, FP, TN, TP = 0, 0, 0, 0
        loss_all = 0.0
        pred_class = list()
        pred_class_t3 = list()
        pred_class_t2 = list()
        pred_class_t5 = list()
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
            loss = self.criterion(outputs, labels, weight=torch.as_tensor(self.cross_weight_auto, dtype=torch.float32, device=device))
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
        time = end_time - start_time
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
        data = CustomDataset(data_set_id=self.data_set_id, dataset_version=self.dataset_version, max_node_num=self.max_node_num, mode=mode,
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
            # 记录最为相似的索引，即top1
            outputs_max_index = torch.argmax(outputs_sm.narrow(1, 1, 1))
            if save_error_excel and (outputs_max_index.item() != (torch.argmax(labels)).item()):
                preds_index = outputs_max_index.item()
                preds_name = online_data_info[3][preds_index]
                one_error = list(online_data_info[0:3]) + [preds_index, preds_name]
                labeled_error_data.append(one_error)
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


            pred_class.append(outputs_max_index)
            label_class.append(torch.argmax(labels))
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
    minf = ModelInferenceNoGcn()
    dataset_name = "train_ticket" if did == 1 else "sock_shop"
    all_dir = os.path.join(os.path.dirname(__file__), "..", "data", "graph_sim_model_parameters", dataset_name)
    error_info = list()
    for root, dirs, files in os.walk(all_dir):
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
    minf = ModelInferenceNoGcn()
    minf.train_model()
