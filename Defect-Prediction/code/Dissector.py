import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle

from typing import *
from tqdm import tqdm

from torch.nn import functional as F
import numpy as np
#from BasicalClass import BasicModule
#from BasicalClass import common_ten2numpy
#from BasicalClass import common_get_maxpos, common_predict, common_cal_accuracy
#from Metric import BasicUncertainty

#fix the main
#fix the common_predict
#add get_hiddenstate

def fit(net, x, y, optim, loss, device):
    net.train()
    x = x.to(device)
    y = y.to(device,dtype = torch.long).reshape(-1)
    pred = net(x)
    #pred = pred.squeeze()
    #print(pred)
    loss_val = loss(pred, y)
    optim.zero_grad()
    loss_val.backward()
    optim.step()
    return loss_val

class HidenDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.lenth = len(x)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.lenth


def build_loader(x, y, batch_size):
    dataset = HidenDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


class PVScore(nn.Module):
    def __init__(self, model, train_loader, dev_loader, train_y = None, args = None):
        super(PVScore, self).__init__()

        self.model = model
        self.device = args.device
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.train_y = train_y
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.eval_batch_size
        self.save_dir = '../code/saved_models'
        self.class_num = 2
        self.train_sub_model(lr=args.learning_rate, epoch=15)
        #self._uncertainty_calculate(self.dev_loader)


    def cal_svc(self, pred_vec, original_pred):
        pred_vec = F.softmax(pred_vec)
        print('pred_vec size: ', pred_vec.size())
        print("pred vec is ", pred_vec)
        pred_order = torch.argsort(pred_vec, dim=1)
        print('pred_order size:', pred_order.size())
        sub_pred = pred_order[:, -1]
        sec_pos = pred_order[:, -2]
        # second_vec = pred_vec[torch.arange(len(pred_vec)), sec_pos].to(self.device)
        second_vec = pred_vec[torch.arange(len(pred_vec)), sec_pos]
        #print("second vec", second_vec.size())
        cor_index = torch.where(sub_pred == original_pred)
        err_index = torch.where(sub_pred != original_pred)
        #print("corr index vec", cor_index)

        # SVscore = torch.zeros([len(pred_vec)]).to(self.device)
        SVscore = torch.zeros([len(pred_vec)])
        #print("svscore vec", SVscore)
        # lsh = torch.zeros([len(pred_vec)]).to(self.device)
        lsh = torch.zeros([len(pred_vec)])

        lsh[cor_index] = second_vec[cor_index]
        #print("LSH vec", lsh)
        # tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][cor_index].to(self.device)
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][cor_index]
        #print("tmp vec", tmp)
        SVscore[cor_index] = tmp / (tmp + lsh[cor_index])
        print("svcore corindex", SVscore)


        lsh[err_index] = pred_vec[torch.arange(len(pred_order)), sub_pred][err_index]
        #print("lsh", lsh)
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][err_index]
        SVscore[err_index] = 1 - lsh[err_index] / (lsh[err_index] + tmp)
        print("svscore err", SVscore)

        return SVscore

    @staticmethod
    def get_pvscore(sv_score_list, snapshot, score_type=0):
        snapshot = torch.tensor(snapshot, dtype=torch.float32)
        if score_type == 0:
            weight = snapshot
        elif score_type == 1:
            weight = torch.log(snapshot)
        elif score_type == 2:
            weight = torch.exp(snapshot)
        else:
            raise ValueError("Not supported score type")
        weight = weight / torch.sum(weight)
        weight_svc = [sv_score_list[i].view([-1]) * weight[i] for i in range(len(weight))]
        weight_svc = torch.stack(weight_svc, dim=0)
        weight_svc = torch.sum(weight_svc, dim=0).view([-1])
        return weight_svc

    def train_sub_model(self, lr, epoch):
        input_length = 768 * 400
        print(len(self.train_loader))
        layer_ids = [1,2,3,4,5,6,7, 8,9, 10, 11,12]
        self.layer_ids = layer_ids
        if self.try_load_submodel():
            return
        #layer_features, labels = self.load_hidden_state(layer_ids)
        #self.sub_layers = [nn.Linear(input_length, 2).to(self.device) for _ in layer_ids]
        self.sub_layers = [nn.Linear(input_length, 4).to(self.device) for _ in layer_ids]
        optimizers = [torch.optim.AdamW(layer.parameters(), lr = lr) for layer in self.sub_layers]
        my_loss = nn.CrossEntropyLoss()
        #my_loss = nn.BCEWithLogitsLoss()
        loss_vals = [0] * len(self.sub_layers)
        for epoch in range(epoch):
            for layer_features, labels in get_hiddenstate(self.model, layer_ids, self.train_loader, self.device, f"Training Epoch {epoch + 1}"):
                for i in range(len(layer_ids)):
                    loss_vals[i] = fit(self.sub_layers[i], layer_features[i], labels, optimizers[i], my_loss, self.device)
            _, pred_y, y = self.predict_with_layers(self.dev_loader)
            for idx, loss_val in enumerate(loss_vals):
                acc = common_cal_accuracy(pred_y[idx], y)
                print("Epoch {}, Layer {}, loss {}, accuracy {}".format(epoch, idx, loss_val, acc))
        for idx, layer_id in enumerate(layer_ids):
            save_path = self.get_submodel_path(layer_id)
            torch.save(self.sub_layers[idx], save_path)
            print("save sub model after layer {} in".format(layer_id), save_path)

    def try_load_submodel(self):
        self.sub_layers = [0] * len(self.layer_ids)
        for idx, id in enumerate(self.layer_ids):
            path = self.get_submodel_path(id)
            if os.path.exists(path):
                self.sub_layers[idx] = torch.load(path).to(self.device)
            else:
                return False
        return True



    def get_submodel_path(self, index):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        dir_name = self.save_dir + '/' + self.__class__.__name__ + '/'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        save_path = dir_name + str(index) + '.bin'

        return save_path

    def get_svscore(self, pred_y, y):
        svscore_list = []
        for sub_pred in pred_y:
            svscore = self.cal_svc(sub_pred, y)
            # print('svscore: ', svscore)
            svscore_list.append(svscore.view([-1, 1]))
        svscore_list = torch.cat(svscore_list, dim=1)
        svscore_list = torch.transpose(svscore_list, 0, 1)
        return svscore_list

    def _uncertainty_calculate(self, data_loader):
        print('Dissector uncertainty evaluation ...')
        weight_list = [0,1,2]
        result = []
        logits, _, y = self.predict_with_layers(data_loader)
        print("logits", logits)
        svscore_list = self.get_svscore(logits, y)
        for weight in weight_list:
            pv_score = self.get_pvscore(svscore_list, self.layer_ids, weight).detach().cpu()
            result.append(common_ten2numpy(pv_score))
        print(result)
        return result

    def predict_with_layers(self, dataloader):
        logits = [[] for _ in self.layer_ids]
        preds = [[] for _ in self.layer_ids]
        labels = []

        for layer_features, label in get_hiddenstate(self.model, self.layer_ids, dataloader, self.device, "Predicting with Layers"):
            for idx, features in enumerate(layer_features):
                self.sub_layers[idx].eval()
                logit, pred = layer_predict(self.sub_layers[idx], features, self.device)
                logits[idx].append(logit)
                preds[idx].append(pred)
            labels.append(label)

        labels = torch.cat(labels)
        return [torch.cat(l, 0) for l in logits], [torch.cat(p, 0) for p in preds], labels

def common_ten2numpy(a:torch.Tensor):
    return a.detach().cpu().numpy()

def layer_predict(model, x, device):
    with torch.no_grad():
        x = x.to(device)
        model = model.to(device)
        logits = model(x)
        pred_y = F.softmax(logits).argmax(dim = 1)
        #pred_y = torch.sigmoid(logits)
        #print(pred_y)
        #print(len(pred_y))
        #pred_y = (pred_y >0.5).float()
        #pred_y = pred_y.squeeze()
        #print(pred_y)
    return logits.detach().cpu(), pred_y.detach().cpu()

def common_cal_accuracy(pred_y, y):
    tmp = (pred_y.view([-1]) == y.view([-1]))
    acc = torch.sum(tmp.float()) / len(y)
    return acc

def get_hiddenstate(model, sub_ids, dataloader, device, message):
        for batch in tqdm(dataloader, desc=message):
            layer_features = [[] for _ in sub_ids]
            inputs = batch[0].to(device)
            label = batch[1].to(device)
            with torch.no_grad():
                outputs, hidden_state = model.foward_with_hidden_states(inputs)
                for idx, layer_id in enumerate(sub_ids):
                    hidden = hidden_state[layer_id]
                    layer_features[idx] = hidden.detach().reshape(len(hidden),-1).cpu()
            yield layer_features, label.detach().cpu()
