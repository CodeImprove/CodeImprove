from __future__ import absolute_import, division, print_function
import pwd
from statistics import variance
import sys
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
#from keras.preprocessing.sequence import pad_sequences

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
import subprocess
import shutil
#sys.path.append('/nas1-nfs1/home/rsr200002/CodeXGLUE/Code-Code/Defect-detection/code1')
from model import Model
import random
from torch import nn
from torch.nn import functional as F
from scipy.stats import entropy
#from Dissector import PVScore
#from Read_Count import get_count,read_format_code
#from pathlib import Path
import math
#from model import Model
#import code
#from code import mutualinfo
#from mutualinfo import mutual_information
from sklearn import preprocessing

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['input'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['id'],js['label'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

class ImportanceScore(nn.Module):
    def __init__(self, model, train_loader, dev_loader, train_y = None, args = None):
        super(ImportanceScore, self).__init__()

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

    def cal_newdissector(self,pred_vec, original_pred, lcr):
        pred_vec = F.softmax(pred_vec)
        print('pred_vec size: ', pred_vec.size())
        print("pred_vec", pred_vec)
        pred_order = torch.argsort(pred_vec, dim=1)
        print('pred_order size:', pred_order.size())
        sub_pred = pred_order[:, -1]
        sec_pos = pred_order[:, -2]
        # second_vec = pred_vec[torch.arange(len(pred_vec)), sec_pos].to(self.device)
        second_vec = pred_vec[torch.arange(len(pred_vec)), sec_pos]
        print("second_vec",second_vec)
        #print("second vec", second_vec.size())
        cor_index = torch.where(sub_pred == original_pred)
        err_index = torch.where(sub_pred != original_pred)

        print("cor_index", cor_index)
        print("err_index", err_index)


        # SVscore = torch.zeros([len(pred_vec)]).to(self.device)
        SVscore = torch.zeros([len(pred_vec)])
        print("SVscore", SVscore)
        #print("svscore vec", SVscore)
        # lsh = torch.zeros([len(pred_vec)]).to(self.device)
        lsh = torch.zeros([len(pred_vec)])
        print("lsh",lsh)
        entropy_Scores = self.cal_entropy(pred_vec)
        mi = self.mutual_information(pred_vec)
        bvsb = self.cal_bvsb(pred_vec)
        lsh[cor_index] = second_vec[cor_index]
        print("lsh",lsh)
        #print("LSH vec", lsh)
        # tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][cor_index].to(self.device)
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][cor_index]
        print("tmp", tmp)
        #print("tmp vec", tmp)
        ####SVscore[cor_index] = tmp / (tmp + lsh[cor_index])
        #SVscore[cor_index] = tmp + (tmp - lsh[cor_index])
        ###SVscore[cor_index] = (-tmp*torch.log(tmp)) + (-lsh[cor_index]*torch.log(lsh[cor_index])) - (-(tmp*torch.log(tmp)+lsh[cor_index]*torch.log(lsh[cor_index])))
        #changed SVscore[cor_index] = tmp - (tmp - lsh[cor_index])#entropy_Scores[cor_index]
        #SVscore[cor_index] = tmp*torch.log(tmp/lsh[cor_index])
        SVscore[cor_index] = tmp +bvsb[cor_index]
        #SVscore[cor_index] =  bvsb[cor_index]
        #print("svcore corindex", SVscore)
        print("SVscore", SVscore)

        lsh[err_index] = pred_vec[torch.arange(len(pred_order)), sub_pred][err_index]
        #print("lsh", lsh)
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][err_index]
        print("tmp", tmp)
        #####SVscore[err_index] = 1 - lsh[err_index] / (lsh[err_index] + tmp)
        #SVscore[err_index] = 1- (lsh[err_index] + (lsh[err_index] - tmp))
        #SVscore[err_index] = lsh[err_index] - (lsh[err_index] - tmp)
        ###SVscore[err_index] = 1- (-tmp*torch.log(tmp)) + (-lsh[err_index]*torch.log(lsh[err_index])) - (-(tmp*torch.log(tmp)+lsh[err_index]*torch.log(lsh[err_index])))
        #changed SVscore[err_index] =  lsh[err_index] + (lsh[err_index]-tmp)#(-lsh[err_index]*torch.log2(lsh[err_index]))/(tmp+ lsh[err_index])
        #SVscore[err_index] = -(lsh[err_index]*torch.log(lsh[err_index]/tmp))
        SVscore[err_index] = 1-(tmp + bvsb[err_index])
        #SVscore[err_index] =  bvsb[err_index]
        SVscore = (SVscore - torch.min(SVscore))/(torch.max(SVscore)- torch.min(SVscore))
        
        print("SVscore", SVscore)

        return SVscore


    def get_overal_uncertainty(self,uncertainty_score_list, snapshot, lcr,score_type=0):
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
        print("weights", weight)
        weight_svc = [uncertainty_score_list[i].view([-1]) * weight[i] for i in range(len(weight))]
        print("weight svc", weight_svc)
        weight_svc = torch.stack(weight_svc, dim=0)
        print("weight svc2", weight_svc)
        weight_svc = torch.sum(weight_svc, dim=0).view([-1])
        weight_svc = weight_svc
        print("weight svc3", weight_svc)
        
        ###normalizing the data


        #weight_svc = (weight_svc - torch.min(weight_svc))/(torch.max(weight_svc)- torch.min(weight_svc))
        #weight_svc = (weight_svc)/2
        print("normaized", weight_svc)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #weight_svc = min_max_scaler.fit_transform(weight_svc.reshape(-1,1))
        
        return weight_svc    



    def train_sub_model(self, lr, epoch):
        input_length = 768 * 400
        print(len(self.train_loader))
        layer_ids = [1,2,3,4,5,6,7, 8,9,10,11,12]
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

    def get_uncertainty(self, pred_y, y, lcr):
        svscore_list = []
        for sub_pred in pred_y:
            svscore = self.cal_newdissector(sub_pred, y, lcr)
            
            svscore_list.append(svscore.view([-1, 1]))
            print('svscore: ', svscore_list)
        svscore_list = torch.cat(svscore_list, dim=1)
        svscore_list = torch.transpose(svscore_list, 0, 1)
        return svscore_list

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

    def _uncertainty_calculate(self, data_loader):
        print('Dissector uncertainty evaluation ...')
        weight_list = [0,1,2]
        result = []
        logits, preds, y = self.predict_with_layers(data_loader)
        print(logits)
        print("---------------")
        print(preds)
        print("----------------")
        print(y)
        lcr_vals = self.get_label_change_rate(preds,y)
        uncertainty_score_list = self.get_uncertainty(logits, y, lcr_vals)
        #uncertainty_score_list = self.get_uncertainty_bvsb(logits)
        overal_uncertainty = self.get_overal_uncertainty(uncertainty_score_list, self.layer_ids, lcr_vals, 0).detach().cpu()
        print("total",overal_uncertainty)
        results = common_ten2numpy(overal_uncertainty)
        return results

    

    def cal_entropy(self,sub_pred):
        k = sub_pred.size(-1)
        #pred_prob = F.softmax(sub_pred, dim=-1) # (N, k)
        etp = entropy(sub_pred, axis=-1)/np.log(k) # np.ndarray
        etp = torch.tensor(etp)
        return etp


    def mutual_information(self, pred_vec):
        """
        Compute the mutual information as defined in [3] given a number of predictions. Thus, this metric expects
         a logit tensor of size batch_size x num_predictions x seq_len x output_size.
        [3] https://arxiv.org/pdf/1803.08533.pdf
        Parameters
        ----------
        logits: torch.FloatTensor
        Logits of the current batch.
        Returns
        -------
        torch.FloatTensor
        Mutual information for the current batch.
        """
        eps = 1e-5
        #logits, _, _ = common_predict_hidden(test_dataloader, model, device)
        #logits = torch.from_numpy(logits)
        #probs = torch.softmax(logits, dim=-1)
        #print(probs.shape)
        #mean_probs = probs.mean(dim=0).reshape(1, -1)
        #mutual_info = -(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + eps)).sum(dim=0) + (probs * torch.log(probs + eps)).sum(dim=0).mean(dim=0)
        #mutual_info = -(mean_probs * torch.log(mean_probs + eps)).sum() + (probs * torch.log(probs + eps)).sum().mean()
        mutual_info = -(pred_vec * torch.log(pred_vec.float() + eps)).sum(dim=1) + torch.log(torch.tensor([pred_vec.shape[1]], dtype=torch.float))
        return mutual_info
    
    

    
    

    def get_label_change_rate(self,pred,y):
        lcr = torch.zeros(len(y))
        print("y shape is ", y.shape[0])
        print("length of y is ", len(y))
        print("length of pred is ", len(pred))
        for preds in pred:
            for i in  range(len(y)):
                lcr[i] += (preds[i] !=y[i]).int() 
        print("lcr ratios",lcr)
        lcr = lcr/len(pred)
        print("lcr ratios",lcr)
        
        return lcr

    def cal_bvsb(self,sub_pred):
        max_values = []
        second_values = []

        #pred_vec = F.softmax(sub_pred)
        print("pred_vec",sub_pred)
        max_val, _ = torch.max(sub_pred, dim=1)
        for vec in sub_pred:
            sort_pred,_ = torch.sort(vec, descending = True)
            #print("sorted", sort_pred)
            second_max = sort_pred[1].item()
            #print("second_max", second_max)
            second_values.append(second_max)
        print("max_values",max_val) 
        
        second_values = torch.tensor(second_values)
        print("sec max", second_values)

        scores = max_val - second_values
        return scores
        

    """
    def cal_bvsb(self,sub_pred):
        max_values = []
        second_values = []

        pred_vec = F.softmax(sub_pred)
        print("pred_vec",pred_vec)
        max_val, _ = torch.max(pred_vec, dim=1)
        for vec in pred_vec:
            sort_pred,_ = torch.sort(vec, descending = True)
            #print("sorted", sort_pred)
            second_max = sort_pred[1].item()
            #print("second_max", second_max)
            second_values.append(second_max)
        print("max_values",max_val) 
        
        second_values = torch.tensor(second_values)
        print("sec max", second_values)

        scores = max_val - second_values
        return scores
    """

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




   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    parser.add_argument('--dissector', action='store_true')

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=4
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)


    """
    args = argparse.ArgumentParser().parse_args([])
    output_dir= "./saved_models"
    model_type="roberta"
    tokenizer_name= "microsoft/codebert-base"
    model_name_or_path= "microsoft/codebert-base"
    
    train_data_file="../dataset/train.jsonl"
    eval_data_file="../dataset/valid.jsonl"
    test_data_file="../dataset/test.jsonl"
    epoch = 5
    args.block_size =400 
    args.train_batch_size = 64 
    args.eval_batch_size  =32 
    args.learning_rate =2e-5 
    max_grad_norm =1.0 
    args.dropout_probability = 0

    torch.cuda.set_device(5)
    args.device = torch.device("cuda", torch.cuda.set_device(5))


    #load model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(model_name_or_path, output_hidden_states=True)
    config.num_labels=1
    #config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if model_name_or_path:
        model = model_class.from_pretrained(model_name_or_path, 
                                            from_tf=bool('.ckpt' in model_name_or_path),
                                            config=config)    
    else:
        model = model_class(config)

    #print(model)
    
    model=Model(model,config,tokenizer,args)
    
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))      
    model.to(args.device)
    model.eval()
  
    layer_ids = [1,2,3,4,5,6,7,8,9,10,11,12]

    #classifier_tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    #classifier_model = model_class.from_pretrained(model_name_or_path, num_labels=1, output_hidden_states=True)

    """
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    print("Loading {}".format(output_dir))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    
    # load dataset
    train_dataset = TextDataset(tokenizer, args,args.train_data_file)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    #train_features, train_y, train_pred = get_hiddenstate_last_layer(model, train_dataloader, args) 
    

    valid_dataset = TextDataset(tokenizer, args,args.eval_data_file)
    valid_sampler = SequentialSampler(valid_dataset) 
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
    

    test_dataset =TextDataset(tokenizer, args,args.test_data_file)
    test_sampler = SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    dissector = ImportanceScore(model=model, train_loader=train_dataloader, dev_loader=valid_dataloader, args=args)

    uncertainty_scores = dissector._uncertainty_calculate(test_dataloader)
    print(uncertainty_scores)

    with open(os.path.join("./saved_models/","changed_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,uncertainty_scores):
                    f.write(example.idx+'\t'+str(sc)+'\n')


    
    

























    

    




