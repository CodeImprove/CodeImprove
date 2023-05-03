# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        #self.classifier=
        self.args=args
    
    """    
    def forward(self, input_ids=None,labels=None): 
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs=self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob = F.softmax(logits)
        #prob=torch.sigmoid(logits)
        if labels is not None:
            #labels=labels.float()
            loss_fct=CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            #loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            #loss=-loss.mean()
            return loss,prob
        else:
            return prob
    """
    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, hidden_states
        else:
            return prob, hidden_states