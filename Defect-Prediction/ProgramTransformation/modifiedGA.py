#masked language prediction function in pre-trained models
"""
    code snippet with several tokens-> context information to predict potential values of masked tokens. 
    generate candidate subsitutues foe each variable. 
    pick subsitutues that are semantically similar  (compute contextualized embeddings of these tokens and calculate cosine similarity with the original tokens)
    rank these candidates and select top-k candidates as natural substitutes. 

1. select variables to rename
2. selecting substitutions
3. deciding whether to accept to replace the variable with selected substitution. 

Naturalness aware substitutions: (MLP and contexualized embedding)
    Convert code snippets into format. 
    generate potential substitutes for each sub-token in sequence using MLP-> produxe ranked list-> select top j substitutes.
    replace sub-tokens in original with candidate sub-tokens-> fetch embeddings and concatenate new embeddings and compute cosine similarity->select the top-k ranked in decscending order


-Transform all programs. code A using 14 transformations: 
-Compute fitness for all 14 codes and select K/2 codes. 
-Interchange transformations of 2 components (if 1 is for loop then do for loop in 2)
-Mutate some random components 

"""

from __future__ import absolute_import, division, print_function
from mimetypes import init
import pwd
import sys
import argparse
import glob
import logging
import os
import pickle
import random
import datetime
import shutil
import re

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
from emsemble import ImportanceScore
#from Read_Count import get_count,read_format_code
from pathlib import Path
import math
#from model import Model
#import code
#from code import mutualinfo
#from mutualinfo import mutual_information

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

#from model import Model
cpu_cont = multiprocessing.cpu_count()
path = ('Path/to/CLONEGEN/CodeTransformationTest')
code_path = os.path.abspath('Path/to/ProgramTransformation/directory')
c_filename = 'motivation.c'
mutatedfile = 'Mutated.c'
program = os.path.join(path, c_filename)
mutated = os.path.join(path,mutatedfile)



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def saveCode(js, code_id):
    #code_sets = json.load(open('../dataset/function.json'))
    #for id in index:
    
    code_content=' '.join(js['input'].split())
    with open(f'{path}/before_transform_train_sets/{code_id}.c', 'w') as code:
        code.write(code_content)
    

def extract_MisclassifiedCode():
    index = []
    
    count = 0
    
    with open("../code/scores/misclassified.txt", "r") as fm:
        for line in fm:
            #print(line)
            line=line.strip()
            idx,label=line.split()
            if (float(label) <0.27):
                index.append(int(idx))
                count = count +1
    print(count)
    #saveCode(index)

    return index
"""
def generate_population(path1, ori):
    os.chdir(f'{path}/RM')
    os.system(f'cp {path1} .')
    filename = path1.split('/')[-1]
    for i in range(1,16):
        os.system(f"./mutation.sh {filename} {i}")
        after_transform = open(f'{path}/RM/Mutated.c').read()
        #print(after_transform)
        if ori == after_transform:
            #print(after_transform)
            continue
        if not os.path.exists(f"{path}/after_transform_train_sets/{filename.split('.')[0]}"):
            os.mkdir(f"{path}/after_transform_train_sets/{filename.split('.')[0]}")
        os.system(f"cp  {path}/RM/Mutated.c {path}/after_transform_train_sets/{filename.split('.')[0]}/{i}.c")
    print(i)
    os.system(f"rm {filename}")

def pack_to_the_jsonfile():
    os.chdir('/nas1-nfs1/home/rsr200002/dissector/ProgramTransformation')
    js = json.load(open('../dataset/function.json'))

    with open('transformed.jsonl', 'w') as output:
        for id_folder in Path(f'{path}/after_transform_train_sets/').iterdir():
            oid = int(str(id_folder).split('/')[-1])
           
            for transformed in id_folder.iterdir():
                opid = int(str(transformed).split('/')[-1].split('.')[0])
                
                struct = {'oid':oid,'opid':opid, 'target':js[int(oid)]['target'],'func':open(transformed).read()}
                
                json.dump(struct, output)
                output.write('\n')

"""
def generate_population(js,ori_code, code_id):
    os.chdir(f'{path}/RM')
    path1 = f'{path}/before_transform_train_sets/{code_id}.c'
    os.system(f'cp {path1} .')
    filename = path1.split('/')[-1]
    generated_Codes = []
    for i in range(1,16):
        os.system(f"./mutation.sh {filename} {i}")
        after_transform = open(f'{path}/RM/Mutated.c').read()
        #print(after_transform)
        if ori_code == after_transform:
            #print(after_transform)
            continue
        else :
            generated_Codes.append({'id': i, 'code': after_transform, 'fitness_Score': 0})

        
        #os.system(f"cp  {path}/RM/Mutated.c {path}/after_transform_train_sets/{filename.split('.')[0]}/{i}.c")
    #print("length of initial_population {}".format(len(generated_Codes)))
    os.system(f"rm {filename}")
    print("length of initial_population {}".format(len(generated_Codes)))
    return generated_Codes



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

def convert_code_into_features(c_id,code_sample,label, tokenizer, args):
    code= code_sample
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,c_id,label)

class GenerateDataset(Dataset):
    def __init__(self, initial_population, code_id, label, tokenizer, args):
        self.examples = []
        for items in initial_population:
            c_id = items['id']
            code_sample = items['code']
            label = label
            self.examples.append(convert_code_into_features(c_id,code_sample,label, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def compute_fitness_score(initial_population, code_id, label, tokenizer, args, model, train_loader, eval_loader):
    print("the labels are {}".format(label))
    dataset = GenerateDataset(initial_population,code_id,label,tokenizer,args)

    #eval_data_set = create_new_dataset(js, tokenizer, args, newcontents)
    genetics_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    genetics_dataloader = DataLoader(dataset, sampler = genetics_sampler, batch_size = 32)
    os.chdir(f'{code_path}')

    dissector = ImportanceScore(model=model, train_loader=train_loader, dev_loader=eval_loader, args=args)
    get_uncertainty_score = dissector._uncertainty_calculate(genetics_dataloader)
    get_uncertainty_score = list(get_uncertainty_score)

    for item, sc in zip(initial_population, get_uncertainty_score):
        item['fitness_Score'] = sc

    return initial_population

def get_identifiers(code):
    my_code = code
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='

    # Find variable names in the code
    variable_names = re.findall(pattern, my_code)
    return variable_names


def select_top_variants(initial_population):
    #no reverse on ascending order
    top_variants = sorted(initial_population, key=lambda x: x['fitness_Score'], reverse=True)[:math.ceil(len(initial_population)/2)]
    print("length of top_population {}".format(len(top_variants)))
    return top_variants

def do_crossover_and_mutation(population, rate):
    new_population = population[:]
    list_of_transformation = [11,12, 13,14,15]
    best_code = max(population, key=lambda x: x['fitness_Score'])
    
    best_variables = get_identifiers(best_code['code'])
    print("best_code", best_variables)
    length = math.ceil(len(best_variables)*rate)
    print("length", length)
    print("len of bes", len(best_variables))

    #replace_variables = []
    if length > len(best_variables):
        length = len(best_variables)

    #if len(best_variables) ==1:
        #replace_variables = random.choice(best_variables)
    #else:
        #replace_variables = random.choices(best_variables, k=length)
    

   
    
    #select the best population 
    # get its identifier names
    #select randomly len(identifier)*rate  
    os.chdir(f'{path}/RM')
    for item in population:
        for i in list_of_transformation: 
            code_content = item['code']
            
            identifier_of_code = get_identifiers(code_content)
            print("code identifier", identifier_of_code)
            #if len(identifier_of_code) ==1:
                #code_variables = random.choice(identifier_of_code)
            #else:
                #code_variables = random.choices(identifier_of_code, k=length)
            #code_variables = random.choices(identifier_of_code, k=length)
            #for x in range(len(replace_variables)):
                #code_content = code_content.replace(code_variables[x],replace_variables[x])
            #get code contents identifiers
            #select len(identifier)*rate identifiers and replace them use .replace method. 
            with open(c_filename, 'w') as f:
                f.write(code_content)
            os.system(f"./mutation.sh {c_filename} {i}")
            after_transform = open(f'{path}/RM/Mutated.c').read()
            with open(c_filename, 'w') as f:
                f.write(after_transform)
            os.system(f"./mutation.sh {c_filename} {1}")
            #after_transform = open(f'{path}/RM/Mutated.c').read()
        
            if code_content == after_transform:
            
                continue
            else :
                new_population.append({'id': i+item['id'], 'code': after_transform, 'fitness_Score': 0})
        
        os.system(f"rm {c_filename}")

            

    print("length of new_population {}".format(len(new_population)))

    return new_population


def geneticAlgorithm(tokenizer, args, model, index, train_loader, eval_loader ):
    filename = "test.jsonl"
    json_list = []
    max_iter = 3
    rate = 0.2
    
    #inital_population = []
    with open(filename) as f:
        for line in f:
            js = json.loads(line.strip())
            code_id = int(js['id'])
            label = js['label']
            itera = 0
            #print(label)
            #if (js['id'] in index):
            
            #if (js['id'] == 42):
            if (js['id'] in index):
                while itera < max_iter:
                    saveCode(js,code_id)
                    ori_code = read_format_code(code_id)
                    initial_population = generate_population(js,ori_code, code_id)
                    print("length of initial_population {}".format(len(initial_population)))
                    if (len(initial_population) <= 0):
                        line = js
                        json_list.append(json.loads(json.dumps(line)))
                        break
                    population = initial_population[:]
                    #while i<3:
                    population = compute_fitness_score(population, code_id, label, tokenizer, args, model, train_loader, eval_loader)
                    for item in population:
                        print("fitness scores are {}".format(item['fitness_Score']))

                    population = select_top_variants(population)
                    print("length of new_population {}".format(len(population)))

                    population = do_crossover_and_mutation(population, rate)
                
                    population = compute_fitness_score(population, code_id, label, tokenizer, args, model, train_loader, eval_loader)
                    best_variant = max(population, key=lambda x: x['fitness_Score'])
                    print(best_variant)
                    
                    if best_variant['fitness_Score'] > 0.27:
                        js['input'] = best_variant['code']
                        line = js
                        json_list.append(json.loads(json.dumps(line)))
                        break
                        

                    else:
                        line = js
                        json_list.append(json.loads(json.dumps(line)))
                        itera = itera+1


                #best_variant = max(population, key=lambda x: x['fitness_Score'])
                #print("------------------")
                #print(best_variant)

                

            else:
                line = js
                json_list.append(json.loads(json.dumps(line)))


    with open("newTransformed_tests.jsonl", 'w') as we:
        for obj in json_list:
            #print(obj)
            we.write(json.dumps(obj))
            we.write('\n')






def read_format_code(code_id):
    os.system(f"txl   -q -s 128  \"{path}/before_transform_train_sets/{code_id}.c\" \"{path}/Txl/RemoveCompoundStateSemicolon.Txl\" > aaa.c")
    os.system(f"txl   -q -s 128 aaa.c \"{path}/Txl/RemoveNullStatements.Txl\" > bbb.c")
    content = open('bbb.c').read()
    os.system(f'rm aaa.c && rm bbb.c')
    #print(content)
    return content



def convert_examples_to_featuresnew(js,tokenizer,args):
    #source
    code=' '.join(js['input'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['id'],js['label'])

class TextDatasetnew(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_featuresnew(js,tokenizer,args))
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def main():
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

    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    torch.cuda.set_device(5)
    args.device = torch.device("cuda", torch.cuda.set_device(5))



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None, output_hidden_states=True)
    config.num_labels=4
    #config.output_hidden_states = True
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

    #print(model)
    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    print(args.device)
    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))      
    model.to(args.device)
    model.eval()
    print(device)


    print(device)

    index = extract_MisclassifiedCode()

    


    
    train_dataset = TextDatasetnew(tokenizer, args, args.train_data_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,num_workers=4,pin_memory=True)

    eval_dataset = TextDatasetnew(tokenizer, args, args.eval_data_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)


    st_time = datetime.datetime.now()
    geneticAlgorithm(tokenizer, args, model, index, train_dataloader, eval_dataloader)
    ed_time = datetime.datetime.now()
    print(iter, 'cost time', ed_time - st_time)
    #print(index)       
    #transform(tokenizer, args, model, index, train_dataloader, dev_dataloader)

if __name__ == '__main__':
    main()


