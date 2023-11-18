import os
import sys
import time
import json
import wandb
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pyarabic.araby as araby
from collections import OrderedDict

from evaluate import load
wer = load("wer")
cer = load("cer")

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve() # does not work as jupyter notebook 
sys.path.append(str(wd))

#import whisper_openAI.whisper as whisper
from lit_jais_big_init.utils import get_batch, adapter_state_from_state_dict, build_prompt
from lit_jais_big_init.modeling_jais import JAISLMHeadModel
from transformers import AutoTokenizer #, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help= 'Path to data json file') 
parser.add_argument('--adapter_path', type=str, help= 'Path to adapter checkpoint') 
parser.add_argument('--batch', type=int, default=1, help= 'Number of datapoints to use for batch inference')
args = parser.parse_args()

batch = args.batch
devices = 1
tokenizer = AutoTokenizer.from_pretrained( 'lit_jais', padding_side='left') 

torch.set_float32_matmul_precision("high") # For A100 GPUs

path_to_inferences = '/ibex/user/radhaks/LLMs/Jais_GitHub/inferences'
path_to_runs = '/ibex/user/radhaks/LLMs/Jais_GitHub/runs'
path_to_data = '/ibex/user/radhaks/LLMs/Jais_GitHub/data'

adapter_path = os.path.join(path_to_runs,args.adapter_path )
data_path = os.path.join(path_to_data, args.data_path)
path_to_inferences = os.path.join(path_to_inferences,args.adapter_path)

if not os.path.isdir(path_to_inferences):
    os.makedirs( path_to_inferences, exist_ok=True)

# Temproray dataset 

with open(data_path,'r') as file:
    dataset = json.load(file)
dataset = dataset[0:100]

wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="Jais_inference", 
    name= f'Clean_big_init_Greedy_{args.adapter_path}',
        config={
    "dataset_len": len(dataset),
    'note':'Made the mistake of running previous run with 1 demo',
    }
)


fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy= "ddp"  if devices > 1 else "auto" , 
    # If the model is not fittng replace DDP with DeepSpeed strategy (DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto")
        precision="bf16-true",
    )
fabric.launch()
fabric.seed_everything(1337 + fabric.global_rank)

if fabric.global_rank == 0:
    
    config = AutoConfig.from_pretrained(
                    os.path.join(os. getcwd(),'lit_jais'),
                    trust_remote_code=True) # The attibutes in cofig dict have conditions for trust_remote_Code to be true
                
cache_dir = '/data/jaise_weights/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d/'
if not os.path.isdir(cache_dir):
    cache_dir = '/home/radhaks/repos/Whispering-LLaMA/jaise_weights/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d/'
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Can't find the pretrained weights at {cache_dir}.")
    
with fabric.init_module():
    model = JAISLMHeadModel(config) # instead of model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,cache_dir= '/data/jaise_weights')
    
saves = []
shards = os.listdir(cache_dir)
shards.sort()
for name in shards:
    if 'bin' in name and 'json' not in name:
        saves.append(os.path.join(cache_dir,name))
    
print('Loading Jais checkpoints')
checkpoint = OrderedDict() 
for i in tqdm(saves):
    this = torch.load(i, map_location = torch.device('cpu'))
    checkpoint = checkpoint | this
    
#adapter_path = '/home/radhaks/repos/Jais_GitHub/runs/from_ibex/ altered_loss_0.001_data/shelf_Whisper_L_Temperature_0/iter-000006.pth'
                
with fabric.init_module():
    # strict=True to check adapter weight compatabiy 
    model.load_state_dict(checkpoint, strict=False)
print('loaded Jias model')

for n,p in model.named_parameters():
        p.requires_grad = False
        

def get_response(text,tokenizer=tokenizer,model=model):
    input_ids = tokenizer(text, return_tensors="pt",padding = True).input_ids
    inputs = input_ids.to(model.device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        # top_p=0.9,
        # temperature=0.3,
        max_length=2048-input_len,
       min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=False,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return response


prompts = [build_prompt(i, dataset, num_demonstrations = 0, num_candidates = 15)[0] for i in tqdm(dataset)]
ground_truths = [i['ground_truth'] for i in tqdm(dataset)]
oracle = []


# Getting oracle
for datapoint in tqdm(dataset):
    ground_truth = datapoint['ground_truth']
    max_wer = 100
    best_candidate = ''

    for i in datapoint['candidates'][:15]:
        candidate_wer =  wer.compute(predictions=[i], references=[ground_truth])
        if  candidate_wer < max_wer:
            best_candidate = i
            max_wer = candidate_wer

    oracle.append(best_candidate)
    
# Clearing the oracle 
for i in range(len(oracle)):
    oracle[i] = ' '.join(oracle[i].replace('،','').replace('.','').replace('؟','').replace(',','').strip().split(' '))
    oracle[i] = araby.strip_diacritics(oracle[i])
    
# Cleaning the groud truths 
for i in range(len(ground_truths)):
    ground_truths[i] = ' '.join(ground_truths[i].replace('،','').replace('.','').replace('؟','').replace(',','').strip().split(' '))
    ground_truths[i] = araby.strip_diacritics(ground_truths[i])



checkpoints = os.listdir(adapter_path)
checkpoints.sort()

for n in checkpoints:
    responses = []
    to_json = []
    
    # Loading adapter
    adapter_checkpoint = torch.load(os.path.join(adapter_path,n), map_location = torch.device('cpu'))
    with fabric.init_module():
        # strict=True to check adapter weight compatabiy 
        model.load_state_dict(adapter_checkpoint, strict=False)
        print('Added adapter checkpoint',n)

    # Running Inference
    print('Running Inference')
    for i in tqdm(range(0, len(prompts), batch)):
        responses.extend(get_response(prompts[i:i+batch]))

    # Checking
    if len(responses) != len(ground_truths):
        raise RuntimeError('Responses and groundtruths have difference number of elements')

    # Cleaning responses
    for i in range(len(responses)):
        responses[i] = responses[i][len(prompts[i]):]
        responses[i] = ' '.join(responses[i].replace('،','').replace('.','').replace('؟','').replace(',','').strip().split(' '))
        responses[i] = araby.strip_diacritics(responses[i])
        #print(i,responses[i])
        
    # Checking 
    if len(responses) != len(oracle):
        raise RuntimeError('Responses and oracle have difference number of elements')


    word_error_rate = round( wer.compute(predictions=responses, references=ground_truths), 2)
    char_error_rate = round( cer.compute(predictions=responses, references=ground_truths), 2)
    print('for {n}')
    print(f"Prediction WER: {word_error_rate} CER: {char_error_rate}")

    oracle_wer = round( wer.compute(predictions= oracle, references=ground_truths), 2)
    oracle_cer = round(cer.compute(predictions= oracle, references=ground_truths), 2)
    print(f"Oracle WER: {oracle_wer} CER: {oracle_cer}")

    wandb.log({
        "word_error_rate": word_error_rate,
        "char_error_rate": char_error_rate,
        "oracle_wer":oracle_wer,
        "oracle_cer":oracle_cer
        })
    
    # Saving as inferences in Json file
    for o,p,g,d in zip(oracle,responses,ground_truths,dataset):
        entry = {
            'oracle': o,
            'prediction':p,
            'ground_truth':g,
        'candidates':d['candidates']}
        to_json.append(entry)
        
    # Saving
    save_file = os.path.join(path_to_inferences, f'{n}.json') 
    with open(save_file,'w') as file:
        json.dump(to_json,file,indent=4)


# If you want to save the outputs
# to_json = []
# for o,p,g,d in zip(oracle,responses,ground_truths,dataset):
#     entry = {
#         'oracle': o,
#         'prediction':p,
#         'ground_truth':g,
#     'candidates':d['candidates']}
#     to_json.append(entry)
# with open(os.path.join('Big_jais_cv11.json'),'w') as file:
#     json.dump(to_json,file,indent=4)

    # from evaluate import load
    # wer = load("wer")
    # cer = load("cer")
    # ground_truths = [i['ground_truth'].replace(',','').replace('.','').strip() for i in tqdm(dataset)]
    # word_error_rate = round( wer.compute(predictions=responses, references=ground_truths), 2)