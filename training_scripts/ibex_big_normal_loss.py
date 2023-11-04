import os
import sys
import time
import wandb

import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import json
from collections import OrderedDict

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve() # does not work as jupyter notebook     
sys.path.append(str(wd))

import whisper_openAI.whisper as whisper
from lit_jais_big.utils import get_batch, build_prompt, adapter_state_from_state_dict
from lit_jais_big.modeling_jais import JAISLMHeadModel
from transformers import AutoTokenizer #, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig



#cli setup 
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3,help='learning rate for the model (default: 1e-3)')
parser.add_argument('--d', type=int, default=1,help='No of GPUs (default: 1)')
parser.add_argument('--data_path', type=str,help='Path to data') 
parser.add_argument('--note', type=str,help='In case you want to add someting to run name',default='') 

args = parser.parse_args()
learning_rate = args.lr
data_path = 'data/shelf_Whisper_L_Temperature_0.9_1500.json'# args.data_path

# Hyperparameters
num_epochs = 10

# Batch and device configuration
devices = args.d
batch_size = 32 / devices 
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size


with open(data_path,'r') as file:
    dataset = json.load(file)
train_data = dataset[:1000]
val_data   = dataset[1000:]

train_data_len = len(train_data)
val_data_len = len(val_data)

# Had to make tokenzier global to be shared b/w get batch and main 
tokenizer = AutoTokenizer.from_pretrained( 'lit_jais', padding = False) # The dataloader/get_batch handels the padding requirements

print('loaded data')

epoch_size = train_data_len // micro_batch_size // devices
max_iters = num_epochs * epoch_size 
eval_iters = val_data_len // micro_batch_size  // devices 
warmup_steps = epoch_size * 0 // devices 

# Context configuration 
max_seq_length = 2000

# Checkpointing configuration

save_interval = epoch_size # save every epoch
log_interval = 1
run_name = f'{args.note} BIG_normal_loss_{learning_rate}_{data_path.split(".")[0]}'
out_dir: str = 'runs/'+run_name

# wandb configuration
wandb.login()
wandb.init(
    project="Jais_inference",
    name=run_name,
    #group=run_name
    config={
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "batch_size": (batch_size*devices),
    "micro_batch_size":micro_batch_size,
    "dataset":'cv_small',
    "note": 'Adapting the training pipeline to ibex',
  #  'devices':devices
    }
)

# Use if needed, we use DDP strategy with the current implementation on 2x A100s
ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}

def main():
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
        os.makedirs(out_dir, exist_ok=True)
        
        config = AutoConfig.from_pretrained(
                        os.path.join(os. getcwd(),'lit_jais'),
                        trust_remote_code=True) # The attibutes in cofig dict have conditions for trust_remote_Code to be true
                    
    cache_dir = '/data/jaise_weights/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d/'
    if not os.path.isdir(cache_dir):
        cache_dir = '/home/radhaks/repos/Whispering-LLaMA/jaise_weights'
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
                    
    with fabric.init_module():
         # strict=False because missing keys due to adapter weights not containted in state dict  
        model.load_state_dict(checkpoint, strict=False)
    print('loaded Jias model')

    for n,p in model.named_parameters():
        if 'adapter' in n :
            p.requires_grad = True
        else : 
            p.requires_grad = False

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params/1e6}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 0)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, out_dir, tokenizer)
    wandb.finish()

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-adapter-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
    tokenizer,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0 # gets updated each time you compleate a batch aka each time you take a step

    for iter_num in tqdm(range(max_iters)):

        t0 = time.time()

        x, y = get_batch(train_data, tokenizer ,train=True, no_of_datapoints = micro_batch_size, no_of_demonstrations = 0, max_context_length = max_seq_length)
        x , y = fabric.to_device((x.pin_memory(), y.pin_memory()))
        logits = model(x)[0]
        loss = loss_fn(logits, y)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)): # Skip gradient synchronization during backward to avoid redundant communication overhead (Sync after gradient accumaltion is done)
            fabric.backward(loss / gradient_accumulation_steps)

        if (iter_num + 1) % gradient_accumulation_steps == 0: # Update model after  step
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            
            # learning rate scheduler
            lr = learning_rate - ((learning_rate - 1e-5)/max_iters)*(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            wandb.log({"lr": lr})

        dt = time.time() - t0
        
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt:.2f}s")
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
            wandb.log({"train_iter": iter_num, "train_Iter_loss": loss.item()})
            
       # Saving Adapter weights at the end of epoch
        if (iter_num + 1) % epoch_size == 0:
            print(f"Saving adapter weights to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{int((iter_num+1)/epoch_size):06d}.pth"))

        # Print and Log val loss 
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.barrier()
            wandb.log({"val_step": iter_num, "val_step_loss": val_loss})
            print('End of epoch ',(iter_num+1)/epoch_size)





@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")

    if len(val_data) == val_data_len : # To adjust eval_iters for val dataset
        eval_iters =  val_data_len // micro_batch_size  // devices
    else :# To adjust eval_iters for train dataset
        eval_iters =  epoch_size // devices

    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x , y = get_batch(val_data, tokenizer ,train=True, no_of_datapoints = micro_batch_size, no_of_demonstrations = 0, max_context_length = max_seq_length)
        x , y = fabric.to_device((x.pin_memory(), y.pin_memory()))
        logits = model(x)[0]
        loss = loss_fn(logits, y)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    # output = generate_response(model, instruction)
    # fabric.print(instruction)
    # fabric.print(output)

    model.train()
    return val_loss.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits.contiguous()
    targets = targets.contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def save_model_checkpoint(fabric, model, file_path):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = file_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:

            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict)
            torch.save(state_dict, file_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict())
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    main()
    