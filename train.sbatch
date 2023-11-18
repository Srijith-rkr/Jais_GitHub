#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J train
#SBATCH -o file.%J.out
#SBATCH -e file.%J.err
#SBATCH --mail-user=srijith.radhakrishnan@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=13:00:00
#SBATCH --mem=128G
# Memory used to be 265 but reduces to check, also reduced cpu from 16 to 12 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
# ddp did not work with 12 cpus - was stuck at model chekpoing loading stage 
# # was 16 cps
#    #SBATCH --constraint=[v100]
#SBATCH --reservation=A100


source /home/radhaks/miniconda3/etc/profile.d/conda.sh

conda activate LLFT
# wandb login

#run the application
# python generate_json_finetune.py  --temp .90 
# python generate_json_shelf.py  --temp .70
#python training_scripts/ibex.py --lr 1e-4 --data_path '/ibex/user/radhaks/LLMs/Jais_GitHub/data/Finetuned_Whisper_L_Temperature_0.9_1500.json'
#python training_scripts/ibex_altered_loss.py --lr 1e-4  --data_path '/ibex/user/radhaks/LLMs/Jais_GitHub/data/Finetuned_Whisper_L_Temperature_0.9_1500.json'
# python training_scripts/ibex_big_normal_loss.py --lr 1e-4

#python lit_jais/inference.py  --adapter_path '/ibex/user/radhaks/LLMs/Jais_GitHub/runs/llm_debugging_0.001_data/shelf_Whisper_L_Temperature_0.9_1500.json' --data_path  'data/shelf_Whisper_L_Temperature_0.9_1500.json' --batch 1
#python lit_jais/inference.py  --adapter_path /ibex/user/radhaks/LLMs/Jais_GitHub/runs/llm_debugging_0.001_/ibex/user/radhaks/LLMs/Jais_GitHub/data/Finetuned_Whisper_L_Temperature_0.9_1500.json --data_path '/ibex/user/radhaks/LLMs/Jais_GitHub/data/Finetuned_Whisper_L_Temperature_0.9_1500.json' --batch 1

# Runs with v100

#python training_scripts/ibex_for_ddp.py --lr 1e-4 --data_path 'temp_cv11' --d 2
#python training_scripts/big_jias_ibex_for_dpp.py --lr 1e-3 --data_path 'fleurs' --d 2 --mask 0 

#python lit_jais/inference.py  --adapter_path runs/Val_lossddp_check_0.001_fleurs_1device --data_path data/fleurs_test.json
#python lit_jais_big/inference.py --adapter_path runs/Big_Jais_DDP_0.001_fleurs_1device --data_path data/fleurs_test.json


python training_scripts/jais_big_init.py --lr 1e-3 --data_path 'temp_cv11' --d 2 --mask 1 --batch 16
#python lit_jais_big_init/inference_beam.py --adapter_path 'NOMASKjais_big_init_ar_prompt0.001_temp_cv11_2device' --data_path temp_cv11_test.json