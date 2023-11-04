import json 
import torch 
import copy 
from transformers import AutoTokenizer
from tqdm import tqdm

# Temproray dataset 
# path = '/home/radhaks/repos/jais_dir/data/shelf_Whisper_L_Temperature_0.7_500.json'
# with open(path,'r') as file:
#     dataset = json.load(file)
    
def build_prompt(datapoint, dataset, num_demonstrations = 1, num_candidates = 15):
    
    prompt = '''### Instruction: Your name is Jais. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You can answer in Arabic and English only. You are now being tasked as a transcript corrector. You will be provided with multiple Arabic transcripts of a sentence which may be semantically or grammatically wrong. Your task is to use the transcripts and produce the most coherent and semantically correct sentence of what the provided transcripts should mean. Complete the task below between [|Human|] and [|AI|], where  [|Human|] provides the input multiple transcripts and : [|AI|] generates the most likely sentence : \n\n'''
    
    ix = torch.randint(len(dataset), (num_demonstrations,)).tolist()
    # Loop for demonstrations
    for i in ix:
        candidates = ''
        
        for sentence in dataset[i]['candidates'][:10]: # Demonstrations have maximum of 10 candidates (to save on context length)
            candidates = candidates + '\n' + sentence
            
        prompt = prompt + f'### Input: [|Human|] {candidates}\n\n### Response: [|AI|]{dataset[i]["ground_truth"]} \n\n'
    
    # Loop of candidates in datapoint
    candidates = '\n'
    for sentence in datapoint['candidates'][:num_candidates]:
            candidates = candidates + '\n' + sentence

    prompt = prompt + f'### Input: [|Human|] {candidates}\n\n ### Response: [|AI|]'
    prompt_with_response = prompt + datapoint['ground_truth']+ '<|endoftext|>' # Did not include space so that this and the groudn truth tokens match in the first level
    return prompt, prompt_with_response

def get_batch(dataset, tokenizer ,train: True, no_of_datapoints = 1, no_of_demonstrations = 0, max_context_length = 2000):
    
    '''
    During train:True, we need the input ids with response and the masked version for training
    During train:False (Test): we only need the input id's without response. And we can also set the no.of demonstrations we need
    '''
    
    ix = torch.randint(len(dataset), (no_of_datapoints,)).tolist()

    ground_truths = [dataset[i]['ground_truth']+'<|endoftext|>' for i in ix]
    encoded_ground_truths =  [tokenizer(ground_truth, return_tensors="pt").input_ids for ground_truth in ground_truths]
    
    if not train:
        if no_of_demonstrations == 0 : print('You can also set the number of demonstration you would like to pass')
        prompts = [build_prompt(dataset[i], dataset,num_demonstrations=0)[0] for i in ix]
        
        if not tokenizer.padding_side == 'left': print("Please set tokenizer padding side to left for inference \n Exiting"); return
        input_ids = tokenizer(prompts, return_tensors="pt", padding = True).input_ids
        return input_ids
    
    # This is for during training
    prompts_with_respones = [build_prompt(dataset[i], dataset,num_demonstrations=0)[1] for i in ix]
    input_ids_with_response = [tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts_with_respones]
    
    #Replacing sequences with lesser number of candidates if greater than specified length
    x = 0 # To adjust index as we remove elements 
    for index in range(len(input_ids_with_response)): 
        _num_candidates = 15
        index = index -x # adjusting index as we removed elements 
         
        while input_ids_with_response[index].shape[-1] > max_context_length:
            _num_candidates = _num_candidates - 4
            
            if _num_candidates < 1: # If context is too long even with one candidate -> skip the datapoint
                print(f'Skipping datapoint {ix[index]}')
                input_ids_with_response.pop(index)
                ix.pop(index)
                x+=1
                break 
                
            prompt_with_reduced_candidates = build_prompt(dataset[ix[index]], dataset,num_demonstrations=0, num_candidates= _num_candidates )[1] 
            input_ids_with_response[index] = tokenizer(prompt_with_reduced_candidates, return_tensors="pt").input_ids
    
    if len(input_ids_with_response) == 0: # To eleminate the edge case where we removed all elements 
        return torch.full((no_of_datapoints,no_of_datapoints), -1), torch.full((no_of_datapoints,no_of_datapoints), -1)

    masked_input_ids_with_response = copy.deepcopy(input_ids_with_response)
    for i in range(len(masked_input_ids_with_response)):
        masked_input_ids_with_response[i][:,:-encoded_ground_truths[i].shape[-1]] = -1
        

    
    max_len = max(s.shape[-1] for s in input_ids_with_response)
    
    def pad_right(x, pad_id):
    # pad right based on the longest sequence
        n = max_len - x.shape[-1]
        return torch.cat(  ( x, torch.full((1,n), pad_id, dtype=x.dtype)), 1 )
    
    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids_with_response]).squeeze(1)
    y = torch.stack([pad_right(x, pad_id=-1) for x in masked_input_ids_with_response]).squeeze(1)
    
    return x, y


def count_no_of_datapoints_within_length(dataset, length):
    '''
    Block to count how many datapoints are skipped due to context length shortage
    '''
    y= []; n=[]
    
    for i in tqdm(range(len(dataset))):
        prompt = build_prompt(dataset[i], dataset,num_demonstrations=0)[1] 
        l = tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
        if l <length:  y.append(i)
        else : n.append(i)
        
    print(f'Num of datapoints within range {len(y)} and {len(n)} are outside')
        
        
# To test loop 
# tokenizer = AutoTokenizer.from_pretrained( '/home/radhaks/repos/jais_dir/lit_jais',padding_side='left')

# count_no_of_datapoints_within_length(dataset, 200)

# for i in tqdm(range(len(dataset))): 
#     x = get_batch(dataset, tokenizer ,no_of_datapoints = 4, max_context_length= 400, train = True)
    