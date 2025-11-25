"""
Semantic Edit Evaluation using NLI (Natural Language Inference)

NLI three-way classification: entailment, neutral, contradiction

For in-scope (ins) queries:
- If pred_response_after_edit -> entails -> new_target & contradicts -> old_target: 1 point
- If pred_response_after_edit -> entails -> new_target & entails/neutral -> old_target: 0.5 points
- If pred_response_after_edit -> neutral/contradicts -> new_target: 0 points

For out-of-scope (oos) queries:
- If pred_response_after_edit -> entails -> old_target & contradicts -> new_target: 1 point
- If pred_response_after_edit -> entails -> old_target & entails/neutral -> new_target: 0.5 points
- If pred_response_after_edit -> neutral/contradicts -> old_target: 0 points

Note: Currently using CPU inference, which may be slow.
"""
import torch
import json
from tqdm import tqdm
from scipy.stats import hmean
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path,device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return tokenizer,model

def nli_inference(tokenizer,model,premise,hypothesis,device):
    max_seq_length=512
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                    max_length=max_seq_length,
                                                    return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contra1diction"
    # },
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
    predict=predicted_probability.index(max(predicted_probability))
    return predicted_probability,predict

def nli_edit_score(dataset,tokenizer,model,data_point,eval_column,device):
    assert dataset in['zsre','counterfact']
    # get premise
    if dataset=='zsre':
        # premise=data_point[eval_column]
        premise=data_point['query']+' '+data_point[eval_column]
    elif dataset=='counterfact':
        premise=data_point['query']+' '+data_point[eval_column]
    # get hypothesis_old
    if data_point['query_type'] in ['naive','paraphrase']:
        hypothesis_old=data_point['requested_rewrite']+' '+data_point['target_old']
    else:
        if dataset == 'zsre':
            hypothesis_old = data_point['query'] + ' ' + data_point['query_target']
        elif dataset == 'counterfact':
            hypothesis_old = data_point['query'] + ' ' + data_point['target_old']     
    hypothesis_new = data_point['requested_rewrite'] + ' ' + data_point['target_new']
    result_old_probability, result_old = nli_inference(tokenizer, model, premise, hypothesis_old, device)
    result_new_probability, result_new = nli_inference(tokenizer, model, premise, hypothesis_new, device)
    if data_point['query_type'] in ['naive', 'paraphrase']: 
        # Only if response entails new_target and contradicts old_target is the edit fully successful
        if result_new == 0 and result_old != 0:
            return 1
        elif result_new == 0 and result_old == 0:
            return 0.5
        else:
            return 0

    elif data_point['query_type'] == 'neighbor':
        # Only if response entails old_target and does not entail new_target is locality successful
        if result_old == 0 and result_new != 0:
            return 1
        elif result_old == 0 and result_new == 0:
            return 0.5
        else:
            return 0

def load_json_data(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def save_json_data(save_file_path,data):
    with open(save_file_path,'w',encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False)

def semantic_edit_eval(dataset, eval_data_path, eval_column, device):
    eval_data = load_json_data(read_file_path=eval_data_path)
    # Load model
    nli_model_path = "/newdisk/sxs/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer, model = load_model(model_path=nli_model_path, device=device)
    # Evaluate
    edit_score_dict={'naive_success':0,'naive_number':0,
                    'paraphrase_success':0,'paraphrase_number':0,
                    'neighbor_success':0,'neighbor_number':0,}
    query_types=[]
    edit_scores=[]
    for i in tqdm(range(len(eval_data))):
        data_point=eval_data[i]
        if data_point['query_type']=='naive':
            edit_score_dict['naive_number']+=1
            edit_score=nli_edit_score(dataset,tokenizer,model,data_point,eval_column,device)
            edit_score_dict['naive_success']+=edit_score
        elif data_point['query_type']=='paraphrase':
            edit_score_dict['paraphrase_number']+=1
            edit_score=nli_edit_score(dataset,tokenizer,model,data_point,eval_column,device)
            edit_score_dict['paraphrase_success']+=edit_score
        elif data_point['query_type']=='neighbor':
            edit_score_dict['neighbor_number']+=1
            edit_score=nli_edit_score(dataset,tokenizer,model,data_point,eval_column,device)
            edit_score_dict['neighbor_success']+=edit_score
        query_types.append(data_point['query_type'])
        edit_scores.append(edit_score)
    # print(edit_score_dict)
    # print(edit_scores)
    naive_score=round(edit_score_dict['naive_success']/edit_score_dict['naive_number']*100,2)
    paraphrase_score=round(edit_score_dict['paraphrase_success']/edit_score_dict['paraphrase_number']*100,2)
    neighbor_score=round(edit_score_dict['neighbor_success']/edit_score_dict['neighbor_number']*100,2)
    all_score=round((edit_score_dict['naive_success']+edit_score_dict['paraphrase_success']+edit_score_dict['neighbor_success'])/(edit_score_dict['naive_number']+edit_score_dict['paraphrase_number']+edit_score_dict['neighbor_number'])*100,2)
    hm_score=round(hmean([naive_score,paraphrase_score,neighbor_score]),2)
    return({'naive':naive_score,'paraphrase':paraphrase_score,'neighbor':neighbor_score,'all':all_score,'hm':hm_score})