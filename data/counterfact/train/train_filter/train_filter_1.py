"""
step1.
对于ins和oos data,过滤出nli_score和contain_score不及格的gpt4_response_after_edit
"""
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def save_json(save_file_path,data):
    with open(save_file_path,'w',encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False)

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

def nli_binary_select(tokenizer,model,data_point,response_before_edit_column,device):
    # nli judgement
    premise=data_point['requested_rewrite']+' '+data_point[response_before_edit_column]
    hypothesis_new=data_point['requested_rewrite']+' '+data_point['target_new']
    if data_point['query_type'] in ['naive','paraphrase']:
        hypothesis_old=data_point['requested_rewrite']+' '+data_point['target_old']
    else:
        hypothesis_old=data_point['requested_rewrite']+' '+data_point['target_old']
        # hypothesis_old=data_point['requested_rewrite']+' '+data_point['query_target']

    result_old_probability,result_old=nli_inference(tokenizer,model,premise,hypothesis_old,device)
    result_new_probability,result_new=nli_inference(tokenizer,model,premise,hypothesis_new,device)
    # 只有response蕴含old_target并且response和new_target矛盾/中立才会被选择
    if data_point['query_type'] in ['naive','paraphrase']:
        return True if result_old==2 and result_new==0 else False
    else:
        return True if result_old==0 and result_new!=0 else False

def contain_binary_select(data_point,response_before_edit_column):
    # contain/hit judgement
    contain_new=True if data_point['target_new'].lower().strip() in data_point[response_before_edit_column].lower() else False
    if data_point['query_type'] in ['naive','paraphrase']:
        contain_old=True if data_point['target_old'].lower().strip() in data_point[response_before_edit_column].lower() else False
    else:
        contain_old=True if data_point['target_old'].lower().strip() in data_point[response_before_edit_column].lower() else False
        # contain_old=True if data_point['query_target'].lower().strip() in data_point[response_before_edit_column].lower() else False
    return True if (not contain_old and contain_new) else False


def main(raw_data_path,nli_model_path,response_before_edit_column,save_filter_data_path,device):
    # 读取数据
    raw_data=read_json(read_file_path=raw_data_path)
    # 加载model
    tokenizer,model=load_model(model_path=nli_model_path,device=device)
    select_results=[]
    # judge
    for i in tqdm(range(len(raw_data))):
        data_point=raw_data[i]
        contain_select=contain_binary_select(data_point,response_before_edit_column) # contain_select判断
        if contain_select:
            nli_select=nli_binary_select(tokenizer,model,data_point,response_before_edit_column,device) # nli_select判断
            if nli_select:
                select_results.append(True)
            else:
                select_results.append(False)
        else:
            select_results.append(False)

    # select
    select_data=[raw_data[i] for i in range(len(raw_data)) if select_results[i]]

    # result
    edit_scores={'naive_success':0,'naive_number':0,
                'paraphrase_success':0,'paraphrase_number':0,
                'neighbor_success':0,'neighbor_number':0,}
    for i in range(len(raw_data)):
        data_point=raw_data[i]
        if data_point['query_type']=='naive':
            edit_scores['naive_number']+=1
            if select_results[i]:
                edit_scores['naive_success']+=1
        elif data_point['query_type']=='paraphrase':
            edit_scores['paraphrase_number']+=1
            if select_results[i]:
                edit_scores['paraphrase_success']+=1
        elif data_point['query_type']=='neighbor':
            edit_scores['neighbor_number']+=1
            if select_results[i]:
                edit_scores['neighbor_success']+=1

    print(edit_scores)
    save_json(save_file_path=save_filter_data_path,data=select_data)


if __name__=='__main__':
    # setting
    # dataset='counterfact'
    raw_data_path="data/zsre/full_train/raw_naive_id_15000-30000-0-10000_filter-chatgpt_0-2000.json"
    nli_model_path="/newdisk/sxs/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    response_before_edit_column='chatgpt_response_after_edit'
    save_filter_data_path="data/zsre/full_train/raw_naive_id_15000-30000-0-10000_filter-chatgpt_0-2000_filter_1.json"
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    # run
    main(raw_data_path,nli_model_path,response_before_edit_column,save_filter_data_path,device)

    # dataset='counterfact'
    # raw_data_path="data/counterfact/train/chatgpt-augment/train_paraphrase_10000-chatgpt_0-10000.json"
    # nli_model_path="/newdisk/sxs/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # response_before_edit_column='chatgpt_response_after_edit'
    # save_filter_data_path="data/counterfact/train/chatgpt-augment/train_paraphrase_10000_filter_1.json"
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # # run
    # main(raw_data_path,nli_model_path,response_before_edit_column,save_filter_data_path,device)

