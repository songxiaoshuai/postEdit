"""
Lexical Edit Evaluation

Scoring rules:
If response_before_edit contains old_target:
    For INS (in-scope):
        - If response_after_edit contains target_new & does not contain target_old: 1 point
        - If response_after_edit contains target_new & contains target_old: 0.5 points
        - If response_after_edit does not contain target_new: 0 points
    For OOS (out-of-scope):
        - If response_after_edit contains target_old & does not contain target_new: 1 point
        - If response_after_edit contains target_old & contains target_new: 0.5 points
        - If response_after_edit does not contain target_old: 0 points
"""
import json
from scipy.stats import hmean

def load_json_data(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def save_json_data(save_file_path,data):
    with open(save_file_path,'w',encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False)

def contain_edit_score(data_point, eval_column):
    contain_new = True if data_point['target_new'].lower().strip() in data_point[eval_column].lower() else False
    if data_point['query_type'] in ['naive', 'paraphrase']:
        contain_old = True if data_point['target_old'].lower().strip() in data_point[eval_column].lower() else False
    else:
        if 'query_target' in data_point:
            contain_old = True if data_point['query_target'].lower().strip() in data_point[eval_column].lower() else False
        else:
            contain_old = True if data_point['target_old'].lower().strip() in data_point[eval_column].lower() else False
    if data_point['query_type'] in ['naive', 'paraphrase']: 
        if contain_new and not contain_old:
            return 1
        elif contain_new and contain_old:
            return 0.5
        else:
            return 0
    elif data_point['query_type'] == 'neighbor':
        if contain_old and not contain_new:
            return 1
        elif contain_old and contain_new:
            return 0.5
        else:
            return 0  
    else:
        raise ValueError(f"Unknown query_type: {data_point['query_type']}")

def lexical_edit_eval(eval_data_path, eval_column):
    eval_data = load_json_data(eval_data_path)
    edit_score_dict = {'naive_success': 0, 'naive_number': 0,
                       'paraphrase_success': 0, 'paraphrase_number': 0,
                       'neighbor_success': 0, 'neighbor_number': 0,}
    query_types = []
    edit_scores = []
    for data_point in eval_data:
        if data_point['query_type'] == 'naive':
            if data_point['target_old'].lower() in data_point['response_before_edit'].lower():
                edit_score_dict['naive_number'] += 1
                edit_score = contain_edit_score(data_point, eval_column)
                edit_score_dict['naive_success'] += edit_score
        elif data_point['query_type'] == 'paraphrase':
            if data_point['target_old'].lower() in data_point['response_before_edit'].lower():
                edit_score_dict['paraphrase_number'] += 1
                edit_score = contain_edit_score(data_point, eval_column)
                edit_score_dict['paraphrase_success'] += edit_score
        elif data_point['query_type'] == 'neighbor':
            edit_score_dict['neighbor_number'] += 1
            edit_score = contain_edit_score(data_point, eval_column)
            edit_score_dict['neighbor_success'] += edit_score
        query_types.append(data_point['query_type'])
        edit_scores.append(edit_score)

    naive_score = round(edit_score_dict['naive_success'] / edit_score_dict['naive_number'] * 100, 2)
    paraphrase_score = round(edit_score_dict['paraphrase_success'] / edit_score_dict['paraphrase_number'] * 100, 2)
    neighbor_score = round(edit_score_dict['neighbor_success'] / edit_score_dict['neighbor_number'] * 100, 2)
    all_score = round((edit_score_dict['naive_success'] + edit_score_dict['paraphrase_success'] + edit_score_dict['neighbor_success']) / (edit_score_dict['naive_number'] + edit_score_dict['paraphrase_number'] + edit_score_dict['neighbor_number']) * 100, 2)
    hm_score = round(hmean([naive_score, paraphrase_score, neighbor_score]), 2)
    return {'naive': naive_score, 'paraphrase': paraphrase_score, 'neighbor': neighbor_score, 'all': all_score, 'hm': hm_score}
    
if __name__=='__main__':
    # setting
    eval_data_path=f'data/counterfact/fullTest/query_data-gpt4_0-1000.json'
    eval_column='gpt4_response_after_edit'
    # run
    print('lexical_edit (hit):',lexical_edit_eval(eval_data_path,eval_column))
