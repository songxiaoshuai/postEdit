"""
Lexical Retention Evaluation

Uses ROUGE score to evaluate style preservation before and after editing.
Note: When calculating ROUGE, in-scope data should also mask the target.
"""
import json
from rouge import Rouge 
from copy import deepcopy
from scipy.stats import hmean

def read_json(read_file_path):
    with open(read_file_path,encoding='UTF-8') as file:
        data=json.load(file)
    return data

def calculate_rouge(hypothesis, reference):
    rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference,avg=True)
    return scores

def lexical_retention_eval(eval_data_path, target_attribute, pred_attribute):
    # Read data
    data = read_json(eval_data_path)
    assert target_attribute in data[0].keys()
    assert pred_attribute in data[0].keys()
    # Mask target for in-scope data
    new_data = []
    for data_point in data:
        if data_point['query_type'] in ['naive', 'paraphrase']:
            data_point[pred_attribute] = data_point[pred_attribute].lower().replace(data_point['target_new'].lower(), 'mask')
            data_point[target_attribute] = data_point[target_attribute].lower().replace(data_point['target_old'].lower(), 'mask')
        new_data.append(data_point)
    data = deepcopy(new_data)
    # Evaluate each part
    rouge1_scores = {}
    for type in ['naive', 'paraphrase', 'neighbor']:
        data_filter = [data_point for data_point in data if data_point['query_type'] == type]
        if len(data_filter) == 0:
            continue
        reference = [data_point[target_attribute] for data_point in data_filter]
        hypothesis = [data_point[pred_attribute] for data_point in data_filter]
        pred_rouge_scores = calculate_rouge(hypothesis=hypothesis, reference=reference)
        rouge1_scores[type] = round(pred_rouge_scores["rouge-1"]['f'] * 100, 2)
    # Evaluate all
    reference = [data_point[target_attribute] for data_point in data]
    hypothesis = [data_point[pred_attribute] for data_point in data]
    pred_rouge_scores = calculate_rouge(hypothesis=hypothesis, reference=reference)
    rouge1_scores['all'] = round(pred_rouge_scores["rouge-1"]['f'] * 100, 2)
    # Harmonic mean
    rouge1_scores['hm'] = round(hmean([rouge1_scores['naive'], rouge1_scores['paraphrase'], rouge1_scores['neighbor']]), 2)
    return rouge1_scores