"""
Semantic Retention Evaluation using SBert

For in-scope (ins):
- Calculate semantic similarity between edited_response (masked target_new) 
  and original_response (masked target_old)
- Higher is better

For out-of-scope (oos):
- Calculate overall semantic similarity directly
"""
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings,cos_sim
from scipy.stats import hmean

def read_json(read_file_path):
    with open(read_file_path,encoding='UTF-8') as file:
        data=json.load(file)
    return data

def calc_sim_score(emb_model, data_point, target_attribute, pred_attribute):
    if data_point['query_type'] == 'neighbor':
        pred = data_point[pred_attribute].lower()
        target = data_point[target_attribute].lower()
    else:
        pred = data_point[pred_attribute].lower().replace(data_point['target_new'].lower(), 'mask')
        target = data_point[target_attribute].lower().replace(data_point['target_old'].lower(), 'mask')
    pred_embedding = emb_model.encode(pred, convert_to_tensor=True)
    pred_embedding = normalize_embeddings(pred_embedding.unsqueeze(0))
    target_embedding = emb_model.encode(target, convert_to_tensor=True)
    target_embedding = normalize_embeddings(target_embedding.unsqueeze(0))
    sim_score = cos_sim(pred_embedding, target_embedding).item()
    return sim_score

def semantic_retention_eval(emb_model_path, eval_data_path, target_attribute, pred_attribute):
    # Load SBert model
    embedding_model = SentenceTransformer(emb_model_path)
    # Read data
    data = read_json(eval_data_path)
    all_sim_scores = []
    sim_score_dict = {}
    for type in ['naive', 'paraphrase', 'neighbor']:
        data_filter = [data_point for data_point in data if data_point['query_type'] == type]
        if len(data_filter) == 0:
            continue
        # Calculate
        sim_scores = []
        for i in range(len(data_filter)):
            data_point = data_filter[i]
            sim_score = calc_sim_score(emb_model=embedding_model,
                                       data_point=data_point,
                                       target_attribute=target_attribute,
                                       pred_attribute=pred_attribute)
            sim_scores.append(sim_score)
            all_sim_scores.append(sim_score)
        sim_score_dict[type] = round(sum(sim_scores) / len(sim_scores) * 100, 2)
    sim_score_dict['all'] = round(sum(all_sim_scores) / len(all_sim_scores) * 100, 2)
    sim_score_dict['hm'] = round(hmean([sim_score_dict['naive'], sim_score_dict['paraphrase'], sim_score_dict['neighbor']]), 2)
    return sim_score_dict