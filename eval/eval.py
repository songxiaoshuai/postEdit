import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from lexical_edit import lexical_edit_eval
from semantic_edit import semantic_edit_eval
from lexical_retention import lexical_retention_eval
from semantic_retention import semantic_retention_eval

def eval(dataset,eval_data_path,target_attribute,pred_attribute,device,emb_model_path):
    lexical_edit_score=lexical_edit_eval(eval_data_path=eval_data_path,eval_column=pred_attribute)
    semantic_edit_score=semantic_edit_eval(dataset=dataset,eval_data_path=eval_data_path,eval_column=pred_attribute,device=device)
    lexical_retention_score=lexical_retention_eval(eval_data_path=eval_data_path,
                                                target_attribute=target_attribute,pred_attribute=pred_attribute)
    semantic_retention_score=semantic_retention_eval(emb_model_path=emb_model_path,eval_data_path=eval_data_path,
                                                target_attribute=target_attribute,pred_attribute=pred_attribute)
    print('Eval File:',eval_data_path)
    print('lexical_edit (hit):',lexical_edit_score)
    print('semantic_edit (NLI):',semantic_edit_score)
    print('lexical_retention (ROUGE-1) :',lexical_retention_score)
    print('semantic_retention (SBert Sim):',semantic_retention_score)

if __name__ == '__main__':
    # Settings
    dataset = 'counterfact'
    eval_data_path = 'your_inference_result.json'
    emb_model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    target_attribute = "response_before_edit"  # Output before editing
    pred_attribute = "pred_response_after_edit"  # Predicted output after editing
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Run evaluation
    eval(dataset, eval_data_path, target_attribute, pred_attribute, device, emb_model_path)  
