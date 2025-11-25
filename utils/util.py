import json
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix

def read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def save_json(save_file_path,data,mode='w'):
    with open(save_file_path,mode,encoding='utf-8')as fp:
        json.dump(data,fp,ensure_ascii=False)

def read_txt(read_file_path):
    with open(read_file_path, 'r') as f:
        content = f.read()
    return content

def eval_classification(preds,targets):
    assert len(preds)==len(targets)
    acc=round(accuracy_score(y_pred=preds,y_true=targets),4)
    pre=round(precision_score(y_pred=preds,y_true=targets),4)
    recall=round(recall_score(y_pred=preds,y_true=targets),4)
    confusion=confusion_matrix(y_pred=preds,y_true=targets)
    return {'accuracy':acc,'precision':pre,'recall':recall,'confusion_metrix':confusion}