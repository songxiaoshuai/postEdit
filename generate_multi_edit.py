"""
Generate edited responses for a query.
"""
import torch
import json
import random
import os
from data.prompter import Prompter
from tqdm import tqdm
from finetune import (
    load_model,
    load_tokenizer,
    smart_tokenizer_and_embedding_resize,
    )


from argparse import ArgumentParser
from peft import PeftModel
from transformers import (
    GenerationConfig, 
)
from sentence_transformers import SentenceTransformer
from utils.util import read_json,save_json
from utils.edit_memory import EditMemory
IGNORE_INDEX = -100

def generate(model,prompter,edit_data_point,query_data_point,tokenizer,temperature,top_p,top_k,num_beams,max_new_tokens):
    prompt = prompter.generate_prompt(edit_data_point=edit_data_point,query_data_point=query_data_point)
    input_ids=tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )['input_ids']

    generation_config = GenerationConfig(
        # do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=False)
    return prompter.get_response(output,query_data_point)


def load_lora_weights(model,lora_weights_path):
    model = PeftModel.from_pretrained(
        model,
        lora_weights_path,
        torch_dtype=torch.float16,
        )
    return model


def save_pred_results(save_file_path,test_data,pred_list):
    assert len(test_data)==len(pred_list)
    new_data=[]
    for i in range(len(test_data)):
        data_point=test_data[i]
        data_point['pred_response_after_edit']=pred_list[i]
        new_data.append(data_point)
    with open(save_file_path,'w',encoding='utf-8') as file:
        json.dump(new_data,file,ensure_ascii=False)


def main(args):
    # initialize edit_memory
    embedding_model=SentenceTransformer(args.embedding_model_path)
    edit_memory=EditMemory(memory_mode=args.edit_memory_mode,
                            embedding_model=embedding_model
                            )
    # load model, tokenizer
    model=load_model(model_path=args.base_model_path,load_in_bits=8)
    model = load_lora_weights(model=model,lora_weights_path=args.lora_weights_path)
    tokenizer,special_tokens_dict=load_tokenizer(model_path=args.base_model_path,
                                                model_max_length=512)
    print(special_tokens_dict)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # load data
    prompter=Prompter(args.template_name)
    edit_data=read_json(read_file_path=args.edit_data_path)
    query_data=read_json(read_file_path=args.query_data_path)
    preds_list=[]
    memory_size = args.memory_size if args.memory_size is not None else len(edit_data)
    if args.edit_memory_mode in ['single_edit', 'sequence_edit']:
        # For single_edit or sequence_edit, continuously update edit_memory and test
        for i in tqdm(range(memory_size)):
            edit_data_point = edit_data[i]
            # Inject edit
            edit_memory.add_edit(edit_id=edit_data_point['edit_id'],
                requested_rewrite=edit_data_point['requested_rewrite'],
                                target_old=edit_data_point['target_old'],
                                target_new=edit_data_point['target_new'])
            # test query
            for j in range(len(query_data)):
                query_data_point=query_data[j]
                if query_data_point['edit_id']==edit_data_point['edit_id']:
                    # retrival edit datapoint
                    _,_,retrival_knowledges=edit_memory.retrival(query_data_point=query_data_point)
                    response=generate(model=model,
                                    prompter=prompter,
                                    tokenizer=tokenizer,
                                    edit_data_point=retrival_knowledges,
                                    query_data_point=query_data_point,
                                    temperature=0.1,
                                    top_p=0.75,
                                    top_k=40,
                                    num_beams=4,
                                    max_new_tokens=512,
                                    )
                    query_data_point['pred_response_after_edit']=response
                    preds_list.append(query_data_point)
                    if len(preds_list) % 10 == 0:
                        save_json(save_file_path=args.save_reult_path, data=preds_list) 
    else:
        # For batch_edit, inject multiple edits at once
        edit_ids = []
        random.seed(args.seed)
        random.shuffle(edit_data)
        for i in tqdm(range(memory_size)):
            edit_data_point = edit_data[i]
            edit_ids.append(edit_data[i]['edit_id'])
            edit_memory.add_edit(
                edit_id=edit_data_point['edit_id'],
                requested_rewrite=edit_data_point['requested_rewrite'],
                target_old=edit_data_point['target_old'],
                target_new=edit_data_point['target_new'])
        edit_memory.print_memory_info()
        # Test query
        for i in tqdm(range(len(query_data))):
            query_data_point = query_data[i]
            if query_data_point['edit_id'] in edit_ids:
                # retrival edit datapoint
                _,_,retrival_knowledges=edit_memory.retrival(query_data_point=query_data_point)
                # generate preds
                response=generate(model=model,
                                prompter=prompter,
                                tokenizer=tokenizer,
                                edit_data_point=retrival_knowledges,
                                query_data_point=query_data_point,
                                temperature=0.1,
                                top_p=0.75,
                                top_k=40,
                                num_beams=4,
                                max_new_tokens=512,
                                )
                query_data_point['pred_response_after_edit']=response
                preds_list.append(query_data_point)
                if len(preds_list) % 10 == 0:
                    save_json(save_file_path=args.save_reult_path, data=preds_list) 
    # Save results
    save_json(save_file_path=args.save_reult_path, data=preds_list)    



if __name__=='__main__':
    parser = ArgumentParser(description='Training Arguments.')
    parser.add_argument('--seed',type=int,default=0)

    # model and tokenizer params
    model_group = parser.add_argument_group(title='model options')
    model_group.add_argument("--base_model_path",type = str,required=True)
    model_group.add_argument("--lora_weights_path",type = str,required=True)
    parser.add_argument("--model_max_length",type = int,default = 512,
                                help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    # edit_memory params
    edit_group = parser.add_argument_group(title='edit options')
    edit_group.add_argument("--embedding_model_path",type = str,default='/newdisk/sxs/all-MiniLM-L6-v2')
    edit_group.add_argument("--edit_memory_mode",type = str,required=True,
                        choices=['single_edit','batch_edit','sequence_edit'])
    edit_group.add_argument("--memory_size",type = int,default=None)
    
    # data params
    data_group = parser.add_argument_group(title='data options')
    data_group.add_argument("--dataset",type = str,required=True,choices=['counterfact','zsre','counterfact+','mquake'])
    data_group.add_argument("--edit_data_path",type = str,required=True) 
    data_group.add_argument("--query_data_path",type = str,required=True) 
    data_group.add_argument("--template_name",type=str,required=True)
    data_group.add_argument("--result_file_name",type = str,required=True)
    # generate params
    args = parser.parse_args()
    args.save_reult_path = os.path.join('outputs', args.dataset, args.edit_memory_mode, f'{args.result_file_name}.json')
    print('save_result_path: ', args.save_reult_path)
    main(args)
    print(f'finish {args.save_reult_path}')