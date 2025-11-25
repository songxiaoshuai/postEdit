import copy
import logging
import os
import random
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer
from argparse import ArgumentParser
from data.prompter import Prompter
from utils.util import read_json
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
    PreTrainedModel,
    PreTrainedTokenizer
)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_RETAIN_TOKEN= "[RETAIN]"




def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)




class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, prompter,data_path: str, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_json(data_path)
        random.shuffle(list_data_dict)
        logging.warning("Formatting inputs...")
        sources = [prompter.generate_prompt(data_point=data_point) for data_point in list_data_dict]
        targets = [f"{prompter.generate_prompt_target(data_point=data_point)}{tokenizer.eos_token}" for data_point in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, prompter,train_data_path,eval_data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,prompter=prompter,data_path=train_data_path)
    if eval_data_path is not None:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, prompter=prompter,data_path=eval_data_path)
    else:
        eval_dataset=None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)



def lora_model(model,lora_r,target_modules):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=16,
        target_modules=target_modules,
        fan_in_fan_out=False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model(model_path,load_in_bits):
    if load_in_bits is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=False
        )   .half().cuda()    
    else:
        # 4-bit quantization config
        bnb_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        # 8-bit quantization config
        bnb_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True if load_in_bits == 8 else False,
            quantization_config=bnb_config_4bit if load_in_bits == 4 else bnb_config_8bit,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            use_safetensors=False
        )     
        if load_in_bits == 8:
            model = prepare_model_for_int8_training(model)
        elif load_in_bits == 4:
            model = prepare_model_for_kbit_training(model)
    return model


def load_tokenizer(model_path,model_max_length):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # add self-defined special token
    # special_tokens_dict['additional_special_tokens'] = [DEFAULT_RETAIN_TOKEN]
    # add pad token
    # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    return tokenizer,special_tokens_dict


def compute_metrics(eval_preds):
    """
    Compute evaluation metrics.
    
    Args:
        eval_preds: EvalPrediction object, a named tuple with at least 
                   predictions and label_ids fields
                   - predictions: ndarray [len(eval_data), max_seq_length, max_vocab_size]
                   - label_ids: ndarray [len(eval_data), max_seq_length]
    
    Returns:
        Dictionary containing metrics and their values
    """
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    # Get predictions
    predictions = np.argmax(logits, axis=-1)
    mask = (labels != -100)
    # Calculate overlapping positions
    overlap_positions = np.logical_and(predictions == labels, mask)
    
    # Calculate total overlaps
    overlap_count = np.sum(overlap_positions)
    
    # Calculate total valid positions (non-100 positions)
    total_valid_positions = np.sum(mask)
    
    # Calculate overlap score
    overlap_score = overlap_count / total_valid_positions
    
    return {'overlap_score': overlap_score}


def train(args):
    # set seed
    set_seed(args.seed)
    # load model
    model=load_model(model_path=args.model_path,load_in_bits=args.load_in_bits)
    if args.use_lora:
        model=lora_model(model=model,
                        lora_r=args.lora_r,
                        target_modules=args.lora_target_modules)
    # load tokenizer and resize model embedding
    tokenizer,special_tokens_dict=load_tokenizer(model_path=args.model_path,
                                                model_max_length=args.model_max_length)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # load data
    prompter=Prompter(template_name=args.template_name)
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        prompter=prompter,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path)
    trainer = Trainer(model=model, 
                    tokenizer=tokenizer, 
                    args=TrainingArguments(
                        output_dir=args.output_dir,
                        num_train_epochs=args.num_train_epochs,
                        per_device_train_batch_size=args.per_device_train_batch_size,
                        per_device_eval_batch_size=args.per_device_eval_batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        evaluation_strategy=args.evaluation_strategy,
                        save_strategy=args.save_strategy,
                        warmup_ratio=args.warmup_ratio,
                        #   save_steps=args.save_steps,
                        #   save_total_limit=args.save_total_limit,
                        learning_rate=args.learning_rate,
                        weight_decay=0., 
                        lr_scheduler_type="cosine",
                        logging_steps=1,
                        #   fsdp="full_shard auto_wrap",
                        #   fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
                        tf32=args.tf32,
                    ), 
                    compute_metrics=compute_metrics,
                    **data_module)
    trainer.train()
    # Save LoRA weights
    model.save_pretrained(os.path.join(args.output_dir, 'lora_weights', args.lora_weight_name))
    results = trainer.evaluate()
    print(results)
    # trainer.save_state()
    # trainer.save_model()
    # trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description='Training Arguments.')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument("--cache_dir",default=None)
    parser.add_argument("--output_dir",type = str,default='output')
    
    # model and tokenizer params
    model_group = parser.add_argument_group(title='model options')
    model_group.add_argument("--model_path",type = str,required=True)
    parser.add_argument("--model_max_length",type = int,default = 512,
                                help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")

    # peft
    peft_group = parser.add_argument_group(title='peft options')
    peft_group.add_argument("--load_in_bits",type=int,default=None,choices=[4,8,None])
    peft_group.add_argument("--use_lora",type = bool,default=False)
    peft_group.add_argument("--lora_r",type = int,default=8,help="lora的秩")
    peft_group.add_argument("--lora_weight_name",type = str,default='test')
    peft_group.add_argument("--lora_target_modules",type = str,nargs='+',
                            default=['q_proj','v_proj','k_proj','o_proj','gate_proj','down_proj','up_proj'])
    # parser.add_argument("--bf16",type = bool,default = True)
    parser.add_argument("--tf32",type = bool,default = True) 

    # data params
    data_group = parser.add_argument_group(title='data options')
    data_group.add_argument("--train_data_path",type = str,required=True)
    data_group.add_argument("--eval_data_path",type = str,default=None)
    data_group.add_argument("--template_name",type=str,default='alpaca')

    # train params
    train_group = parser.add_argument_group(title='training options')
    train_group.add_argument("--num_train_epochs",type = int,default = 3)
    train_group.add_argument("--per_device_train_batch_size",type = int,default = 4)
    train_group.add_argument("--per_device_eval_batch_size",type = int,default = 4)
    train_group.add_argument("--gradient_accumulation_steps",type = int,default = 8)
    train_group.add_argument("--learning_rate", type=float, default=2e-5, 
                            help="Full fine-tuning needs smaller lr, LoRA needs larger lr")
    train_group.add_argument("--warmup_ratio",type = float,default = 0.1)
    train_group.add_argument("--lr_scheduler_type",type = str,default = 'cosine')
    train_group.add_argument("--optim",type = str,default='adamw_torch')
    train_group.add_argument("--weight_decay",type = float,default = 0.)
    train_group.add_argument("--evaluation_strategy",type = str,default = "no",choices=['no','epoch','steps'])
    # save checkpoint
    train_group.add_argument("--save_strategy",type = str,default = 'no',choices=['no','epoch','steps'])
    train_group.add_argument("--save_steps",type = int,default = 2)
    train_group.add_argument("--save_total_limit",type = int,default = 2)
    train_group.add_argument("--fsdp",type = str,default = 'full_shard auto_wrap')
    train_group.add_argument("--fsdp_transformer_layer_cls_to_wrap",type = str,default = 'LlamaDecoderLayer')
    train_group.add_argument("--report_to",type = str,default = 'tensorboard')

    args = parser.parse_args()
    os.environ["WANDB_DISABLED"]="true"
    train(args)
