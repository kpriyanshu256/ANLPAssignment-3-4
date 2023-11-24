'''
CUDA_VISIBLE_DEVICES=1 python train_peft.py -tr kpriyanshu256/semeval-task-8-b-v2-mistral-7b -sb B --model mistralai/Mistral-7B-v0.1 -p preds.jsonl -sp mistral
'''
from datasets import Dataset, load_dataset
import pandas as pd
import evaluate
import numpy as np
import transformers
from transformers import (AutoModelForSequenceClassification, 
                            TrainingArguments, 
                            Trainer, 
                            DataCollatorWithPadding, 
                            AutoTokenizer, 
                            set_seed, 
                            BitsAndBytesConfig)
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers.trainer_pt_utils import get_parameter_names

MAX_LEN = 512*2

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=MAX_LEN)


def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(acc_metric.compute(predictions=predictions, references = labels))
    return results


def fine_tune(train_dataset, valid_dataset, checkpoints_path, id2label, label2id, model_name):
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)     # put your model here
    tokenizer.pad_token = tokenizer.eos_token
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id,
        quantization_config=bnb_config,
        device_map={"":0}
    )
    model.config.pretraining_tp = 1 # 1 is 7b
    model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        optim='paged_adamw_32bit',
        load_best_model_at_end=True,
        report_to = "none",
        gradient_accumulation_steps=16,
        # gradient_checkpointing = True,
        max_grad_norm=0.3,
        save_total_limit = 1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # optimizers=(adam_bnb_optim, None)
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)


def test(test_dataset, model_name, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load best model
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id,
        quantization_config=bnb_config,
        device_map={"":0}
    )
    model.config.pretraining_tp = 1 # 1 is 7b
    model.config.pad_token_id = tokenizer.pad_token_id
            
    model = PeftModel.from_pretrained(model, model_path)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)#, max_length=MAX_LEN)
    

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument("--save_path", "-sp", required=True, help="Path where to save the prediction file.", type=str)
    

    args = parser.parse_args()

    random_seed = 42
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl
    save_path = args.save_path

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)
        
    dataset = load_dataset(args.dataset_name)    
    train_dataset = dataset['train']
    valid_dataset = dataset["val"]
    test_dataset = dataset["test"]
   
    
    # train detector model
    logging.info("Starting training.......")
    fine_tune(train_dataset, valid_dataset,
              f"{save_path}/subtask{subtask}/{random_seed}", id2label, label2id, model)


    test_dataset = dataset['test']
    
    # test detector model
    results, predictions = test(test_dataset, model, f"{save_path}/subtask{subtask}/{random_seed}/best/", id2label, label2id)

    logging.info(results)
    
    test_ids = []
    for x in test_dataset:
        test_ids.append(x["id"])

    print(results)
    predictions_df = pd.DataFrame({'id': test_ids, 'label': predictions})
    predictions_df.to_json(f"{save_path}/subtask{subtask}/{random_seed}/best/{prediction_path}", lines=True, orient='records')
