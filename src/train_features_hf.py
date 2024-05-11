import pprint

from tqdm.notebook import tqdm
from datasets import Dataset, load_dataset
import pandas as pd
import evaluate
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, AutoConfig
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging

import torch
import torch.nn as nn
import bitsandbytes as bnb
import safetensors

import transformers
from transformers.trainer_pt_utils import get_parameter_names
from pytorch_tabnet.tab_network import TabNet
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union

MAX_LEN = 512

import joblib as jb

M = "mistral-7b"
F = ["estimated_loss", "mean_lowest25", "mean_highest25", "max", "min", "range", "mean", "std", "entropy", "kurtosis", "skewness", "perplexity"]

F = [M+"_"+i for i in F]


# class Head(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


    
# class Model(nn.Module):
#     def __init__(self, model_name, config):
#         super().__init__()
#         self.transformer = transformers.AutoModel.from_pretrained(model_name)
#         self.classifier = Head(config)
    
#     def forward(self, input_ids=None, attention_mask=None, labels=None, features=None):
#         outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#         )
#         sequence_output = outputs[0]
#         logits = self.classifier(sequence_output)

        
#         loss_fct = nn.CrossEntropyLoss()
        
#         loss = loss_fct(logits, labels.view(-1))
        
#         return SequenceClassifierOutput(
#             loss=loss, logits=logits, 
#             hidden_states=outputs.hidden_states, attentions=outputs.attentions
#         )

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # self.tabnet = TabNet(**{
        #                     'n_d': 4,
        #                     'n_a': 4,
        #                     'n_steps': 4,
        #                     'input_dim': config.feature_size,
        #                     'output_dim': 64,
        #                     'group_attention_matrix': nn.Parameter(torch.eye(config.feature_size), requires_grad=False),
            
        #                 }
        #             )

        # self.feat_transform = nn.Linear(12, 64)
        # self.feat_transform = nn.Sequential(
        #                 nn.BatchNorm1d(config.feature_size),
        #                 nn.Linear(config.feature_size, 64), 
        #                 nn.ReLU(),
        #                 nn.BatchNorm1d(64),
        #                 nn.Linear(64, 64), 
        #                 nn.ReLU(),
        #                 nn.BatchNorm1d(64),
        #                 nn.Linear(64, 64), 
        #                 nn.ReLU(),
        # )
        self.combine = nn.Linear(12 + config.num_labels, config.num_labels)

    def forward(self, features, features_1, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # print(x.shape, features_1.shape)
        # concatenating text features and log prob features
        # features_1 = self.tabnet(features_1)[0]
        # features_1 = self.feat_transform(features_1)
        # x = torch.cat((x, features_1), axis=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = torch.cat((x, features_1), axis=-1)
        x = self.combine(x)
        return x

class Model(transformers.RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = transformers.RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

   
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        features: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, features)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    


def preprocess_function(examples, **fn_kwargs):
    model_inputs = fn_kwargs['tokenizer'](examples["text"], 
                                          truncation=True, 
                                          max_length=MAX_LEN)
    
    features = [np.array(examples[x]).reshape(-1, 1) for x in F]        
    features = np.concatenate(features, axis=1)
    
    model_inputs["features"] = features
    return model_inputs


def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(acc_metric.compute(predictions=predictions, references = labels))
    return results


def fine_tune(train_dataset, valid_dataset, checkpoints_path, id2label, label2id, model):
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    
    model_config = AutoConfig.from_pretrained(model)
    model_config.update({
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
        "feature_size": 12,
    })
    
    model = Model(model_config)
    
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, 
                                                batched=True, 
                                                fn_kwargs={'tokenizer': tokenizer})
    
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, 
                                                batched=True,  
                                                fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)

    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=250,
        load_best_model_at_end=True,
        report_to = "none",
        # gradient_accumulation_steps = 2,
        fp16 = True,
        # gradient_checkpointing = True,
        save_total_limit = 1,
        # max_steps = 10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)
    return trainer


def test_trainer(test_dataset, trainer):
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': trainer.tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=trainer.tokenizer, max_length=MAX_LEN)
    
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds    

def test(test_dataset, model, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model_config = AutoConfig.from_pretrained(model)
    model_config.update({
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
        "feature_dim": 12,
    })
    
#     model = Model(model_config)
#     model.load_state_dict(safetensors.torch.load_file(os.path.join(model_path, "model.safetensors")))

    model = Model.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)
    

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

    args = parser.parse_args()

    random_seed = 0
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl
    

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
    # train_dataset = Dataset.from_pandas(pd.DataFrame(dataset['train']).sample(30000))
    train_dataset = dataset["train"]
    valid_dataset = dataset["val"]#.select(range(100))
    test_dataset = dataset["test"]#.select(range(100))
   
    
    # train detector model
    logging.info("Starting training.......")
    trainer = fine_tune(train_dataset, valid_dataset,
              f"subtask{subtask}/{random_seed}", id2label, label2id, model)


    test_dataset = dataset['test']
    
    # test detector model
    results, predictions = test_trainer(test_dataset, trainer)

    # results, predictions = test(test_dataset, model, f"subtask{subtask}/{random_seed}/best/", id2label, label2id)
    logging.info(results)
    
    test_ids = []
    for x in test_dataset:
        test_ids.append(x["id"])

    pprint.pprint(results)
    predictions_df = pd.DataFrame({'id': test_ids, 'label': predictions})
    predictions_df.to_json(f"subtask{subtask}/{random_seed}/{prediction_path}", lines=True, orient='records')