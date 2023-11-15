import os
import gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import math
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import entropy

from accelerate import cpu_offload, dispatch_model
from accelerate.utils.modeling import infer_auto_device_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "mistralai/Mistral-7B-v0.1"
model_name = "mistral-7b"

dataset = load_dataset("kpriyanshu256/semeval-task-8-b-v2")

MAX_LEN = 512

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=not True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=bnb_config,
                                             use_cache = False, 
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def extract_features(text, model):

    input_ids = tokenizer(text, truncation=True, max_length=MAX_LEN)
    output_ids = tokenizer(text, truncation=True, max_length=MAX_LEN)

    input_tokens = len(input_ids.input_ids)
    output_tokens = len(output_ids.input_ids)

    max_length = min(MAX_LEN, max(input_tokens, output_tokens))
    input_ids = tokenizer.pad(input_ids, 
                            padding='max_length', 
                            max_length=max_length, 
                            return_tensors="pt").input_ids.reshape((1, -1))

    output_ids = tokenizer.pad(output_ids, 
                                padding='max_length', 
                                max_length=max_length, 
                                return_tensors="pt").input_ids.reshape((1, -1))


    outputs = model(input_ids, labels=output_ids)
    
    
    loss = outputs.loss
    probs = outputs.logits.softmax(-1).detach().cpu()
    ids = output_ids.tolist()[0][1:]

    del input_tokens, output_tokens
    gc.collect()
    torch.cuda.empty_cache()
    
    tokens = []
    logprobs = []
    l = 0
    val_ids = 0
    for i, id in enumerate(ids):
        p = probs[0, i, id].item()
        token = tokenizer.decode(id)
        tokens.append(token)
        logprob = math.log(p)
        logprobs.append(logprob)
        if id != tokenizer.eos_token_id:
            l -= logprob
            val_ids += 1

    estimated_loss = l / val_ids
    mean_lowest25 = np.mean(sorted(logprobs)[:25])
    mean_highest25 = np.mean(sorted(logprobs)[-25:])
    maxp = np.max(logprobs)
    minp = np.min(logprobs)
    rangep = maxp - minp
    meanp = np.mean(logprobs)
    stdp = np.std(logprobs)
    entropyp = entropy(np.exp(logprobs))
    if stdp != 0:
        kurtosisp = np.mean((logprobs - meanp)**4) / stdp ** 4
        skewnessp = np.mean((logprobs - meanp)**3) / stdp ** 3
    else:
        kurtosisp = 0
        skewnessp = 0
    perplexityp = np.exp(-np.mean(logprobs))

    return [
        estimated_loss,
        mean_lowest25,
        mean_highest25,
        maxp,
        minp,
        rangep,
        meanp,
        stdp,
        entropyp,
        kurtosisp,
        skewnessp,
        perplexityp,
    ]


def compute_features(ds, model, model_name):

    new_df = list()
    base_features = ["estimated_loss", "mean_lowest25", "mean_highest25", "max", "min", "range", "mean", "std", "entropy", "kurtosis", "skewness", "perplexity"]
    df_features = [f"{model_name}_{item}" for item in base_features]

    for i, x in tqdm(enumerate(ds), total=len(ds)):
        with torch.no_grad():
            new_df.append(extract_features(x['text'], model))

    new_df = pd.DataFrame(new_df, columns=df_features)
    return new_df



logprob_train = compute_features(dataset['train'], model, model_name)
logprob_train.to_csv("mistral_train.csv", index=False)

logprob_val = compute_features(dataset['val'], model, model_name)
logprob_val.to_csv("mistral_val.csv", index=False)

logprob_test = compute_features(dataset['test'], model, model_name)
logprob_test.to_csv("mistral_test.csv", index=False)

logprob_train = Dataset.from_pandas(logprob_train)
logprob_val = Dataset.from_pandas(logprob_val)
logprob_test = Dataset.from_pandas(logprob_test)

train_dataset = concatenate_datasets([dataset['train'], logprob_train], axis=1)
val_dataset = concatenate_datasets([dataset['val'], logprob_val], axis=1)
test_dataset = concatenate_datasets([dataset['test'], logprob_test], axis=1)


ds = DatasetDict({"train": train_dataset, 
                   "val": val_dataset, 
                   "test":test_dataset})

ds.push_to_hub("kpriyanshu256/semeval-task-8-b-v2-mistral-7b")
