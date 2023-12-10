from datasets import load_dataset
from sklearn import linear_model
import numpy as np

M = "mistral-7b"
F = ["estimated_loss", "mean_lowest25", "mean_highest25",\
      "max", "min", "range", "mean", "std", "entropy", \
        "kurtosis", "skewness", "perplexity"]

F = [M+"_"+i for i in F]

DS = [
    "kpriyanshu256/semeval-task-8-a-mono-v2-mistral-7b",
    "kpriyanshu256/semeval-task-8-a-multi-v2-mistral-7b",
    "kpriyanshu256/semeval-task-8-b-v2-mistral-7b",
]

ds = load_dataset(DS[2])

def load_matrix(ds, split):
    X, y = [], []
    for examples in ds[split]:
        X.append([np.array(examples[x]) for x in F])
        y.append(examples['label'])
    
    X = np.array(X)
    y = np.array(y)
    return X, y


X_train, y_train = load_matrix(ds, "train")
X_val, y_val = load_matrix(ds, "val")
X_test, y_test = load_matrix(ds, "test")

model = linear_model.LogisticRegression()

model.fit(X_train, y_train)
print("Train accuracy ", model.score(X_train, y_train))
print("Val accuracy ", model.score(X_val, y_val))
print("Test accuracy ", model.score(X_test, y_test))