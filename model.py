import numpy as np
import torch
import transformers

import evaluate
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

def build_classifier(ds_train, ds_val):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(ds_train.classes), id2label=ds_train.classes, label2id=ds_train.label2id
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = transformers.TrainingArguments(
        output_dir="output_dir",
        learning_rate=7e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.03,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    return model, trainer

def predict_sample(model, ds, idx):
    sample = ds.__getitem__(idx, 'pt')
    input_ids = sample['input_ids']
    prediction = torch.argmax(model(input_ids.cuda()).logits).item()
    return prediction

def evaluate_sample(model, ds, idx):
    prediction = predict_sample(model, ds, idx)
    label = ds.__getitem__(idx, 'pt')['labels'].item()
    return prediction, label

def evaluate_classifier(model, ds):
    predictions, labels = [], []
    correct = [0] * len(ds)
    for i in tqdm(range(len(ds))):
        prediction, label = evaluate_sample(model, ds, i)
        predictions.append(prediction)
        labels.append(label)
        if prediction == label:
            correct[i] = 1
    print()

    metrics = dict(
        accuracy = accuracy_score(labels, predictions),
        f1_metrics = dict(zip(
            ['Precision', 'Recall', 'F1', 'Support'], 
            precision_recall_fscore_support(labels, predictions, zero_division=0)
        )),
        macro = f1_score(labels, predictions, average='macro', zero_division=0),
        weighted = f1_score(labels, predictions, average='weighted', zero_division=0),
        correct = correct,
    )

    return metrics