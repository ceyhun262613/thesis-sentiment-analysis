# Importing necessary libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import re


# Defining the BERT network class
class BertNetwork(nn.Module):
    def __init__(self, device):
        super(BertNetwork, self).__init__()
        self.model = BertModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        self.nb_features = self.model.pooler.dense.out_features
        self.drop = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.nb_features, 3)
        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None):
        output = self.model(input_ids, attention_mask, return_dict=True)
        ft = self.drop(output["pooler_output"])
        x = self.fc1(ft)
        return x

    def train(self, train_loader, epochs, criterion, optimizer, scheduler):
        total = 0.0
        correct = 0.0
        tr_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for input_id, attention_mask, label in tepoch:
                input_id = input_id.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label = label.to(self.device)
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

                optimizer.zero_grad()
                output = self.forward(input_id, attention_mask)
                loss = criterion(output, label)
                tr_loss += loss.item()
                scores, prediction = torch.max(output, dim=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total += label.size(0)
                correct += (prediction == label).sum().item()
                accuracy = correct / total
                tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * accuracy)

            torch.cuda.empty_cache()
            print(
                f"\nFinished Training in {epoch + 1} with loss={loss:.2f} and accuracy={100 * accuracy:.2f}"
            )
        return tr_loss, correct

    def test(self, testloader):
        total = 0.0
        correct = 0.0
        tst_loss = 0.0
        y_pred = []
        y_true = []

        with torch.no_grad():
            with tqdm(testloader, unit="batch") as tepoch:
                for input_id, attention_mask, label in tepoch:
                    input_id = input_id.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label = label.to(self.device)
                    y_true.extend(label.cpu().numpy())

                    output = self.forward(input_id, attention_mask)
                    loss = criterion(output, label)
                    tst_loss += loss.item()
                    scores, prediction = torch.max(output.data, 1)
                    y_pred.extend(prediction.cpu().numpy())

                    total += label.size(0)
                    correct += (prediction == label).sum().item()
                    tepoch.set_postfix(accuracy=100.0 * (correct / total))
        print(f"\nFinished Testing with accuracy={100 * (correct / total):.2f}")
        return y_true, y_pred, tst_loss, correct


def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=60,
            padding="max_length",  # Use 'longest' to pad to the longest sequence
            return_attention_mask=True,
            truncation=True,
        )

        input_ids.append(encoded_sent.get("input_ids"))
        attention_masks.append(encoded_sent.get("attention_mask"))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


def preprocess(text):
    text = re.sub(r"(@.*?)[\s]", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")


def save_confusion_matrix_plot(confusion_matrix, fold):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True sentiment")
    plt.xlabel("Predicted sentiment")
    plt.title(f"Confusion Matrix - Fold {fold} - After Epochs")
    plt.savefig(f"confusion_matrix_fold_{fold}.png")
    plt.close()


def save_loss_plot(train_loss, test_loss, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_of_epochs + 1), train_loss, label="Train Loss")
    plt.plot(range(1, num_of_epochs + 1), test_loss, label="Test Loss")
    plt.title(f"Fold {fold} - Training and Testing Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_plot_fold_{fold}.png")
    plt.close()


def save_accuracy_plot(train_acc, test_acc, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_of_epochs + 1), train_acc, label="Train Accuracy")
    plt.plot(range(1, num_of_epochs + 1), test_acc, label="Test Accuracy")
    plt.title(f"Fold {fold} - Training and Testing Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"accuracy_plot_fold_{fold}.png")
    plt.close()


def save_accuracy_plot(train_acc, test_acc, fold):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_of_epochs + 1), train_acc, label="Train Accuracy")
    plt.plot(range(1, num_of_epochs + 1), test_acc, label="Test Accuracy")
    plt.title(f"Fold {fold} - Training and Testing Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"accuracy_plot_fold_{fold}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-epochs", help="Set the number of epochs", type=int, default=10
    )
    parser.add_argument("-folds", help="Set the number of folds", type=int, default=10)
    parser.add_argument(
        "-batch", help="Set the batch size number", type=int, default=32
    )
    parser.add_argument(
        "-model",
        help="Select the transformer model",
        default="nlpaueb/bert-base-greek-uncased-v1",
    )

    args = parser.parse_args()
    num_of_epochs = args.epochs
    num_of_folds = args.folds
    batch_size = args.batch
    class_names = ["negative", "neutral", "positive"]

    df = pd.read_csv("twitter.csv")
    sentences = df["text"].values
    labels = df["sentiment"].values
    num_records, num_columns = df.shape
    print(f"Number of records: {num_records}")
    print(f"Number of columns: {num_columns}")

    tokenizer = BertTokenizer.from_pretrained(
        "nlpaueb/bert-base-greek-uncased-v1", do_lower_case=True
    )
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pytorch running on: {device}")

    fold = 0
    foldperf = {}

    sum_cm = np.zeros((len(class_names), len(class_names)))

    kf = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=42)

    lr_true_labels = []
    lr_predicted_labels = []

    nb_true_labels = []
    nb_predicted_labels = []

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    average_test_accuracy_across_folds = []

    kf = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(sentences, labels), 1):
        print("Fold {}".format(fold))

        X_train, X_val = sentences[train_index], sentences[test_index]
        y_train, y_val = labels[train_index], labels[test_index]

        # Convert text data to TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_val_tfidf = tfidf_vectorizer.transform(X_val)

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)  # Adjust parameters as needed
        lr_model.fit(X_train_tfidf, y_train)
        lr_predictions = lr_model.predict(X_val_tfidf)

        lr_true_labels.extend(y_val)
        lr_predicted_labels.extend(lr_predictions)

        # Naïve Bayes
        nb_model = MultinomialNB()  # Adjust parameters as needed
        nb_model.fit(X_train_tfidf, y_train)
        nb_predictions = nb_model.predict(X_val_tfidf)

        nb_true_labels.extend(y_val)
        nb_predicted_labels.extend(nb_predictions)

        train_inputs, train_masks = preprocessing_for_bert(X_train)
        tst_inputs, tst_masks = preprocessing_for_bert(X_val)
        train_labels = torch.tensor(y_train)
        tst_labels = torch.tensor(y_val)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        tst_data = TensorDataset(tst_inputs, tst_masks, tst_labels)
        tst_sampler = SequentialSampler(tst_data)
        tst_dataloader = DataLoader(
            tst_data, sampler=tst_sampler, batch_size=batch_size
        )

        net = BertNetwork(device).to(device)

        optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5, eps=1e-8)

        total_steps = len(train_dataloader) * num_of_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(num_of_epochs):
            train_loss, train_correct = net.train(
                train_dataloader, num_of_epochs, criterion, optimizer, scheduler
            )
            true_labels, predicted_labels, test_loss, test_correct = net.test(
                tst_dataloader
            )

            train_loss = train_loss / len(train_dataloader.sampler)
            train_acc = train_correct / len(train_dataloader.sampler) * 100
            test_loss = test_loss / len(tst_dataloader.sampler)
            test_acc = test_correct / len(tst_dataloader.sampler) * 100

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            # print(f"\nEpoch {epoch + 1}/{num_of_epochs}")

        # Display confusion matrix after all epochs in the current fold
        cm_fold = confusion_matrix(tst_labels, predicted_labels, labels=[0, 1, 2])
        sum_cm += cm_fold  # Sum the confusion matrices across all folds

        df_cm_fold = pd.DataFrame(cm_fold, index=class_names, columns=class_names)

        # Calculate accuracy for the current fold
        _, _, _, correct_test = net.test(tst_dataloader)
        test_accuracy = correct_test / len(tst_dataloader.sampler) * 100
        average_test_accuracy_across_folds.append(test_accuracy)
        print(f"Accuracy for Fold {fold}: {test_accuracy:.2f}%")

        save_confusion_matrix_plot(df_cm_fold, fold)
        save_loss_plot(history["train_loss"], history["test_loss"], fold)
        save_accuracy_plot(history["train_acc"], history["test_acc"], fold)

        foldperf["fold{}".format(fold)] = history

        # Display confusion matrix

    df_sum_cm = pd.DataFrame(sum_cm, index=class_names, columns=class_names)
    save_confusion_matrix_plot(df_sum_cm, "sum")

    average_test_accuracy = sum(average_test_accuracy_across_folds) / num_of_folds
    print(f"\nAverage Test Accuracy Across all Folds: {average_test_accuracy:.2f}%")

    print("\nLogistic Regression Classification Report:")
    print(classification_report(lr_true_labels, lr_predicted_labels))

    print("\nNaïve Bayes Classification Report:")
    print(classification_report(nb_true_labels, nb_predicted_labels))

    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    for f in range(1, num_of_folds + 1):
        tl_f.append(np.mean(foldperf["fold{}".format(f)]["train_loss"]))
        testl_f.append(np.mean(foldperf["fold{}".format(f)]["test_loss"]))

        ta_f.append(np.mean(foldperf["fold{}".format(f)]["train_acc"]))
        testa_f.append(np.mean(foldperf["fold{}".format(f)]["test_acc"]))
    print("Performance of {} fold cross-validation".format(num_of_folds))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)
        )
    )

    diz_ep = {
        "train_loss_ep": [],
        "test_loss_ep": [],
        "train_acc_ep": [],
        "test_acc_ep": [],
    }

    for i in range(num_of_epochs):
        diz_ep["train_loss_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["train_loss"][i]
                    for f in range(num_of_folds)
                ]
            )
        )
        diz_ep["test_loss_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["test_loss"][i]
                    for f in range(num_of_folds)
                ]
            )
        )
        diz_ep["train_acc_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["train_acc"][i]
                    for f in range(num_of_folds)
                ]
            )
        )
        diz_ep["test_acc_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["test_acc"][i]
                    for f in range(num_of_folds)
                ]
            )
        )

    for f in range(1, num_of_folds + 1):
        tst_data_fold = TensorDataset(tst_inputs, tst_masks, tst_labels)
        tst_sampler_fold = SequentialSampler(tst_data_fold)
        tst_dataloader_fold = DataLoader(
            tst_data_fold, sampler=tst_sampler_fold, batch_size=batch_size
        )

        true_labels_fold, predicted_labels_fold, _, _ = net.test(tst_dataloader_fold)
        precision_fold = precision_score(
            true_labels_fold, predicted_labels_fold, average="weighted", zero_division=1
        )
        recall_fold = recall_score(
            true_labels_fold, predicted_labels_fold, average="weighted", zero_division=1
        )
        f1_fold = f1_score(
            true_labels_fold, predicted_labels_fold, average="weighted", zero_division=1
        )

        print(f"\nMetrics for Fold {f}:")
        print(
            f"Precision: {precision_fold:.4f}, Recall: {recall_fold:.4f}, F1-Score: {f1_fold:.4f}"
        )

        foldperf["fold{}".format(f)]["precision"] = precision_fold
        foldperf["fold{}".format(f)]["recall"] = recall_fold
        foldperf["fold{}".format(f)]["f1_score"] = f1_fold

    precision_avg = np.mean(
        [foldperf["fold{}".format(f)]["precision"] for f in range(1, num_of_folds + 1)]
    )
    recall_avg = np.mean(
        [foldperf["fold{}".format(f)]["recall"] for f in range(1, num_of_folds + 1)]
    )
    f1_avg = np.mean(
        [foldperf["fold{}".format(f)]["f1_score"] for f in range(1, num_of_folds + 1)]
    )

    print("\nAverage Metrics Across all Folds:")
    print(
        f"Average Precision: {precision_avg:.4f}, Average Recall: {recall_avg:.4f}, Average F1-Score: {f1_avg:.4f}"
    )
