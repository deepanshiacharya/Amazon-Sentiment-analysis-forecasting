import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
from model import BERT_BiLSTM_Attention, SentimentClassifier


'''Device Configuration'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

'''Pairwise distance function for triplet loss calculation'''
def pairwise_distance(embeddings):
    dot = torch.matmul(embeddings, embeddings.T)
    norm = torch.diagonal(dot)
    return torch.clamp(norm.unsqueeze(1) - 2 * dot + norm.unsqueeze(0), min=0.0)


'''Batch hard triplet loss function'''
def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    dists = pairwise_distance(embeddings)
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.T).float()
    mask_neg = (labels != labels.T).float()

    hardest_pos = (dists * mask_pos).max(dim=1)[0]
    hardest_neg = (dists + dists.max().item() * (1 - mask_neg)).min(dim=1)[0]
    return torch.relu(hardest_pos - hardest_neg + margin).mean()


'''Dataset for triplet generation (anchor, positive, negative)'''
class TripletSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer, self.max_len = tokenizer, max_len
        self.class_to_indices = {
            label: df[df['sentiment'] == label].reset_index(drop=True) for label in df['sentiment'].unique()
        }

    def __len__(self): return len(self.class_to_indices[0])

    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")

    def __getitem__(self, idx):
        anchor_sentiment = random.choice([0, 1, -1])  
        anchor_row = self.class_to_indices[anchor_sentiment].iloc[idx]
        anchor_text = anchor_row['summary']

        '''Positive: sample from same class (excluding the anchor)'''
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.class_to_indices[anchor_sentiment].index.tolist())
        positive_text = self.class_to_indices[anchor_sentiment].loc[pos_idx, 'summary']
        
        '''Negative: sample from a different class'''
        negative_classes = [label for label in self.class_to_indices if label != anchor_sentiment]
        neg_class = random.choice(negative_classes)
        neg_idx = random.choice(self.class_to_indices[neg_class].index.tolist())
        negative_text = self.class_to_indices[neg_class].loc[neg_idx, 'summary']

        ''' Tokenize'''
        anchor = self.tokenize(anchor_text)
        pos = self.tokenize(positive_text)
        neg = self.tokenize(negative_text)

        return {
            'anchor_input_ids': anchor['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor['attention_mask'].squeeze(0),
            'pos_input_ids': pos['input_ids'].squeeze(0),
            'pos_attention_mask': pos['attention_mask'].squeeze(0),
            'neg_input_ids': neg['input_ids'].squeeze(0),
            'neg_attention_mask': neg['attention_mask'].squeeze(0)
        }


'''Encoder training function with triplet loss'''
def train_triplet_encoder(model, df, tokenizer, batch_size=32, patience=5):
    model.train()
    loader = DataLoader(TripletSentimentDataset(df, tokenizer), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_train_loss = float('inf')
    patience_counter = 0
    epoch = 0

    while True:
        epoch += 1
        total_loss = 0
        model.train()

        for batch in tqdm(loader, desc=f"Triplet Epoch {epoch}"):
            input_ids = torch.cat([batch['anchor_input_ids'], batch['pos_input_ids'], batch['neg_input_ids']], dim=0).to(device)
            attention_mask = torch.cat([batch['anchor_attention_mask'], batch['pos_attention_mask'], batch['neg_attention_mask']], dim=0).to(device)
            labels = torch.cat([
                torch.zeros(len(batch['anchor_input_ids'])),
                torch.ones(len(batch['pos_input_ids'])),
                torch.full((len(batch['neg_input_ids']),), 2)
            ]).long().to(device)

            embeddings = model(input_ids, attention_mask)
            loss = batch_hard_triplet_loss(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_train_loss - 1e-3:
            best_train_loss = avg_loss
            patience_counter = 0
        elif epoch == 10            :
            break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no significant improvement)")
                break


'''Function to extract embeddings from the encoder'''
def extract_embeddings(df, model, tokenizer, batch_size=64, max_len=128, device="cpu"):
    texts, sentiments = df["summary"].tolist(), df["sentiment"].tolist()
    model.eval(); all_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        input_ids, attn = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        with torch.no_grad():
            embeds = model(input_ids, attn).cpu().numpy()
        all_embeds.append(embeds)

    return np.vstack(all_embeds), sentiments


'''Classifier training function with early stopping'''
def train_classifier(encoder, classifier, df, tokenizer, batch_size=64, patience=5, delta=0.001):
    classifier.train()
    embeddings, labels = extract_embeddings(df, encoder, tokenizer, device=device)
    labels = torch.tensor([label + 1 for label in labels], dtype=torch.long).to(device)
    dataset = DataLoader(TensorDataset(torch.tensor(embeddings, dtype=torch.float32), labels), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = np.inf
    epochs_without_improvement = 0

    while epochs_without_improvement < patience:
        total_loss = 0
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            logits = classifier(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Avg Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epochs without improvement: {epochs_without_improvement}")

    print("Early stopping triggered!")


''' Load data and combine both files into one dataframe'''
def load_and_combine_data(annotated_jsonl_path, initial_labeled_jsonl_path):
    summary_list = []
    sentiment_list = []

    with open(annotated_jsonl_path, "r", encoding="utf-8") as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                text = item.get("text", "").strip()
                label_raw = item.get("label", None)

                if label_raw is None:
                    continue

                label = int(label_raw[0]) if isinstance(label_raw, list) else int(label_raw)
                summary_list.append(text)
                sentiment_list.append(label)

            except Exception:
                continue

    with open(initial_labeled_jsonl_path, "r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                text = item.get("summary", "").strip()
                label = int(item.get("sentiment"))
                summary_list.append(text)
                sentiment_list.append(label)
            except Exception:
                continue

    df_combined = pd.DataFrame({"summary": summary_list, "sentiment": sentiment_list})
    return df_combined

if __name__ == "__main__":
    annotated_jsonl_path = "/teamspace/studios/this_studio/Final_pipeline/manually_annotated/iter_07.jsonl"
    initial_labeled_jsonl_path = "/teamspace/studios/this_studio/Final_pipeline/updated_train.jsonl"

    df_combined = load_and_combine_data(annotated_jsonl_path, initial_labeled_jsonl_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BERT_BiLSTM_Attention().to(device)
    classifier = SentimentClassifier().to(device)

   
    train_triplet_encoder(encoder, df_combined, tokenizer)
    train_classifier(encoder, classifier, df_combined, tokenizer)
    torch.save(classifier.state_dict(), "/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/classifier/classifier_model5.pt")
    torch.save(encoder.state_dict(), "/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/encoder/encoder_model5.pt")
    print("Fine-tuned models saved!")
