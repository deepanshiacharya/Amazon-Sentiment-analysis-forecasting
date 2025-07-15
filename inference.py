import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from model import BERT_BiLSTM_Attention, SentimentClassifier
from utils import extract_embeddings

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(path):
    df = pd.read_parquet(path)
    if 'text' not in df.columns:
        raise ValueError(f"'text' column is required, found: {df.columns.tolist()}")
    df['summary'] = df['text'].astype(str).str.strip()
    return df

def calculate_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-9), axis=1)

def main():
    input_path = "/teamspace/studios/this_studio/Final_pipeline/remaining/remaining.parquet"
    encoder_path = "/teamspace/studios/this_studio/final_triplet_encoder.pt"
    classifier_path = "/teamspace/studios/this_studio/Final_pipeline/models/Classifer/fine_tuned_classifier.pt"
    output_parquet = "/teamspace/studios/this_studio/Final_pipeline/inferences/inference_01.parquet"

    df = load_data(input_path)
    texts = df['summary'].tolist()
    rids = df.get('rid').tolist() if 'rid' in df.columns else [f"index_{i}" for i in range(len(texts))]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BERT_BiLSTM_Attention().to(device)
    classifier = SentimentClassifier().to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    embeddings = extract_embeddings(texts, encoder, tokenizer, device=device)
    with torch.no_grad():
        logits = classifier(torch.tensor(embeddings, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    conf_scores = np.max(probs, axis=1)
    entropy_scores = calculate_entropy(probs)

    label_map_numeric = {0: -1, 1: 0, 2: 1}
    label_map_word = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_labels_numeric = [label_map_numeric[p] for p in preds]
    predicted_labels_word = [label_map_word[p] for p in preds]

    result_df = pd.DataFrame({
        "rid": rids,
        "text": texts,
        "label": [[str(label)] for label in predicted_labels_numeric],
        "sentiment_word": predicted_labels_word,
        "negative_prob": probs[:, 0],
        "neutral_prob": probs[:, 1],
        "positive_prob": probs[:, 2],
        "confidence_score": conf_scores,
        "entropy": entropy_scores
    })

    result_df.to_parquet(output_parquet, index=False)
    print(f"Saved predictions and summary to {output_parquet}")

if __name__ == "__main__":
    main()
