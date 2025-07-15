import json
import logging
import torch
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_object = json.loads(line.strip())
                if 'text' in json_object:
                    label = json_object.get('label', None)
                    if label is not None and len(label) > 0:
                        if isinstance(label, list):
                            label = label[0]
                        data.append({'text': json_object['text'], 'label': label})
                    else:
                        logging.warning(f"Missing label: {line.strip()}")
                else:
                    logging.warning(f"Missing text: {line.strip()}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, line: {line.strip()}")
    return data

def extract_embeddings(texts, model, tokenizer, device, batch_size=64, max_len=128):
    model.eval()
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], truncation=True, padding='max_length',
                          max_length=max_len, return_tensors='pt')
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            embeds = model(input_ids, attention_mask).cpu().numpy()
        all_embeds.append(embeds)
    return np.vstack(all_embeds)
