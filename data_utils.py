import os
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from transformers import DataCollatorWithPadding
from config import BASE_PATH


def CoT_prompt(row):
    base_prompt = """You are an expert neurologist specializing in dementia diagnosis.

    Question:
    Based on the participant's verbal description of the kitchen scene, can you assess the likelihood of cognitive impairment by evaluating their attention to detail, spatial awareness, and cognitive coherence?
    
    Analysis Steps:
    1. Visual Recognition Analysis
       - Did the participant identify key elements (e.g., mother, children, kitchen objects)?
       - Did they recognize activities and spatial relationships?
       - Did they notice safety hazards in the scene?
    2. Descriptive Language Assessment
       - Is the description complete and logically sequenced?
       - Is vocabulary rich and is the narrative coherent?
       - Is there a logical flow and organization of thoughts?
    3. Detail Observation
       - Are specific objects (e.g., ladder, sink, curtain) mentioned?
       - Is there awareness of characters' actions and interactions?
       - Is there depth of understanding or are critical elements missing?
    4. Memory and Attention
       - Is the description consistent, without forgetting or repeating elements?
       - Does the participant maintain focus on relevant details?
       - Is the thought process structured and organized?
    
    Participant's Description:
    """

    cue_elements = ["stool", "sink", "dish", "wash", "jar", "cookie", 
                    "child", "mother", "window", "cabinet", "kitchen", "water"]

    text = row['text'].lower()

    cue_element_count = sum(1 for element in cue_elements if element in text)

    # categorization
    if cue_element_count >= len(cue_elements) * 0.75:
        category = "Highly Detailed (Non-AD Likely)"
    elif cue_element_count >= len(cue_elements) / 2:
        category = "Moderately Detailed (Borderline Case)"
    else:
        category = "Low Detail (AD Candidate)"

    return (
        f"{base_prompt}"
        f"{text}\n\n"
        f"Detail Category: {category}\n\n"
        f"Answer:\n"
    )


def load_and_process_data():
    # data load
    metadata = pd.read_csv(os.path.join(BASE_PATH, 'metadata.csv'))
    
    # add CoT prompt
    metadata = metadata.reset_index(drop=True)
    metadata['text'] = metadata.apply(
        lambda row: CoT_prompt(row) if pd.notnull(row['text']) else None, 
        axis=1
    )
    
    train_data = metadata[metadata['ID'].str.contains('train')]
    test_data = metadata[metadata['ID'].str.contains('test')]
    
    train_data = train_data[['text', 'AD']].dropna()
    test_data = test_data[['text', 'AD']].dropna()
    
    train_data['labels'] = train_data['AD'].astype(int)
    test_data['labels'] = test_data['AD'].astype(int)
    
    train_data = train_data.drop('AD', axis=1)
    test_data = test_data.drop('AD', axis=1)
    
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    return train_dataset, test_dataset


def tokenize_with_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors=None,
        return_attention_mask=True
    )
    tokenized["labels"] = [int(label) for label in examples["labels"]]
    return tokenized


class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    def __call__(self, features):
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)
        }
        return batch
