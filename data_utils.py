import os
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from transformers import DataCollatorWithPadding
from config import BASE_PATH

def CoT_prompt(row):
    base_prompt = """You are an expert neurologist specializing in dementia diagnosis. Analyze the participant's description of the kitchen scene image for signs of cognitive impairment.

Task: Evaluate the participant's ability to describe the image, focusing on their attention to detail, spatial awareness, and cognitive coherence.

Analysis Framework:
1. **Visual Recognition Analysis**
   - Identification of key elements (e.g., mother, children, kitchen objects)
   - Recognition of activities and spatial relationships
   - Awareness of safety hazards in the scene

2. **Descriptive Language Assessment**
   - Completeness and sequencing of the description
   - Vocabulary richness and narrative coherence
   - Logical flow and organization of thoughts

3. **Detail Observation**
   - Mention of specific objects (e.g., ladder, sink, curtain)
   - Awareness of characters' actions and interactions
   - Depth of understanding and missing critical elements

4. **Memory and Attention**
   - Consistency in description without forgetting or repeating elements
   - Ability to maintain focus on relevant details
   - Structured and organized thought process

Input: """
    
    key_elements = ["stool", "sink", "dish", "wash", "jar", "cookie", 
                   "child", "mother", "window", "cabinet", "kitchen", "water"]

    text = row['text']
    key_element_count = sum(1 for element in key_elements if element in text.lower())

    if key_element_count >= len(key_elements) * 0.75:
        category = "Highly Detailed (Non-AD Likely)"
    elif key_element_count >= len(key_elements) / 2:
        category = "Moderately Detailed (Borderline Case)"
    else:
        category = "Low Detail (AD Candidate)"

    index = row.name if isinstance(row.name, (int, np.integer)) else 0
    
    if index <= 54:
        patient_group = "Speech data from non-dementia patients"
    elif index <= 108:
        patient_group = "Speech data from dementia patients"
    else:
        patient_group = "Speech data from unclassified group"

    return (
        f"{base_prompt}\n"
        f"Patient Group: {patient_group}\n"
        f"Category: {category}\n"
        f"Text Analysis:\n{text}"
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
