import os
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from transformers import DataCollatorWithPadding
from config import BASE_PATH


def CoT_prompt(row):
    import numpy as np

    cue_elements = ["adult", "child", "cookie jar", "dish", "sink", "stool", "reach", "stand", "wash"]

    cue_elements_str = ", ".join(cue_elements)

    base_prompt = f"""You are an expert neurologist specializing in dementia diagnosis.
Analyze the participant's description of the kitchen scene based solely on the presence and relevance of critical cues.

Task: Step-by-step, evaluate the presence of each cues and infer the participant's cognitive detail level.

Cue elements to check: {cue_elements_str}
*If a different but semantically similar word is used (e.g., "mother" ≈ "adult", "cabinet" ≈ "cupboard"), count it as the corresponding cue element.*

Step 1. Identify the cues in the person's utterance. Check if each cue element is mentioned. List the present cues.
Step 2. Evaluate cue relevance from the identified cues. Count the number of cues present. Calculate the percentage of cues observed out of the total check cues (the {len(cue_elements)} cue elements listed above).
Step 3. Diagnose based on the evaluation result. 
(1) high detail: Includes at least 75% of the cues → likely non-AD
(2) moderate detail: Includes 50% or more but less than 75% of the cues → borderline case
(3) low detail: Includes less than 50% of the cues → AD candidate

Example 1:
Input: "The adult is by the sink, the child is on the stool reaching for the cookie jar."
Step 1. Cues found = ["adult", "sink", "child", "stool", "reach", "cookie jar"]
Step 2. 6/{len(cue_elements)} cues present (=67%)
Step 3. moderate detail (borderline case)

Example 2:  
Input: "There are some people there."
Step 1. cues found = []
Step 2. 0/{len(cue_elements)} cues present (=0%)
Step 3. low detail (AD candidate)
"""

    text = row['text']
    
    return f"{base_prompt}\nParticipant's Description:\n{text}"


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

