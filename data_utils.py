import os
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from transformers import DataCollatorWithPadding
from config import BASE_PATH

def CoT_prompt(row):
    base_prompt = """You are an expert neurologist specializing in dementia diagnosis. Analyze the participant's description of the kitchen scene image for signs of cognitive impairment.

    Analysis Steps:
    1. Check for mention of key cue elements (e.g., stool, sink, dish, etc.).
    2. Evaluate awareness of safety hazards (e.g., stool, water, window).
    3. Assess logical flow and narrative coherence using connecting words.
    4. Integrate all findings to determine the likelihood of dementia.
    
    Input:"""

    cue_elements = ["stool", "sink", "dish", "wash", "jar", "cookie", 
                    "child", "mother", "window", "cabinet", "kitchen", "water"]

    text = row['text'].lower()

    cue_element_count = sum(1 for element in cue_elements if element in text)

    # Detail categorization with explanation
    if cue_element_count >= len(cue_elements) * 0.75:
        category = "Highly Detailed (Unlikely AD)"
        reasoning_detail = "A higher count suggests excellent visual recognition and attention to detail, with clear awareness of the scene."
    elif cue_element_count >= len(cue_elements) / 2:
        category = "Moderately Detailed (Borderline Case)"
        reasoning_detail = "This level of detail shows moderate scene comprehension and recall, though some relevant elements are missing."
    else:
        category = "Low Detail (Possible AD)"
        reasoning_detail = "A lower count suggests limited visual recognition or memory recall, potentially indicating early signs of cognitive impairment."

    reasoning = (
        f"The participant mentioned {cue_element_count} out of {len(cue_elements)} key elements "
        f"({', '.join([e for e in cue_elements if e in text])}).\n"
        f"{reasoning_detail}\n"
    )

    return (
        f"{base_prompt}\n\n"
        f"Reasoning Process:\n{reasoning}\n\n"
        f"Final Assessment: {category}\n\n"
        f"### Participant Description ###:\n{row['text']}"
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
