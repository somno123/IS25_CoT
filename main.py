import os
import torch
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

import config
from data_utils import load_and_process_data, tokenize_with_features, CustomDataCollator
from model import setup_model_and_tokenizer
from train import setup_trainer
from evaluation import evaluate_model

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(config.SEED)
    scaler = StandardScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data load
    train_dataset, test_dataset = load_and_process_data()
    
    model, tokenizer, lora_config = setup_model_and_tokenizer()
    
    train_dataset = train_dataset.map(
        lambda examples: tokenize_with_features(examples, tokenizer), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda examples: tokenize_with_features(examples, tokenizer), 
        batched=True
    )
    
    collate_fn = CustomDataCollator(tokenizer=tokenizer)
    
    trainer = setup_trainer(
        model, tokenizer, lora_config, 
        train_dataset, test_dataset, collate_fn
    )
    
    # model train
    trainer.train()
    
    # model evaluation
    results = evaluate_model(trainer, test_dataset)
    
    return results

if __name__ == "__main__":
    main()
