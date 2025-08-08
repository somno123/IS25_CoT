import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# data path
BASE_PATH = '/data'

# seed
SEED = 42

# model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LABELS = 2
MAX_SEQ_LENGTH = 512

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# train
OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_rate = 1e-4
WEIGHT_DECAY = 0.001
MAX_STEPS = 1000
WARMUP_RATIO = 0.1
save_steps=25
LOGGING_STEPS = 25
FP16 = False
BF16 = True
MAX_GRAD_NORM = 0.3
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "linear"
GROUP_BY_LENGTH = True
PACKING = False
load_best_model_at_end=True  
metric_for_best_model="accuracy"
save_total_limit=2  

