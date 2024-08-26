import torch
from transformers import AutoTokenizer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 也可以加载本地模型："bert-base-chinese" 换成本地模型路径
tokenizer = AutoTokenizer.from_pretrained(r"D:\模型\bart-base-chinese")

MODEL_DIR = './data/output/models/'
LABEL_PATH = './data/input/class.txt'

TEXT_LEN = 30

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 10

EPOCH = 100
LR = 1e-3
