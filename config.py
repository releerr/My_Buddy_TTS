import torch
import os

#path setting: 获取当前绝对路径，确保调用时路径不会出错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw','')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

LOG_DIR = os.path.join(BASE_DIR, 'logs')

# models paths
# TODO
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')
# CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints_tt')
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'pretrained')
CHINESE_ROBERTA_MODEL_DIR = os.path.join(BASE_DIR,'models','pretrained','chinese-roberta-wwm-ext')
VOC_PATH = os.path.join(BASE_DIR,'models','pretrained','nvidia','bigvgan_v2_44khz_128band_256x')

#utils paths
DATA_LOADER_PATH = os.path.join(BASE_DIR, 'utils', 'data_loader.py')
AUDIO_TOOLS_PATH = os.path.join(BASE_DIR, 'utils', 'audio_tools.py')
TEXT_TOOLS_PATH = os.path.join(BASE_DIR, 'utils', 'text_tools.py')

#output paths
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')



#check if the dir exists, if not, print a warning
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, PRETRAINED_MODELS_DIR, CHINESE_ROBERTA_MODEL_DIR, DATA_LOADER_PATH, AUDIO_TOOLS_PATH, TEXT_TOOLS_PATH, OUTPUT_DIR]:
    if not os.path.exists(path):
        print(f"[Hey,Warning]: {path} does not exist. Please create it before running the code.\n")


# Audio Standards
#N_FFT = 1024、HOP_SIZE = 256会消耗更多算力，但能提供更高的频谱分辨率，适合高质量音频生成。N_FFT = 2048、HOP_SIZE = 512是常见的设置，能在质量和效率之间取得平衡。
SAMPLE_RATE = 44100 # 22050
N_FFT = 1024
HOP_SIZE = 256
WIN_LENGTH = 1024
N_MELS = 128
MEL_BINS = 128 #80


FMIN = 0
FMAX = SAMPLE_RATE // 2
REF_DB = 20   
MAX_DB = 100      
PREEMPHASIS = 0.97 

MAX_FRAME_LEN = 5000  # 对应 SinusoidalPositionalEmbedding 的 max_len
MODEL_HIDDEN_DIM = 192 # 模型内部处理的隐藏维度

# general training hyperparameters
BATCH_SIZE = 16
EPOCHS = 100 #20
LEARNING_RATE_BERT = 2e-5
LEARNING_RATE_HEAD = 1e-4

SEED = 42
MAX_LENGTH = 128
OUTPUT_DIM = 256
NUM_SOUND_EVENTS = 10
NUM_EMOTIONS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')



