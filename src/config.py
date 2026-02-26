"""
Configurações centrais do projeto de Análise de Sentimentos.
Constantes, paths, hiperparâmetros e mapeamentos.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

# ──────────────────────────────────────────────
# Reprodutibilidade
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
HF_DATASET = "ruanchaves/b2w-reviews01"
TEXT_COLUMN = "review_text"
RATING_COLUMN = "overall_rating"

# ──────────────────────────────────────────────
# Mapeamento de sentimento (ternário)
# ──────────────────────────────────────────────
RATING_TO_SENTIMENT = {
    1: "negativo",
    2: "negativo",
    3: "neutro",
    4: "positivo",
    5: "positivo",
}

SENTIMENT_LABELS = ["negativo", "neutro", "positivo"]
SENTIMENT_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}

# ──────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ──────────────────────────────────────────────
# Preprocessamento
# ──────────────────────────────────────────────
SPACY_MODEL = "pt_core_news_sm"

# ──────────────────────────────────────────────
# TF-IDF
# ──────────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
EMBEDDING_DIM = 300
NILC_FASTTEXT_URL = "http://143.107.183.175:22980/download.php?file=embeddings/fasttext/cbow_s300.zip"
FACEBOOK_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz"

# ──────────────────────────────────────────────
# SVM
# ──────────────────────────────────────────────
SVM_C_VALUES = [0.1, 1, 10]
SVM_RBF_C_VALUES = [0.1, 1, 10, 100]
SVM_GAMMA_VALUES = ["scale", "auto"]

# ──────────────────────────────────────────────
# BERT
# ──────────────────────────────────────────────
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BERT_MAX_LENGTH = 256
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_BATCH_SIZE = 16
BERT_WEIGHT_DECAY = 0.01

# ──────────────────────────────────────────────
# ICL (Claude)
# ──────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
ICL_SUBSET_SIZE = 500
ICL_FEW_SHOT_PER_CLASS = 2
