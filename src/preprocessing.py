"""
Pipeline de preprocessamento de texto para análise de sentimentos.
Limpeza, remoção de stopwords, stemming, lematização.
"""

import re
import unicodedata

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from src.config import SPACY_MODEL

# ──────────────────────────────────────────────
# Lazy loading de recursos NLP
# ──────────────────────────────────────────────
_stopwords_pt: set | None = None
_stemmer: RSLPStemmer | None = None
_nlp = None


def _get_stopwords() -> set:
    global _stopwords_pt
    if _stopwords_pt is None:
        _stopwords_pt = set(stopwords.words("portuguese"))
    return _stopwords_pt


def _get_stemmer() -> RSLPStemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = RSLPStemmer()
    return _stemmer


def _get_spacy():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    return _nlp


# ──────────────────────────────────────────────
# Funções de limpeza
# ──────────────────────────────────────────────

def limpar_texto(texto: str) -> str:
    """
    Limpeza básica de texto:
    - Lowercase
    - Normalização Unicode (acentos mantidos)
    - Remoção de URLs
    - Remoção de emails
    - Remoção de números
    - Remoção de pontuação (exceto espaços)
    - Remoção de espaços múltiplos
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^\w\sáàâãéèêíìîóòôõúùûçü]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def remover_stopwords(texto: str) -> str:
    """Remove stopwords em português (NLTK)."""
    sw = _get_stopwords()
    palavras = texto.split()
    return " ".join(p for p in palavras if p not in sw)


def aplicar_stemming(texto: str) -> str:
    """Aplica stemming RSLP (português)."""
    stemmer = _get_stemmer()
    palavras = texto.split()
    return " ".join(stemmer.stem(p) for p in palavras)


def aplicar_lematizacao(texto: str) -> str:
    """Aplica lematização com spaCy (pt_core_news_sm)."""
    nlp = _get_spacy()
    doc = nlp(texto)
    return " ".join(token.lemma_ for token in doc if not token.is_space)


# ──────────────────────────────────────────────
# Pipelines compostas
# ──────────────────────────────────────────────

def preprocessar_para_svm(texto: str) -> str:
    """
    Pipeline completa para SVM:
    limpeza → stopwords → stemming
    """
    texto = limpar_texto(texto)
    texto = remover_stopwords(texto)
    texto = aplicar_stemming(texto)
    return texto


def preprocessar_para_embeddings(texto: str) -> str:
    """
    Pipeline para embeddings (sem stemming para preservar semântica):
    limpeza → stopwords
    """
    texto = limpar_texto(texto)
    texto = remover_stopwords(texto)
    return texto
