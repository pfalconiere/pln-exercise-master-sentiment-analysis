"""
Extração de features: TF-IDF e embeddings (mean pooling).
"""

from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from src.config import (
    EMBEDDING_DIM,
    EMBEDDINGS_DIR,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)


# ──────────────────────────────────────────────
# TF-IDF
# ──────────────────────────────────────────────

def construir_tfidf(
    textos_treino: list[str],
    textos_val: list[str],
    textos_teste: list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
) -> tuple:
    """
    Constrói TF-IDF: fit no treino, transform em todos os splits.
    Retorna (vectorizer, X_train, X_val, X_test).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(textos_treino)
    X_val = vectorizer.transform(textos_val)
    X_test = vectorizer.transform(textos_teste)

    print(f"TF-IDF: vocabulário={len(vectorizer.vocabulary_):,}, "
          f"shape treino={X_train.shape}")

    return vectorizer, X_train, X_val, X_test


# ──────────────────────────────────────────────
# Embeddings (FastText)
# ──────────────────────────────────────────────

def carregar_embeddings(caminho: str | Path | None = None) -> KeyedVectors:
    """
    Carrega embeddings pré-treinados (NILC FastText ou Facebook).
    Se caminho não fornecido, tenta localizar em EMBEDDINGS_DIR.
    """
    if caminho is None:
        # Tentar encontrar automaticamente
        possiveis = (
            list(EMBEDDINGS_DIR.glob("*.kv"))
            + list(EMBEDDINGS_DIR.glob("*.bin"))
            + list(EMBEDDINGS_DIR.glob("*.txt"))
        )
        possiveis = [p for p in possiveis if p.stat().st_size > 0]
        if not possiveis:
            raise FileNotFoundError(
                f"Nenhum arquivo de embeddings encontrado em {EMBEDDINGS_DIR}. "
                "Baixe NILC FastText (cbow_s300.txt) ou Facebook cc.pt.300.bin "
                "e coloque em data/embeddings/"
            )
        caminho = possiveis[0]

    caminho = Path(caminho)
    print(f"Carregando embeddings de {caminho.name}...")

    if caminho.suffix == ".kv":
        wv = KeyedVectors.load(str(caminho))
    elif caminho.suffix == ".bin":
        wv = KeyedVectors.load_word2vec_format(str(caminho), binary=True)
    else:
        wv = KeyedVectors.load_word2vec_format(str(caminho), binary=False)

    print(f"Embeddings carregados: {len(wv):,} palavras, dim={wv.vector_size}")
    return wv


def extrair_embeddings_spacy(
    textos: list[str],
    model_name: str = "pt_core_news_sm",
) -> np.ndarray:
    """
    Fallback: extrai embeddings usando spaCy (96d para pt_core_news_sm).
    Útil quando embeddings FastText não estão disponíveis.
    """
    import spacy
    nlp = spacy.load(model_name)

    embeddings = []
    for texto in tqdm(textos, desc="spaCy Embeddings"):
        doc = nlp(texto)
        embeddings.append(doc.vector)

    result = np.array(embeddings)
    n_zeros = np.sum(np.all(result == 0, axis=1))
    if n_zeros > 0:
        print(f"Aviso: {n_zeros} textos sem embedding (vetor zero)")
    print(f"Embeddings spaCy: shape={result.shape}")
    return result


def texto_para_embedding(
    texto: str,
    wv: KeyedVectors,
    dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """
    Mean pooling: média dos vetores de palavras do texto.
    Retorna vetor zero se nenhuma palavra encontrada.
    """
    palavras = texto.split()
    vetores = [wv[p] for p in palavras if p in wv]

    if not vetores:
        return np.zeros(dim)

    return np.mean(vetores, axis=0)


def textos_para_embeddings(
    textos: list[str],
    wv: KeyedVectors,
    dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Aplica mean pooling a uma lista de textos."""
    embeddings = np.array([
        texto_para_embedding(t, wv, dim) for t in tqdm(textos, desc="Embeddings")
    ])
    n_zeros = np.sum(np.all(embeddings == 0, axis=1))
    if n_zeros > 0:
        print(f"Aviso: {n_zeros} textos sem embedding (vetor zero)")
    return embeddings
