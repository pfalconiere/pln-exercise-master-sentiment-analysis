"""
Carregamento, mapeamento e split do dataset B2W-Reviews01.
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    HF_DATASET,
    PROCESSED_DIR,
    RATING_COLUMN,
    RATING_TO_SENTIMENT,
    SEED,
    SENTIMENT_TO_ID,
    TEST_SIZE,
    TEXT_COLUMN,
    TRAIN_SIZE,
    VAL_SIZE,
)


def carregar_dataset() -> pd.DataFrame:
    """Carrega o B2W-Reviews01 do HuggingFace e retorna DataFrame."""
    ds = load_dataset(HF_DATASET, split="train")
    df = ds.to_pandas()
    print(f"Dataset carregado: {len(df):,} avaliações")
    return df


def filtrar_e_mapear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra nulos em review_text/overall_rating e mapeia ratings para sentimento.
    """
    antes = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()
    df = df[df[TEXT_COLUMN].str.strip().astype(bool)].copy()
    depois = len(df)
    print(f"Filtrados {antes - depois:,} registros nulos/vazios → {depois:,} restantes")

    # Mapear rating -> sentimento
    df["sentimento"] = df[RATING_COLUMN].astype(int).map(RATING_TO_SENTIMENT)
    df["label"] = df["sentimento"].map(SENTIMENT_TO_ID)

    # Remover qualquer rating fora de 1-5
    df = df.dropna(subset=["sentimento"]).copy()
    print(f"Distribuição de sentimento:\n{df['sentimento'].value_counts().to_string()}")
    return df


def criar_splits(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado 70/15/15."""
    # Primeiro: separa treino do resto (val+teste)
    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )

    # Segundo: separa val e teste do resto
    relative_test_size = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_size,
        random_state=seed,
        stratify=df_temp["label"],
    )

    print(f"Splits: treino={len(df_train):,} | val={len(df_val):,} | teste={len(df_test):,}")
    return df_train, df_val, df_test


def salvar_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Salva splits em parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    df_val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    df_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Splits salvos em {PROCESSED_DIR}")


def carregar_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega splits do parquet."""
    df_train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    df_val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    df_test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    print(f"Splits carregados: treino={len(df_train):,} | val={len(df_val):,} | teste={len(df_test):,}")
    return df_train, df_val, df_test


def pipeline_completo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa o pipeline completo: download → filtro → split → salvar."""
    df = carregar_dataset()
    df = filtrar_e_mapear(df)
    df_train, df_val, df_test = criar_splits(df)
    salvar_splits(df_train, df_val, df_test)
    return df_train, df_val, df_test
