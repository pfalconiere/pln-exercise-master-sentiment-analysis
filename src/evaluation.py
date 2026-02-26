"""
Métricas de avaliação e visualizações para comparação de modelos.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIGURES_DIR, MODELS_DIR, SENTIMENT_LABELS


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str = "Modelo",
) -> dict:
    """
    Calcula accuracy, F1 (weighted), precision (weighted), recall (weighted).
    Imprime classification report e retorna dicionário de métricas.
    """
    metricas = {
        "modelo": nome_modelo,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }

    # F1 por classe
    f1_por_classe = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(SENTIMENT_LABELS):
        metricas[f"f1_{label}"] = f1_por_classe[i]

    print(f"\n{'='*50}")
    print(f"Resultados: {nome_modelo}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metricas['accuracy']:.4f}")
    print(f"F1 (weighted): {metricas['f1_weighted']:.4f}")
    print(f"Precision: {metricas['precision_weighted']:.4f}")
    print(f"Recall:    {metricas['recall_weighted']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=SENTIMENT_LABELS))

    return metricas


def salvar_metricas(metricas: dict, nome_arquivo: str | None = None) -> Path:
    """Salva métricas em JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    nome = nome_arquivo or f"metricas_{metricas['modelo'].lower().replace(' ', '_')}.json"
    caminho = MODELS_DIR / nome
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em {caminho}")
    return caminho


def carregar_metricas(caminho: str | Path) -> dict:
    """Carrega métricas de JSON."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def plotar_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str = "Modelo",
    salvar: bool = True,
) -> plt.Figure:
    """Plota heatmap da matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão — {nome_modelo}", fontsize=13)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()

    if salvar:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        nome_arquivo = f"cm_{nome_modelo.lower().replace(' ', '_')}.png"
        fig.savefig(FIGURES_DIR / nome_arquivo, dpi=150)

    return fig


def plotar_comparacao_modelos(
    lista_metricas: list[dict],
    metricas_plot: list[str] | None = None,
    salvar: bool = True,
) -> plt.Figure:
    """Gráfico de barras agrupado comparando modelos."""
    if metricas_plot is None:
        metricas_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

    labels_plot = {
        "accuracy": "Acurácia",
        "f1_weighted": "F1 (weighted)",
        "precision_weighted": "Precisão",
        "recall_weighted": "Recall",
    }

    modelos = [m["modelo"] for m in lista_metricas]
    n_modelos = len(modelos)
    n_metricas = len(metricas_plot)
    x = np.arange(n_modelos)
    largura = 0.8 / n_metricas

    fig, ax = plt.subplots(figsize=(12, 6))
    cores = sns.color_palette("Set2", n_metricas)

    for i, metrica in enumerate(metricas_plot):
        valores = [m.get(metrica, 0) for m in lista_metricas]
        offset = (i - n_metricas / 2 + 0.5) * largura
        bars = ax.bar(x + offset, valores, largura, label=labels_plot.get(metrica, metrica), color=cores[i])
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Comparação de Modelos", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if salvar:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "comparacao_modelos.png", dpi=150)

    return fig


def plotar_f1_por_classe(
    lista_metricas: list[dict],
    salvar: bool = True,
) -> plt.Figure:
    """Gráfico de barras agrupado com F1 por classe para cada modelo."""
    modelos = [m["modelo"] for m in lista_metricas]
    n_modelos = len(modelos)
    x = np.arange(n_modelos)
    largura = 0.8 / len(SENTIMENT_LABELS)
    cores = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(SENTIMENT_LABELS):
        valores = [m.get(f"f1_{label}", 0) for m in lista_metricas]
        offset = (i - len(SENTIMENT_LABELS) / 2 + 0.5) * largura
        bars = ax.bar(x + offset, valores, largura, label=label.capitalize(), color=cores[i])
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("F1-Score por Classe", fontsize=14)
    ax.set_ylabel("F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if salvar:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "f1_por_classe.png", dpi=150)

    return fig
