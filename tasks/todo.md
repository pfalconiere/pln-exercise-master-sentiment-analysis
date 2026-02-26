# Análise de Sentimentos — Tracking

## Fase 0: Scaffolding
- [x] Inicializar git repo
- [x] Criar `.gitignore`, `.env.example`
- [x] Criar `pyproject.toml`
- [x] Criar estrutura de diretórios
- [x] Criar `src/config.py` e `src/utils.py`
- [x] `poetry install` + downloads NLP

## Fase 1: Dados e Exploração
- [x] `src/data_loader.py`
- [x] `notebooks/01_exploracao_dados.ipynb`

## Fase 2: Preprocessamento
- [x] `src/preprocessing.py`
- [x] `notebooks/02_preprocessamento.ipynb`

## Fase 3: Feature Extraction + Evaluation
- [x] `src/feature_extraction.py`
- [x] `src/evaluation.py`

## Fase 4: SVM + Bag of Words
- [x] `notebooks/03_svm_bow.ipynb`

## Fase 5: SVM + Embeddings
- [x] `notebooks/04_svm_embeddings.ipynb`

## Fase 6: BERT Fine-tuning
- [x] `notebooks/05_bert.ipynb`

## Fase 7: ICL com Claude (Bonus)
- [x] `notebooks/06_icl_claude.ipynb`

## Fase 8: Comparação Final
- [x] `notebooks/07_comparacao_final.ipynb`
- [x] `README.md`

---

## Resultados

| Modelo | Acurácia | F1 (weighted) | F1 Neg | F1 Neu | F1 Pos |
|--------|----------|---------------|--------|--------|--------|
| BERTimbau | 0.8538 | 0.8464 | 0.8872 | 0.4191 | 0.9152 |
| ICL GPT-4o-mini | 0.8380 | 0.8454 | 0.8848 | 0.4571 | 0.9069 |
| SVM+BoW | 0.8170 | 0.8110 | 0.8461 | 0.3588 | 0.8873 |
| SVM+Embeddings | 0.6892 | 0.6919 | 0.6758 | 0.2382 | 0.7903 |

**Melhor modelo**: BERTimbau (F1=0.846)
**Classe mais difícil**: Neutro (F1 máximo 0.457 com ICL)
