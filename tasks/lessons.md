# Lições Aprendidas

_Atualizado ao longo do projeto com padrões e correções._

---

## 1. Infraestrutura e Dependências

- **Versões de pacotes importam**: `accelerate 0.34` era incompatível com `transformers 4.57` (`keep_torch_compile` error). Sempre verificar compatibilidade antes de fixar versões.
- **SDK do Anthropic**: versão 0.34 tinha bug com `proxies` no httpx. Atualizar para >=0.84 resolveu.
- **SSL em ambiente corporativo**: download do NLTK falhou por verificação SSL. Workaround: `ssl._create_unverified_context`.

## 2. Embeddings

- **Servidor NILC indisponível**: o endpoint de download dos embeddings FastText NILC retornou 0 bytes. Sempre ter um fallback.
- **spaCy como fallback**: `pt_core_news_sm` tem embeddings de 96d (vs 300d do FastText). Funciona, mas com desempenho inferior (~0.69 F1 vs estimado ~0.75+ com FastText 300d).
- **Vetores vazios**: spaCy retorna vetor `(0,)` para documentos sem tokens reconhecidos. Tratar com vetor-zero explícito para evitar `ValueError: inhomogeneous shape`.

## 3. Modelagem

- **LinearSVC vs SVC**: para datasets grandes (>90K), LinearSVC é O(n) vs O(n²) do SVC com kernel RBF. Usar subsample (10K) para GridSearch do RBF, depois retreinar no dataset completo.
- **BERT no MPS (Apple Silicon)**: fp16 não funciona bem no MPS. Usar fp32. Dataset de 93K × 3 épocas ultrapassava 2h; reduzir para 30K subset manteve F1=0.846.
- **max_length do BERT**: 128 tokens cobrem a vasta maioria dos reviews B2W (mediana ~16 palavras). Não há ganho significativo com 256.
- **Classe neutro é difícil**: F1 máximo de 0.457 (ICL). Reviews neutros são ambíguos por natureza — misturam elogios e críticas.

## 4. ICL (In-Context Learning)

- **Few-shot é surpreendentemente bom**: GPT-4o-mini com apenas 6 exemplos atingiu F1=0.845, quase igual ao BERTimbau fine-tuned.
- **Comparação justa**: ICL avaliado em subset de 500 amostras (vs 19K do teste completo). Resultados não são diretamente comparáveis — documentar isso claramente.
- **Custo-benefício**: ICL não requer treino, mas tem custo por chamada API. Ideal para prototipagem rápida.

## 5. Boas Práticas

- **Dados ordenados pelo glob**: `sorted(MODELS_DIR.glob(...))` retorna ordem alfabética. Nunca hardcodar listas que assumem uma ordem específica — usar dicionários com lookup por nome.
- **Consistência de nomes**: manter nomes de modelos idênticos em métricas, figuras e tabelas. Renomeações parciais causam confusão (ex: "Claude" vs "GPT-4o-mini").
- **Salvar métricas em JSON**: facilita consolidação no notebook final. Padronizar chaves: `modelo`, `accuracy`, `f1_weighted`, `f1_{classe}`.
