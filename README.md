# SR-Automation

Pipeline para automação de revisão sistemática de literatura usando LLMs, desenvolvido como parte de dissertação de Mestrado Profissional em Engenharia de Software (C.E.S.A.R).

O sistema automatiza as etapas de triagem, extração de dados e sumarização de artigos acadêmicos, utilizando Llama 3.3 70B via API (Groq + Together AI como fallback).

## Pré-requisitos

- Python 3.11+
- Conta gratuita no [Groq](https://console.groq.com/) (provider primário)
- Conta no [Together AI](https://www.together.ai/) (provider de fallback)
- (Opcional) Chave da [Semantic Scholar API](https://www.semanticscholar.org/product/api)

## Setup

### 1. Clonar e criar ambiente virtual

```bash
git clone <url-do-repositório>
cd automation-sistematic-review
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

Principais dependências:
- `requests` — chamadas HTTP às APIs de LLM e fontes acadêmicas
- `chromadb` + `sentence-transformers` — embeddings e busca semântica
- `scikit-learn` — métricas (Cohen's Kappa, confusion matrix)
- `pandas` — manipulação de dados
- `pyyaml` + `python-dotenv` — configuração
- `tqdm` — progress bars
- `matplotlib` + `seaborn` — gráficos e figuras

### 3. Configurar API keys

```bash
cp .env.example .env
```

Editar `.env` com suas chaves:

```
GROQ_API_KEY=gsk_...          # Obrigatória — https://console.groq.com/keys
TOGETHER_API_KEY=...           # Obrigatória — https://api.together.ai/settings/api-keys
S2_API_KEY=...                 # Opcional — https://www.semanticscholar.org/product/api
OPENALEX_EMAIL=seu@email.com   # Opcional — melhora rate limits da OpenAlex
```

**Sobre os providers**:

| Provider | Papel | Modelo | Tier gratuito |
|---|---|---|---|
| Groq | Primário | `llama-3.3-70b-versatile` | 1.000 req/dia, 30 req/min |
| Together AI | Fallback | `Meta-Llama-3.1-70B-Instruct-Turbo` | Pay-as-you-go (~$0.88/M tokens) |

O pipeline tenta Groq primeiro. Se receber rate limit (429) ou timeout, faz fallback automático para Together AI. Após 3 rate limits consecutivos do Groq, pula diretamente para Together AI.

## Uso

### Pipeline completo

```bash
python main.py
```

### Passos individuais (ordem recomendada)

```bash
# Fase 1: Coleta e processamento
python main.py --step corpus      # Coleta artigos de 4 fontes → data/raw/corpus.csv
python main.py --step triage      # Classifica YES/NO → outputs/triage_results.jsonl
python main.py --step extract     # Extrai 5 campos dos incluídos → outputs/extraction_results.jsonl
python main.py --step summarize   # Gera TL;DR 3 frases → outputs/summaries.jsonl

# Fase 2: Amostragem para verificação manual
python main.py --step hallcheck   # Amostra claims para revisão → outputs/hallucination_sample.csv
python main.py --step validate    # Amostra campos extraídos → outputs/extraction_validation.csv

# Fase 3: Avaliação (requer tarefas manuais concluídas)
python main.py --step metrics     # Calcula métricas vs gold standard → outputs/metrics.json
python main.py --step crossval    # 3 runs com variações de prompt → outputs/cross_validation.json
python main.py --step report      # Gera tabelas LaTeX + figuras → outputs/latex_tables.tex
```

### Dependências entre passos

```
corpus → triage → extract → hallcheck
                → summarize → hallcheck
                → metrics (requer gold_standard.csv preenchido)
         crossval (independente, usa corpus)
         report (requer metrics + crossval)
```

## Configuração

Toda configuração está em `config.yaml`:

```yaml
model:
  provider: "groq"
  model_name: "llama-3.3-70b-versatile"       # Modelo principal
  fallback_provider: "together"
  fallback_model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

inference:
  triage:
    temperature: 0.1    # Baixa — decisão binária YES/NO
    top_p: 0.9
    max_tokens: 100
  extraction:
    temperature: 0.0    # Determinístico — extração factual
    top_p: 1.0
    max_tokens: 1024
  summarization:
    temperature: 0.3    # Leve criatividade para redação
    top_p: 0.95
    max_tokens: 512

corpus:
  query: '"intelligent tutoring system" AND ("large language model" OR ...)'
  year_range: [2022, 2024]
  target_size: [80, 120]              # Range alvo de artigos
  dedup_threshold: 0.9                # Similaridade de título para dedup
  sources: ["semantic_scholar", "openalex", "arxiv", "eric"]

cross_validation:
  num_runs: 3
  expected_kappa: 0.8
  force_together: true    # Usa Together AI direto (evita rate limit do Groq)

evaluation:
  recall_target: 0.85                 # Meta mínima de recall
  manual_baseline_minutes_per_article: 5
```

## Estrutura do Projeto

```
automation-sistematic-review/
├── config.yaml                    # Configuração central
├── .env                           # API keys (não versionado)
├── .env.example                   # Template de API keys
├── requirements.txt               # Dependências Python
├── main.py                        # Orquestrador CLI (argparse)
│
├── prompts/                       # Templates de prompt (editáveis)
│   ├── triage.txt                 # Prompt de triagem (YES/NO + justificativa)
│   ├── extraction.txt             # Prompt de extração (5 campos JSON)
│   └── summarization.txt          # Prompt de sumarização (TL;DR 3 frases)
│
├── src/
│   ├── __init__.py
│   ├── audit_logger.py            # Logger JSONL — registra toda chamada LLM
│   ├── utils.py                   # Config, prompts, call_llm() com cache/fallback
│   ├── corpus_builder.py          # Coleta de 4 fontes acadêmicas
│   ├── retrieval.py               # Embeddings (all-MiniLM-L6-v2) + ChromaDB + re-ranking
│   ├── triage.py                  # Classificação zero-shot YES/NO
│   ├── extraction.py              # Extração de 5 campos estruturados (JSON)
│   ├── summarization.py           # TL;DR em 3 frases
│   ├── metrics.py                 # Recall, Precision, F1, Specificity, WR
│   ├── cross_validation.py        # 3 runs com variações de prompt (Cohen's Kappa)
│   ├── hallucination_check.py     # Amostragem de claims para verificação manual
│   ├── extraction_validator.py    # Spot-check de extração
│   └── report_generator.py        # Tabelas LaTeX + figuras (matplotlib/seaborn)
│
├── data/
│   ├── raw/corpus.csv             # Corpus coletado (gerado pelo pipeline)
│   ├── gold_standard.csv          # Labels humanos (preenchido manualmente)
│   └── gold_standard_template.csv # Template com headers
│
├── outputs/                       # Todos os resultados (gerados)
│   ├── audit_log.jsonl            # Log de auditoria completo
│   ├── triage_results.jsonl       # Decisões de triagem
│   ├── extraction_results.jsonl   # Dados extraídos (JSON por artigo)
│   ├── summaries.jsonl            # Resumos TL;DR
│   ├── hallucination_sample.csv   # Amostra para verificação de alucinação
│   ├── extraction_validation.csv  # Amostra para validação de extração
│   ├── metrics.json               # Métricas vs gold standard
│   ├── cross_validation.json      # Resultados da cross-validation
│   ├── latex_tables.tex           # Tabelas prontas para LaTeX
│   ├── figures/                   # Gráficos (PNG 300 DPI)
│   └── .llm_cache/               # Cache de respostas LLM
│
├── notebooks/
│   └── analysis.ipynb             # Análises e gráficos para dissertação
│
└── docs/                          # Documentação do processo
    ├── 01_passo_a_passo.md        # Histórico de decisões e implementação
    ├── 02_tempo_custo.md          # Levantamento de tempo e custo por etapa
    ├── 03_otimizacao_requests.md  # Análise de otimização de requests
    └── 04_rascunho_resultados.md  # Rascunho do Capítulo 4 da dissertação
```

## Como Funciona o Pipeline

### 1. Coleta de Corpus (`corpus_builder.py`)

Consulta 4 APIs acadêmicas:

| Fonte | API | Autenticação |
|---|---|---|
| Semantic Scholar | `api.semanticscholar.org/graph/v1/paper/search` | Opcional (S2_API_KEY) |
| OpenAlex | `api.openalex.org/works` | Não necessária |
| arXiv | `export.arxiv.org/api/query` | Não necessária |
| ERIC | `api.ies.ed.gov/eric/` | Não necessária |

Após coleta: deduplicação por DOI + similaridade de título (threshold 0.9), filtro de abstracts válidos (>= 50 caracteres), truncamento proporcional por fonte para manter o corpus dentro do range alvo [80, 120].

### 2. Triagem (`triage.py`)

Para cada artigo, envia título + abstract ao LLM com os critérios de inclusão. O modelo responde YES ou NO com justificativa em uma frase. Parsing robusto para variações como "YES.", "Yes,", "YES -", etc.

Critérios de inclusão (configuráveis em `config.yaml`):
- Apresenta um ITS ou ferramenta educacional
- Usa um LLM como componente central
- Reporta avaliação empírica ou prova de conceito

### 3. Extração (`extraction.py`)

Para artigos incluídos (YES), extrai 5 campos em formato JSON:
- `study_objective` — objetivo principal do estudo
- `methodology` — métodos e técnicas utilizados
- `main_results` — principais achados
- `conclusions_limitations` — conclusões e limitações
- `sample_data` — dados populacionais/amostrais

### 4. Sumarização (`summarization.py`)

Gera resumo TL;DR em exatamente 3 frases:
1. Problema abordado
2. Solução proposta
3. Principais achados

### 5. Avaliação (`metrics.py`)

Compara decisões da triagem com gold standard humano:
- Recall, Precision, Specificity, F1-Score (Eq. 3.1-3.4 da dissertação)
- Workload Reduction (Eq. 3.6)
- Análise de falsos negativos

### 6. Cross-Validation (`cross_validation.py`)

3 runs da triagem com variações de prompt:
- Run 1: Prompt original
- Run 2: Critérios reescritos com sinônimos
- Run 3: Ordem invertida (abstract antes do título)

Calcula Cohen's Kappa entre cada par de runs. Meta: kappa >= 0.8.

## Tarefas Manuais Necessárias

O pipeline gera templates que devem ser preenchidos manualmente antes das etapas de avaliação:

### Gold Standard (`data/gold_standard.csv`)

Dois revisores classificam independentemente cada artigo do corpus:

```csv
id,title,reviewer_a,reviewer_b,consensus,justification
oalex_W4390414972,Using LLMs to bring...,INCLUDE,INCLUDE,INCLUDE,Apresenta ferramenta com LLM e avaliacao empirica
```

- `reviewer_a` / `reviewer_b`: INCLUDE ou EXCLUDE
- `consensus`: decisao final apos discussao de divergencias
- Meta: Cohen's Kappa entre revisores >= 0.7

### Hallucination Sample (`outputs/hallucination_sample.csv`)

Preencher coluna `classification` para cada claim:
- `GROUNDED` — afirmacao presente no abstract original
- `INFERRED` — inferencia razoavel mas nao explicita
- `HALLUCINATED` — informacao inventada pelo modelo

### Extraction Validation (`outputs/extraction_validation.csv`)

Preencher coluna `error_type` para cada campo:
- `CORRECT` — extracao precisa
- `HALLUCINATION` — informacao inventada
- `OMISSION` — campo marcado "NOT MENTIONED" mas a informacao esta no abstract
- `IMPRECISION` — parcialmente correto

## Mecanismos de Resiliencia

### Fallback automatico de providers

```
Groq (primario) --[429/5xx/timeout]--> retry 1x apos 60s --> Together AI (fallback)
                --[3 falhas consecutivas]--> skip Groq --> Together AI direto
```

### Cache de respostas

Respostas do LLM sao salvas em `outputs/.llm_cache/` por hash SHA-256 do prompt. Re-execucoes do mesmo pipeline sao instantaneas (0 chamadas API). O cache e desabilitado automaticamente na cross-validation (cada run precisa de respostas independentes).

### Rate limiting

- Sleep de 1,5s entre chamadas Groq (free tier: 30 req/min)
- Backoff exponencial nas APIs de coleta (Semantic Scholar, arXiv, etc.)
- Cross-validation usa Together AI diretamente (`force_together: true`) para evitar esgotar quota diaria do Groq

### Interrupcao graceful

`Ctrl+C` durante qualquer etapa salva resultados parciais em JSONL antes de encerrar.

## Auditabilidade

Toda chamada ao LLM e registrada em `outputs/audit_log.jsonl` com:

```json
{
  "timestamp": "2026-02-22T21:42:46.171452+00:00",
  "module": "triage",
  "article_id": "oalex_W4366783381",
  "decision": "NO, because the article is a systematic review...",
  "confidence": 1.0,
  "prompt_hash": "a1b2c3d4e5f6g7h8",
  "provider": "groq",
  "tokens_in": 425,
  "tokens_out": 42,
  "latency_ms": 761.0,
  "raw_response": "NO, because the article is a systematic review..."
}
```

## Reproducao Completa

Para reproduzir os resultados a partir do zero:

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env com suas API keys

# 2. Pipeline automatizado
python main.py --step corpus       # ~3 min — coleta 80-120 artigos
python main.py --step triage       # ~5 min — classifica YES/NO
python main.py --step extract      # ~3 min — extrai 5 campos JSON
python main.py --step summarize    # ~4 min — gera TL;DR

# 3. Amostragem para revisao manual
python main.py --step hallcheck    # gera hallucination_sample.csv
python main.py --step validate     # gera extraction_validation.csv

# 4. [MANUAL] Preencher:
#    - data/gold_standard.csv (2 revisores x 120 artigos)
#    - outputs/hallucination_sample.csv (coluna classification)
#    - outputs/extraction_validation.csv (coluna error_type)

# 5. Avaliacao
python main.py --step metrics      # requer gold_standard.csv
python main.py --step crossval     # ~17 min (via Together AI)
python main.py --step report       # gera tabelas LaTeX + figuras PNG

# 6. Analise
jupyter notebook notebooks/analysis.ipynb
```

**Tempo total estimado** (excluindo tarefas manuais): ~35 minutos
**Custo total estimado**: ~$0.20 (Together AI para cross-validation + fallbacks)

## Notas sobre Reproducao

- Os resultados da coleta de corpus podem variar entre execucoes, pois as APIs retornam resultados em tempo real. Artigos novos podem ser adicionados ou removidos das bases.
- Para reproducao exata, usar o `data/raw/corpus.csv` gerado na primeira execucao e pular `--step corpus`.
- O cache em `outputs/.llm_cache/` garante que re-execucoes com o mesmo corpus e prompts produzam resultados identicos.
- A cross-validation desabilita o cache propositalmente para gerar respostas independentes em cada run.
- Os modelos Llama podem ser atualizados ou removidos dos providers. Verificar modelos disponiveis em [Groq Models](https://console.groq.com/docs/models) e [Together Models](https://docs.together.ai/docs/inference-models).

## Referencias

- **Dissertacao**: Capitulo 3 — Metodologia (secoes 3.1 a 3.7)
- **Modelo**: [Llama 3.3 70B](https://ai.meta.com/blog/llama-3/) via Groq e Together AI
- **Embeddings**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensoes)
- **Re-ranking**: [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
