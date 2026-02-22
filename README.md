# SR-Automation

Pipeline para automação de revisão sistemática de literatura usando LLM (Llama 3.1 70B).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # preencher com as API keys
```

## Uso

```bash
python main.py                    # pipeline completo
python main.py --step corpus      # coleta de artigos
python main.py --step triage      # triagem YES/NO
python main.py --step extract     # extração de dados
python main.py --step summarize   # sumarização
python main.py --step metrics     # métricas vs gold standard
python main.py --step crossval    # validação cruzada (3 runs)
python main.py --step hallcheck   # amostra para verificação de alucinação
python main.py --step validate    # spot-check de extração
python main.py --step report      # tabelas LaTeX + figuras
```

## Estrutura

```
config.yaml          # configuração do pipeline
prompts/             # templates de prompt (triage, extraction, summarization)
src/                 # módulos do pipeline
data/                # corpus e gold standard
outputs/             # resultados, métricas, figuras
notebooks/           # análise dos resultados
```
