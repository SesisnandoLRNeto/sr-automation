#!/bin/bash
# =============================================================
# run_fresh.sh — Limpa TUDO e roda o pipeline do zero
# =============================================================
#
# O que faz:
#   1. Apaga todos os outputs anteriores (corpus, resultados, cache)
#   2. Coleta artigos de 4 bases acadêmicas
#   3. Roda triagem, extração, sumarização
#   4. Gera amostras para verificação manual
#
# Uso:
#   bash run_fresh.sh
#   bash run_fresh.sh '"educational chatbot" AND ("LLM" OR "GPT")'
#
# Se nenhuma query for passada, usa a query padrão do config.yaml.
# =============================================================

set -e

QUERY_ARG=""
QUERY_DISPLAY="(padrão do config.yaml)"
if [ -n "$1" ]; then
    QUERY_ARG="--query $1"
    QUERY_DISPLAY="$1"
fi

echo ""
echo "============================================"
echo "  SR-Automation — Execução do Zero"
echo "  Query: $QUERY_DISPLAY"
echo "============================================"
echo ""

# --- PASSO 1: Limpar tudo ---
echo "[1/7] Limpando outputs anteriores..."
rm -f data/raw/corpus.csv
rm -f data/gold_standard.csv
rm -f outputs/triage_results.jsonl
rm -f outputs/extraction_results.jsonl
rm -f outputs/summaries.jsonl
rm -f outputs/hallucination_sample.csv
rm -f outputs/extraction_validation.csv
rm -f outputs/audit_log.jsonl
rm -f outputs/metrics.json
rm -f outputs/cross_validation.json
rm -f outputs/latex_tables.tex
rm -rf outputs/.llm_cache/
rm -rf outputs/figures/*.png
echo "   Tudo limpo."
echo ""

# --- PASSO 2: Ativar venv ---
source .venv/bin/activate

# --- PASSO 3: Coletar corpus ---
echo "[2/7] Coletando artigos (Semantic Scholar + OpenAlex + arXiv + ERIC)..."
echo "      Tempo estimado: ~3 min"
time python main.py --step corpus $QUERY_ARG
echo ""

# --- PASSO 4: Triagem automática ---
echo "[3/7] Triagem automática (YES/NO para cada artigo)..."
echo "      Tempo estimado: ~4 min"
time python main.py --step triage
echo ""

# --- PASSO 5: Extração de dados ---
echo "[4/7] Extração de dados (5 campos por artigo incluído)..."
echo "      Tempo estimado: ~2 min"
time python main.py --step extract
echo ""

# --- PASSO 6: Sumarização ---
echo "[5/7] Sumarização (TL;DR 3 frases por artigo)..."
echo "      Tempo estimado: ~3 min"
time python main.py --step summarize
echo ""

# --- PASSO 7: Amostras para verificação manual ---
echo "[6/7] Gerando amostra para verificação de alucinação..."
python main.py --step hallcheck
echo ""

echo "[7/7] Gerando amostra para validação de extração..."
python main.py --step validate
echo ""

# --- RESUMO ---
echo "============================================"
echo "  CONCLUÍDO"
echo "============================================"
echo ""

python3 -c "
import csv, json

with open('data/raw/corpus.csv') as f:
    corpus = list(csv.DictReader(f))
sources = {}
for r in corpus:
    s = r.get('source','?')
    sources[s] = sources.get(s,0) + 1

with open('outputs/triage_results.jsonl') as f:
    triage = [json.loads(l) for l in f if l.strip()]
yes = sum(1 for t in triage if t['decision'] == 'YES')
no = sum(1 for t in triage if t['decision'] == 'NO')

print(f'Corpus: {len(corpus)} artigos')
for s, c in sorted(sources.items()):
    print(f'  {s}: {c}')
print(f'Triagem: {yes} incluídos, {no} excluídos')
print(f'Workload Reduction: {no/len(corpus)*100:.1f}%')
print()
print('Próximos passos:')
print('  1. Preencher data/gold_standard.csv (você + orientador)')
print('  2. Preencher outputs/hallucination_sample.csv (coluna classification)')
print('  3. Preencher outputs/extraction_validation.csv (coluna error_type)')
print('  4. python main.py --step metrics')
print('  5. python main.py --step crossval')
print('  6. python main.py --step report')
"
