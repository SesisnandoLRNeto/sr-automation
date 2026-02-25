#!/bin/bash
# Script para execução do zero — pipeline completo
# Uso: bash run_fresh.sh

set -e

echo "============================================"
echo "  SR-Automation — Execução do Zero"
echo "============================================"
echo ""

# 1. Limpar outputs anteriores
echo "[1/8] Limpando outputs anteriores..."
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
rm -rf outputs/.llm_cache/
echo "   Limpo."
echo ""

# 2. Ativar venv
source .venv/bin/activate

# 3. Corpus
echo "[2/8] Coletando corpus (4 fontes: S2 + OpenAlex + arXiv + ERIC)..."
echo "   Tempo estimado: ~3 min"
time python main.py --step corpus
echo ""

# 4. Triage
echo "[3/8] Executando triagem automática..."
echo "   Tempo estimado: ~4 min"
time python main.py --step triage
echo ""

# 5. Extract
echo "[4/8] Executando extração de dados..."
echo "   Tempo estimado: ~2 min"
time python main.py --step extract
echo ""

# 6. Summarize
echo "[5/8] Executando sumarização..."
echo "   Tempo estimado: ~3 min"
time python main.py --step summarize
echo ""

# 7. Hallcheck
echo "[6/8] Gerando amostra para verificação de alucinação..."
python main.py --step hallcheck
echo ""

# 8. Validate
echo "[7/8] Gerando amostra para validação de extração..."
python main.py --step validate
echo ""

# 9. Gerar gold_standard.csv
echo "[8/8] Gerando gold_standard.csv com artigos do corpus..."
python3 -c "
import csv
with open('data/raw/corpus.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
with open('data/gold_standard.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['id','title','reviewer_a','reviewer_b','consensus','justification'])
    for r in rows:
        w.writerow([r['id'], r['title'], '', '', '', ''])
print(f'   Gold standard gerado: {len(rows)} artigos')
"
echo ""

echo "============================================"
echo "  PIPELINE CONCLUÍDO"
echo "============================================"
echo ""
echo "Arquivos gerados:"
echo "  - data/raw/corpus.csv"
echo "  - outputs/triage_results.jsonl"
echo "  - outputs/extraction_results.jsonl"
echo "  - outputs/summaries.jsonl"
echo "  - outputs/hallucination_sample.csv"
echo "  - outputs/extraction_validation.csv"
echo "  - data/gold_standard.csv"
echo ""
echo "Próximo passo: preencher gold_standard.csv com o orientador"
echo "  (colunas reviewer_a, reviewer_b, consensus, justification)"
echo ""

# Resumo rápido
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

print('--- Resumo ---')
print(f'Corpus: {len(corpus)} artigos')
for s, c in sources.items():
    print(f'  {s}: {c}')
print(f'Triagem: {yes} YES, {no} NO')
print(f'Workload Reduction: {no/len(corpus)*100:.1f}%')
"
