"""Backend FastAPI para a interface web do SR-Automation."""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import random
import shutil
import tempfile
import threading
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Diretório raiz do projeto (um nível acima de web/)
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Importa funções do pipeline
import sys
sys.path.insert(0, str(ROOT))
from main import (
    step_corpus,
    step_crossval,
    step_extract,
    step_gold,
    step_hallcheck,
    step_metrics,
    step_report,
    step_summarize,
    step_triage,
    step_validate,
    _load_corpus,
    _load_triage_results,
)
from src.utils import load_config, setup_logging

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_current_step: str | None = None
_current_thread: threading.Thread | None = None
_log_buffer: deque = deque(maxlen=500)
_log_index: int = 0  # Contador global para SSE
_lock = threading.Lock()


class WebLogHandler(logging.Handler):
    """Captura logs do Python para streaming via SSE."""

    def emit(self, record):
        global _log_index
        try:
            msg = self.format(record)
            with _lock:
                _log_index += 1
                _log_buffer.append({"id": _log_index, "msg": msg})
        except Exception:
            pass


# Configura logging
setup_logging()
_web_handler = WebLogHandler()
_web_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                      datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_web_handler)

logger = logging.getLogger("web")

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="SR-Automation Web", docs_url="/docs")

# Monta arquivos estáticos
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _config() -> dict:
    return load_config(str(ROOT / "config.yaml"))


def _path(rel: str) -> Path:
    return ROOT / rel


STEP_FILES = {
    "corpus": "data/raw/corpus.csv",
    "triage": "outputs/triage_results.jsonl",
    "extract": "outputs/extraction_results.jsonl",
    "summarize": "outputs/summaries.jsonl",
    "hallcheck": "outputs/hallucination_sample.csv",
    "validate": "outputs/extraction_validation.csv",
    "gold": "data/gold_standard.csv",
    "metrics": "outputs/metrics.json",
    "crossval": "outputs/cross_validation.json",
    "report": "outputs/figures/confusion_matrix.png",
}

STEP_DEPS = {
    "corpus": [],
    "triage": ["corpus"],
    "extract": ["triage"],
    "summarize": ["extract"],
    "hallcheck": ["extract"],
    "validate": ["extract"],
    "gold": ["corpus"],
    "metrics": ["gold"],
    "crossval": ["corpus"],
    "report": ["metrics", "crossval"],
}


def _cleanup_stuck_step():
    """Limpa _current_step se a thread morreu (erro/crash)."""
    global _current_step, _current_thread
    with _lock:
        if _current_step and _current_thread and not _current_thread.is_alive():
            logger.warning(f"Step '{_current_step}' travado (thread morta). Limpando.")
            _current_step = None
            _current_thread = None


def _get_step_status(step: str) -> str:
    global _current_step
    _cleanup_stuck_step()
    if _current_step == step:
        return "running"
    f = STEP_FILES.get(step)
    if f and _path(f).exists():
        return "done"
    return "pending"


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _read_csv_as_dicts(path: Path) -> list[dict]:
    if path.exists():
        df = pd.read_csv(path)
        df = df.fillna("")
        return df.to_dict(orient="records")
    return []


# ---------------------------------------------------------------------------
# Endpoints: Status e Controle
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status():
    statuses = {}
    for step in STEP_FILES:
        statuses[step] = _get_step_status(step)

    # Contagens básicas
    counts = {}
    corpus_path = _path("data/raw/corpus.csv")
    if corpus_path.exists():
        try:
            df = pd.read_csv(corpus_path)
            counts["corpus_total"] = len(df)
            if "source" in df.columns:
                counts["corpus_sources"] = df["source"].nunique()
        except Exception:
            pass

    triage_path = _path("outputs/triage_results.jsonl")
    if triage_path.exists():
        try:
            records = _read_jsonl(triage_path)
            counts["triage_yes"] = sum(1 for r in records if r.get("decision") == "YES")
            counts["triage_no"] = sum(1 for r in records if r.get("decision") == "NO")
        except Exception:
            pass

    return {"steps": statuses, "counts": counts, "running": _current_step}


@app.get("/api/config")
async def api_get_config():
    config = _config()
    return {
        "query": config.get("corpus", {}).get("query", ""),
        "year_range": config.get("corpus", {}).get("year_range", [2022, 2024]),
    }


class ConfigUpdate(BaseModel):
    query: str | None = None
    year_range: list[int] | None = None


@app.post("/api/config")
async def api_update_config(update: ConfigUpdate):
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if update.query is not None:
        config["corpus"]["query"] = update.query
    if update.year_range is not None and len(update.year_range) == 2:
        config["corpus"]["year_range"] = update.year_range

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return {"ok": True}


# ---------------------------------------------------------------------------
# Execução de steps em background
# ---------------------------------------------------------------------------
def _run_step_in_thread(step: str):
    """Executa um step do pipeline numa thread separada."""
    global _current_step
    config = _config()

    try:
        logger.info(f"Iniciando step: {step}")

        if step == "corpus":
            step_corpus(config)
        elif step == "triage":
            corpus_df = _load_corpus(config)
            step_triage(config, corpus_df)
        elif step == "extract":
            corpus_df = _load_corpus(config)
            triage_df = _load_triage_results(config)
            step_extract(config, triage_df, corpus_df)
        elif step == "summarize":
            corpus_df = _load_corpus(config)
            triage_df = _load_triage_results(config)
            step_summarize(config, triage_df, corpus_df)
        elif step == "hallcheck":
            step_hallcheck(config)
        elif step == "validate":
            step_validate(config)
        elif step == "gold":
            step_gold(config)
        elif step == "metrics":
            step_metrics(config)
        elif step == "crossval":
            corpus_df = _load_corpus(config)
            step_crossval(config, corpus_df)
        elif step == "report":
            step_report(config)

        logger.info(f"Step '{step}' concluido com sucesso.")
    except SystemExit:
        logger.error(f"Step '{step}' falhou (SystemExit).")
    except Exception as e:
        logger.error(f"Step '{step}' falhou: {e}")
    finally:
        with _lock:
            _current_step = None
            _current_thread = None


@app.post("/api/run/{step}")
async def api_run_step(step: str):
    global _current_step, _current_thread

    if step not in STEP_FILES:
        raise HTTPException(400, f"Step desconhecido: {step}")

    with _lock:
        # Limpar step travado (thread morta)
        if _current_step and _current_thread and not _current_thread.is_alive():
            _current_step = None
            _current_thread = None

        if _current_step is not None:
            raise HTTPException(409, f"Step '{_current_step}' ja esta rodando.")
        _current_step = step

    t = threading.Thread(target=_run_step_in_thread, args=(step,), daemon=True)
    t.start()

    with _lock:
        _current_thread = t

    return {"ok": True, "step": step}


@app.post("/api/reset")
async def api_reset():
    global _current_step

    with _lock:
        if _current_step is not None:
            raise HTTPException(409, f"Step '{_current_step}' esta rodando. Aguarde.")

    files_to_remove = [
        "data/raw/corpus.csv",
        "data/gold_standard.csv",
        "outputs/triage_results.jsonl",
        "outputs/extraction_results.jsonl",
        "outputs/summaries.jsonl",
        "outputs/hallucination_sample.csv",
        "outputs/extraction_validation.csv",
        "outputs/likert_evaluation.csv",
        "outputs/audit_log.jsonl",
        "outputs/metrics.json",
        "outputs/cross_validation.json",
        "outputs/latex_tables.tex",
    ]

    removed = []
    for f in files_to_remove:
        p = _path(f)
        if p.exists():
            p.unlink()
            removed.append(f)

    # Limpar cache e figuras
    cache_dir = _path("outputs/.llm_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        removed.append("outputs/.llm_cache/")

    figures_dir = _path("outputs/figures")
    if figures_dir.exists():
        for png in figures_dir.glob("*.png"):
            png.unlink()
            removed.append(str(png.relative_to(ROOT)))

    logger.info(f"Reset: {len(removed)} arquivos removidos.")
    return {"ok": True, "removed": removed}


# ---------------------------------------------------------------------------
# SSE: Logs em tempo real
# ---------------------------------------------------------------------------
@app.get("/api/logs")
async def api_logs():
    async def event_stream():
        last_id = 0
        while True:
            with _lock:
                new_msgs = [m for m in _log_buffer if m["id"] > last_id]
            for m in new_msgs:
                last_id = m["id"]
                yield f"data: {json.dumps(m)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Endpoints: Dados (leitura)
# ---------------------------------------------------------------------------
@app.get("/api/data/corpus")
async def api_data_corpus():
    return _read_csv_as_dicts(_path("data/raw/corpus.csv"))


@app.get("/api/data/triage")
async def api_data_triage():
    return _read_jsonl(_path("outputs/triage_results.jsonl"))


@app.get("/api/data/extraction")
async def api_data_extraction():
    return _read_jsonl(_path("outputs/extraction_results.jsonl"))


@app.get("/api/data/summaries")
async def api_data_summaries():
    return _read_jsonl(_path("outputs/summaries.jsonl"))


@app.get("/api/data/metrics")
async def api_data_metrics():
    p = _path("outputs/metrics.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/api/data/crossval")
async def api_data_crossval():
    p = _path("outputs/cross_validation.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Endpoints: Revisao manual (leitura + escrita)
# ---------------------------------------------------------------------------
@app.get("/api/data/gold")
async def api_data_gold():
    return _read_csv_as_dicts(_path("data/gold_standard.csv"))


@app.post("/api/data/gold")
async def api_save_gold(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(_path("data/gold_standard.csv"), index=False, encoding="utf-8")
    logger.info(f"Gold standard salvo: {len(df)} linhas.")
    return {"ok": True, "count": len(df)}


@app.get("/api/data/hallucination")
async def api_data_hallucination():
    return _read_csv_as_dicts(_path("outputs/hallucination_sample.csv"))


@app.post("/api/data/hallucination")
async def api_save_hallucination(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(_path("outputs/hallucination_sample.csv"), index=False, encoding="utf-8")
    logger.info(f"Hallucination sample salvo: {len(df)} linhas.")
    return {"ok": True, "count": len(df)}


@app.get("/api/data/validation")
async def api_data_validation():
    return _read_csv_as_dicts(_path("outputs/extraction_validation.csv"))


@app.post("/api/data/validation")
async def api_save_validation(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(_path("outputs/extraction_validation.csv"), index=False, encoding="utf-8")
    logger.info(f"Extraction validation salvo: {len(df)} linhas.")
    return {"ok": True, "count": len(df)}


@app.get("/api/data/likert")
async def api_data_likert():
    return _read_csv_as_dicts(_path("outputs/likert_evaluation.csv"))


@app.post("/api/data/likert")
async def api_save_likert(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(_path("outputs/likert_evaluation.csv"), index=False, encoding="utf-8")
    logger.info(f"Likert evaluation salvo: {len(df)} linhas.")
    return {"ok": True, "count": len(df)}


@app.post("/api/generate-likert-sample")
async def api_generate_likert_sample():
    summaries_path = _path("outputs/summaries.jsonl")
    triage_path = _path("outputs/triage_results.jsonl")

    if not summaries_path.exists():
        raise HTTPException(404, "Arquivo summaries.jsonl nao encontrado. Execute o step Resumos primeiro.")
    if not triage_path.exists():
        raise HTTPException(404, "Arquivo triage_results.jsonl nao encontrado. Execute o step Triagem primeiro.")

    summaries = _read_jsonl(summaries_path)
    triage = _read_jsonl(triage_path)

    # Lookup de resumos por article_id
    summary_lookup = {}
    for s in summaries:
        sid = s.get("id") or s.get("article_id", "")
        # Tentar campo summary/tldr, senao montar a partir de problem/solution/findings ou raw_response
        text = s.get("summary") or s.get("tldr", "")
        if not text:
            parts = []
            for field in ("problem", "solution", "findings"):
                if s.get(field):
                    parts.append(s[field])
            text = " ".join(parts) if parts else s.get("raw_response", "")
        summary_lookup[sid] = text

    # Selecionar artigos que possuem resumo (apenas YES terao resumo)
    articles_with_summary = [r for r in triage if (r.get("id") or r.get("article_id", "")) in summary_lookup]

    # Selecionar ate 20 artigos aleatoriamente
    sample = random.sample(articles_with_summary, min(20, len(articles_with_summary)))

    # Carregar corpus para titulos
    corpus_path = _path("data/raw/corpus.csv")
    corpus_lookup = {}
    if corpus_path.exists():
        corpus_df = pd.read_csv(corpus_path)
        for _, row in corpus_df.iterrows():
            corpus_lookup[row.get("id", "")] = row.get("title", "")

    rows = []
    for art in sample:
        aid = art.get("id") or art.get("article_id", "")
        title = art.get("title") or corpus_lookup.get(aid, "")
        summary = summary_lookup.get(aid, "")
        rows.append({
            "article_id": aid,
            "title": title,
            "summary": summary,
            "clareza": "",
            "completude": "",
            "acuracia": "",
            "utilidade": "",
            "notas": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(_path("outputs/likert_evaluation.csv"), index=False, encoding="utf-8")
    logger.info(f"Likert sample gerado: {len(df)} resumos.")
    return {"ok": True, "count": len(df)}


# ---------------------------------------------------------------------------
# Exportar resultados (zip com timestamp)
# ---------------------------------------------------------------------------
@app.post("/api/export")
async def api_export():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"sr_export_{timestamp}.zip"

    # Arquivos a incluir no zip
    export_files = [
        "config.yaml",
        "data/raw/corpus.csv",
        "data/gold_standard.csv",
        "outputs/triage_results.jsonl",
        "outputs/extraction_results.jsonl",
        "outputs/summaries.jsonl",
        "outputs/hallucination_sample.csv",
        "outputs/extraction_validation.csv",
        "outputs/likert_evaluation.csv",
        "outputs/audit_log.jsonl",
        "outputs/metrics.json",
        "outputs/cross_validation.json",
        "outputs/latex_tables.tex",
    ]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            count = 0
            for rel in export_files:
                p = _path(rel)
                if p.exists():
                    zf.write(p, rel)
                    count += 1

            # Adicionar figuras
            figures_dir = _path("outputs/figures")
            if figures_dir.exists():
                for png in figures_dir.glob("*.png"):
                    rel_path = str(png.relative_to(ROOT))
                    zf.write(png, rel_path)
                    count += 1

        if count == 0:
            os.unlink(tmp.name)
            raise HTTPException(404, "Nenhum arquivo para exportar.")

        logger.info(f"Export: {count} arquivos compactados em {zip_name}")
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=zip_name,
            background=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(500, f"Erro ao criar zip: {e}")


# ---------------------------------------------------------------------------
# Servir imagens de figuras
# ---------------------------------------------------------------------------
@app.get("/api/figures/{filename}")
async def api_figure(filename: str):
    # Sanitiza o nome do arquivo
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Nome de arquivo invalido.")
    p = _path(f"outputs/figures/{filename}")
    if not p.exists():
        raise HTTPException(404, f"Figura nao encontrada: {filename}")
    return FileResponse(str(p), media_type="image/png")
