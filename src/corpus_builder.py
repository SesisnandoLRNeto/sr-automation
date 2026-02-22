"""
Coleta de corpus de 3 fontes: Semantic Scholar, arXiv, ERIC.
Ref: §3.5.6 Fase 1 — busca, deduplicação, filtro.

Meta: 80-120 artigos após dedup e filtro.
"""

import difflib
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _safe_get(url: str, params: dict = None, headers: dict = None,
              retries: int = 3, backoff: float = 2.0) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = backoff ** (attempt + 1)
                logger.warning(f"Rate limit 429. Aguardando {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = backoff ** (attempt + 1)
                logger.warning(f"Erro ({e}). Retry em {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Falha após {retries} tentativas: {e}")
                return None
    return None


def query_semantic_scholar(config: dict) -> list[dict]:
    """Busca artigos via Semantic Scholar API."""
    logger.info("Buscando no Semantic Scholar...")
    articles = []
    query = config["corpus"]["query"]
    year_start, year_end = config["corpus"]["year_range"]
    headers = {}
    s2_key = __import__("os").getenv("S2_API_KEY", "")
    if s2_key:
        headers["x-api-key"] = s2_key

    fields = "title,abstract,authors,year,externalIds,openAccessPdf"
    offset = 0
    limit = 100

    while offset < 300:
        resp = _safe_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "year": f"{year_start}-{year_end}",
                "fields": fields,
                "offset": offset,
                "limit": limit,
                "fieldsOfStudy": "Computer Science",
            },
            headers=headers,
        )
        if resp is None:
            break

        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            break

        for p in papers:
            ext_ids = p.get("externalIds") or {}
            pdf_info = p.get("openAccessPdf") or {}
            articles.append({
                "id": f"s2_{p.get('paperId', '')}",
                "title": (p.get("title") or "").strip(),
                "abstract": (p.get("abstract") or "").strip(),
                "authors": ", ".join(
                    a.get("name", "") for a in (p.get("authors") or [])
                ),
                "year": p.get("year"),
                "doi": ext_ids.get("DOI", ""),
                "source": "semantic_scholar",
                "url": pdf_info.get("url", ""),
            })

        total = data.get("total", 0)
        offset += limit
        if offset >= total:
            break
        time.sleep(1)

    logger.info(f"Semantic Scholar: {len(articles)} artigos coletados")
    return articles


def query_arxiv(config: dict) -> list[dict]:
    """Busca artigos via arXiv API."""
    logger.info("Buscando no arXiv...")
    articles = []
    query = config["corpus"]["query"]
    year_start, year_end = config["corpus"]["year_range"]

    # Construir query arXiv com categorias relevantes
    search_query = (
        f"all:{query} AND "
        f"(cat:cs.CL OR cat:cs.AI OR cat:cs.CY)"
    )

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    start = 0
    max_results = 100

    while start < 300:
        resp = _safe_get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        if resp is None:
            break

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            title = (entry.findtext("atom:title", "", ns) or "").strip()
            title = re.sub(r"\s+", " ", title)
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
            abstract = re.sub(r"\s+", " ", abstract)
            published = entry.findtext("atom:published", "", ns)
            year = int(published[:4]) if published else None

            if year and (year < year_start or year > year_end):
                continue

            authors = [
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            ]
            arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]

            # Extrair DOI dos links se disponível
            doi = ""
            for link in entry.findall("atom:link", ns):
                href = link.get("href", "")
                if "doi.org" in href:
                    doi = href.replace("http://dx.doi.org/", "").replace("https://doi.org/", "")

            articles.append({
                "id": f"arxiv_{arxiv_id}",
                "title": title,
                "abstract": abstract,
                "authors": ", ".join(authors),
                "year": year,
                "doi": doi,
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            })

        start += max_results
        if len(entries) < max_results:
            break
        time.sleep(3)  # arXiv pede 3s entre requests

    logger.info(f"arXiv: {len(articles)} artigos coletados")
    return articles


def query_eric(config: dict) -> list[dict]:
    """Busca artigos via ERIC API (https://api.ies.ed.gov/eric/)."""
    logger.info("Buscando no ERIC...")
    articles = []
    year_start, year_end = config["corpus"]["year_range"]

    search_terms = "intelligent tutoring system AND large language model"
    start = 0
    rows = 200

    resp = _safe_get(
        "https://api.ies.ed.gov/eric/",
        params={
            "search": search_terms,
            "start": start,
            "rows": rows,
            "format": "json",
        },
    )
    if resp is None:
        logger.error("ERIC: falha na consulta")
        return articles

    try:
        data = resp.json()
    except Exception as e:
        logger.error(f"ERIC: erro ao parsear JSON: {e}")
        return articles

    docs = data.get("response", {}).get("docs", [])
    for doc in docs:
        year_str = doc.get("publicationdateyear", "")
        try:
            year = int(year_str) if year_str else None
        except ValueError:
            year = None

        if year and (year < year_start or year > year_end):
            continue

        title = (doc.get("title", "") or "").strip()
        abstract = (doc.get("description", "") or "").strip()

        articles.append({
            "id": f"eric_{doc.get('id', '')}",
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(doc.get("author", [])),
            "year": year,
            "doi": doc.get("doi", ""),
            "source": "eric",
            "url": f"https://eric.ed.gov/?id={doc.get('id', '')}",
        })

    logger.info(f"ERIC: {len(articles)} artigos coletados")
    return articles


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _deduplicate(articles: list[dict], threshold: float = 0.9) -> list[dict]:
    """Deduplicação por DOI exato + similaridade de título."""
    seen_dois = set()
    seen_titles = []
    unique = []

    for art in articles:
        # Dedup por DOI
        doi = (art.get("doi") or "").strip().lower()
        if doi and doi in seen_dois:
            continue

        # Dedup por título similar
        norm_title = _normalize_title(art["title"])
        is_dup = False
        for existing in seen_titles:
            ratio = difflib.SequenceMatcher(None, norm_title, existing).ratio()
            if ratio >= threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(art)
            seen_titles.append(norm_title)
            if doi:
                seen_dois.add(doi)

    return unique


def build_corpus(config: dict) -> pd.DataFrame:
    """
    Coleta 80-120 artigos de 3 fontes, deduplica e filtra.
    Salva em data/raw/corpus.csv.
    """
    all_articles = []

    # Coleta de cada fonte (tolerante a falhas)
    sources = config["corpus"].get("sources", ["semantic_scholar", "arxiv", "eric"])

    if "semantic_scholar" in sources:
        try:
            all_articles.extend(query_semantic_scholar(config))
        except Exception as e:
            logger.error(f"Semantic Scholar falhou: {e}")

    if "arxiv" in sources:
        try:
            all_articles.extend(query_arxiv(config))
        except Exception as e:
            logger.error(f"arXiv falhou: {e}")

    if "eric" in sources:
        try:
            all_articles.extend(query_eric(config))
        except Exception as e:
            logger.error(f"ERIC falhou: {e}")

    logger.info(f"Total coletado (bruto): {len(all_articles)}")

    # Contagem por fonte
    source_counts = {}
    for a in all_articles:
        src = a.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in source_counts.items():
        logger.info(f"  {src}: {count}")

    # Deduplicação
    threshold = config["corpus"].get("dedup_threshold", 0.9)
    unique = _deduplicate(all_articles, threshold)
    logger.info(f"Após deduplicação: {len(unique)}")

    # Filtro: abstract válido (>= 50 chars)
    filtered = [a for a in unique if a.get("abstract") and len(a["abstract"]) >= 50]
    logger.info(f"Após filtro de abstract: {len(filtered)}")

    # Verificar range alvo
    target_min, target_max = config["corpus"].get("target_size", [80, 120])
    if len(filtered) < target_min:
        logger.warning(
            f"ATENÇÃO: {len(filtered)} artigos — abaixo do mínimo ({target_min})"
        )
    elif len(filtered) > target_max:
        logger.warning(
            f"ATENÇÃO: {len(filtered)} artigos — acima do máximo ({target_max})"
        )
    else:
        logger.info(f"OK: {len(filtered)} artigos dentro do range [{target_min}, {target_max}]")

    # Salvar CSV
    df = pd.DataFrame(filtered)
    output_path = config["paths"]["corpus"]
    __import__("os").makedirs(__import__("os").path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Corpus salvo em {output_path}")

    return df
