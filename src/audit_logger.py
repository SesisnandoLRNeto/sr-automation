"""Logging estruturado JSONL para rastreabilidade das chamadas LLM."""

import json
import os
from datetime import datetime, timezone
from typing import Optional


class AuditLogger:
    """Logger JSONL append-only."""

    def __init__(self, output_path: str = "outputs/audit_log.jsonl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._path = output_path
        self._file = open(output_path, "a", encoding="utf-8")

    def log(
        self,
        module: str,
        article_id: str,
        decision: str,
        confidence: float,
        prompt_hash: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        raw_response: str,
        extra: Optional[dict] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "article_id": article_id,
            "decision": decision,
            "confidence": confidence,
            "prompt_hash": prompt_hash,
            "provider": provider,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 2),
            "raw_response": raw_response,
        }
        if extra:
            record.update(extra)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
