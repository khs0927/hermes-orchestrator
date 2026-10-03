from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _bounded_strings(value: Any, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value[:limit]:
        if isinstance(item, str) and item and len(item) <= 200:
            rows.append(item)
    return rows


def query_resource_hints(prompt: str, k: int = 8) -> list[dict[str, Any]]:
    """Read advisory resource candidates from zcode-resource-router.

    This is discovery only. The returned candidates never authorize tools,
    writes, browser actions, CAD mutations, or other consequential execution.
    """
    raw_path = os.environ.get("HERMES_RESOURCE_ROUTER", "").strip()
    if not raw_path or not prompt.strip():
        return []
    router = Path(raw_path)
    if not router.is_file():
        return []

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(router),
                prompt,
                "--k",
                str(max(1, min(k, 20))),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    hints: list[dict[str, Any]] = []
    for item in payload[:20]:
        if not isinstance(item, dict):
            continue
        resource_id = item.get("id")
        resource_type = item.get("type")
        name = item.get("name")
        if not all(isinstance(value, str) and value for value in (resource_id, resource_type, name)):
            continue
        score = item.get("score")
        verification_status = item.get("verification_status")
        hint = {
            "id": resource_id,
            "type": resource_type,
            "name": name,
            "score": score if isinstance(score, (int, float)) else None,
        }
        if isinstance(verification_status, str) and verification_status:
            hint["verification_status"] = verification_status
        if resource_type == "project_contract":
            hint["verified_consumers"] = _bounded_strings(item.get("verified_consumers"))
            hint["pending_consumers"] = _bounded_strings(item.get("pending_consumers"))
            real_cad_e2e = item.get("real_cad_e2e")
            hint["real_cad_e2e"] = real_cad_e2e if isinstance(real_cad_e2e, bool) else False
        hints.append(hint)
    return hints
