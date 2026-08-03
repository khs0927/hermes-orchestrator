#!/usr/bin/env python3
"""
model_sync.py - 새 모델 자동 발견 및 config.json 업데이트

Usage:
  python3 model_sync.py --check    # 새 모델만 확인
  python3 model_sync.py --update   # 새 모델을 폴백 체인에 추가
"""
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path("/var/minis/skills/hybrid-orchestrator")
CONFIG_FILE = SKILL_DIR / "config" / "config.json"


def discover_new_models():
    """minis-model-use list로 전체 모델 중 config에 없는 것 발견"""
    result = subprocess.run(
        ["minis-model-use", "list"], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"❌ 모델 목록 가져오기 실패: {result.stderr}")
        return []

    data = json.loads(result.stdout)
    current_models = set()
    for m in data.get("data", {}).get("models", []):
        if "text_output" in m.get("modalities", []):
            current_models.add(m["model_id"])

    # config에 이미 등록된 모델 확인
    config = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    known_models = set()
    for role_key in ["fast", "expert", "code", "vision", "safety", "translate"]:
        known_models.update(config.get("role_map", {}).get(role_key, {}).get("candidates", []))

    new_models = current_models - known_models
    return sorted(new_models)


def suggest_role(model_id: str) -> str:
    """모델 이름 기반 적절한 역할 추천"""
    lower = model_id.lower()
    if any(k in lower for k in ["vision", "vl", "gemma", "omni", "inkling"]):
        return "vision"
    if any(k in lower for k in ["code", "coder", "laguna", "dev"]):
        return "code"
    if any(k in lower for k in ["550b", "405b", "ultra", "expert", "reasoning", "thinking"]):
        return "expert"
    return "fast"


def update_config(new_models: list):
    """config.json에 새 모델 추가"""
    if not CONFIG_FILE.exists():
        print("❌ config.json 없음")
        return

    config = json.loads(CONFIG_FILE.read_text())
    if "role_map" not in config:
        config["role_map"] = {}

    added = 0
    for model_id in new_models:
        role = suggest_role(model_id)
        chains = config["role_map"].setdefault(role, {})
        candidates = chains.setdefault("candidates", [])
        if model_id not in candidates:
            candidates.append(model_id)
            added += 1

    if added > 0:
        config["last_sync"] = datetime.now().isoformat()
        config["total_models"] = sum(
            len(v.get("candidates", [])) for v in config["role_map"].values()
        )
        CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"✅ {added}개 새 모델이 config.json에 추가 첨부 완료")
    else:
        print("✅ 추가할 새 모델 없음")

    return added


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "--check"

    if action == "--check":
        new = discover_new_models()
        print(f"🔍 새 모델: {len(new)}개")
        for m in new:
            role = suggest_role(m)
            print(f"  🆕 {m} → {role}")
    elif action == "--update":
        new = discover_new_models()
        update_config(new)
    else:
        print("Usage: python3 model_sync.py [--check|--update]")