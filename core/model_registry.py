"""
model_registry.py — Maintains all available models with health, capabilities, and fallback chains.
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

WORKSPACE = Path("/var/minis/workspace")
CACHE_FILE = WORKSPACE / "nvidia_model_cache.json"
FULL_CACHE_FILE = WORKSPACE / "hermes-framework" / "config" / "model_cache.json"
HEALTH_FILE = WORKSPACE / "hermes-framework" / "config" / "health_status.json"


class ModelRegistry:
    """전체 모델 레지스트리 — 발견, 상태 추적, 폴백 체인 관리"""

    def __init__(self):
        self.models: Dict[str, dict] = {}
        self.health: Dict[str, dict] = {}
        self.fallback_chains: Dict[str, List[str]] = {}
        self._load()

    # ── 초기 로드 ─────────────────────────────────────────────
    def _load(self):
        """초기 데이터 로드 — 캐시 또는 새로 발견"""
        if FULL_CACHE_FILE.exists():
            self.models = json.loads(FULL_CACHE_FILE.read_text())
        else:
            self.discover()

        if HEALTH_FILE.exists():
            self.health = json.loads(HEALTH_FILE.read_text())
        else:
            self.health = {}

        # 폴백 체인 빌드
        self.fallback_chains = {
            "fast": self._build_fast_chain(),
            "expert": self._build_expert_chain(),
            "code": self._build_code_chain(),
            "vision": self._build_vision_chain(),
            "safety": self._build_safety_chain(),
            "translate": self._build_translate_chain(),
        }

    # ── 모델 발견 ─────────────────────────────────────────────
    def discover(self, force: bool = False) -> dict:
        """minis-model-use list를 호출해 전체 모델 카탈로그 발견"""
        if not force and FULL_CACHE_FILE.exists() and self._cache_fresh():
            return {"models": len(self.models), "cached": True}

        result = subprocess.run(
            ["minis-model-use", "list"], capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f"모델 발견 실패: {result.stderr}")

        data = json.loads(result.stdout)
        models = data.get("data", {}).get("models", [])

        # 텍스트 출력 가능한 모델만
        usable = [m for m in models if "text_output" in m.get("modalities", [])]

        self.models = {}
        for m in usable:
            self.models[m["model_id"]] = {
                "model_id": m["model_id"],
                "display_name": m["display_name"],
                "provider": m["instance_label"],
                "provider_type": m["provider_type"],
                "context_window": m["context_window"],
                "modalities": m["modalities"],
                "entry_id": m["entry_id"],
            }

        FULL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        FULL_CACHE_FILE.write_text(
            json.dumps(self.models, indent=2, ensure_ascii=False)
        )
        return {"models": len(self.models), "cached": False}

    def _cache_fresh(self) -> bool:
        try:
            mtime = datetime.fromtimestamp(FULL_CACHE_FILE.stat().st_mtime)
            return datetime.now() - mtime < timedelta(hours=24)
        except Exception:
            return False

    # ── 상태 관리 ─────────────────────────────────────────────
    def mark_success(self, model_id: str, latency_ms: int):
        h = self.health.setdefault(model_id, {})
        h["last_success"] = datetime.now().isoformat()
        h["consecutive_errors"] = 0
        h["total_success"] = h.get("total_success", 0) + 1
        h["latency_ms"] = latency_ms
        h["status"] = "HEALTHY"
        self._save_health()

    def mark_error(self, model_id: str, error: str):
        h = self.health.setdefault(model_id, {})
        h["last_error"] = datetime.now().isoformat()
        h["consecutive_errors"] = h.get("consecutive_errors", 0) + 1
        h["last_error_msg"] = error[:200]
        h["total_errors"] = h.get("total_errors", 0) + 1
        if h["consecutive_errors"] >= 3:
            h["status"] = "DEAD"
        elif h.get("total_errors", 0) >= 2:
            h["status"] = "SLOW"
        else:
            h["status"] = "WARN"
        self._save_health()

    def _save_health(self):
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(json.dumps(self.health, indent=2, ensure_ascii=False))

    def is_usable(self, model_id: str) -> bool:
        h = self.health.get(model_id, {})
        return h.get("status") != "DEAD"

    # ── 폴백 체인 ─────────────────────────────────────────────
    def _build_fast_chain(self) -> List[str]:
        """Fast(빠른 응답) 역할 — 8B~20B 작은 모델 우선, 그 다음 큰 모델"""
        candidates = [
            "meta/llama-3.2-1b-instruct",
            "meta/llama-3.1-8b-instruct",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "mistralai/mistral-nemotron",
            "nvidia/nvidia-nemotron-nano-9b-v2",
            "nvidia/nemotron-mini-4b-instruct",
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-v4-pro",
            "nvidia/nemotron-3-nano-30b-a3b",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-medium-3.5-128b",
            "nvidia/nemotron-3-super-120b-a12b",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def _build_expert_chain(self) -> List[str]:
        """Expert(심층 추론) 역할 — 대형 모델 우선"""
        candidates = [
            "nvidia/nemotron-3-ultra-550b-a55b",
            "z-ai/glm-5.2",
            "thinkingmachines/inkling",
            "minimaxai/minimax-m3",
            "deepseek-ai/deepseek-v4-pro",
            "nvidia/nemotron-3-super-120b-a12b",
            "mistralai/mistral-small-4-119b-2603",
            "mistralai/mistral-medium-3.5-128b",
            "meta/llama-3.3-70b-instruct",
            "openai/gpt-oss-120b",
            "mistralai/mistral-nemotron",
            "moonshotai/kimi-k2.6",
            "qwen/qwen3-next-80b-a3b-instruct",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def _build_code_chain(self) -> List[str]:
        """Code(코드) 역할 — 코드 전문 모델 우선"""
        candidates = [
            "poolside/laguna-xs-2.1",
            "deepseek-ai/deepseek-v4-pro",
            "deepseek-ai/deepseek-v4-flash",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-next-80b-a3b-instruct",
            "minimaxai/minimax-m3",
            "z-ai/glm-5.2",
            "stepfun-ai/step-3.7-flash",
            "nvidia/nemotron-3-nano-30b-a3b",
            "meta/llama-3.3-70b-instruct",
            "mistralai/mistral-nemotron",
            "meta/llama-3.1-8b-instruct",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def _build_vision_chain(self) -> List[str]:
        """Vision(멀티모달) 역할 — 이미지 입력 가능한 모델"""
        candidates = [
            "thinkingmachines/inkling",
            "minimaxai/minimax-m3",
            "google/gemma-4-31b-it",
            "google/gemma-3n-e4b-it",
            "google/gemma-3n-e2b-it",
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/nemotron-nano-12b-v2-vl",
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "moonshotai/kimi-k2.6",
            "stepfun-ai/step-3.7-flash",
            "mistralai/mistral-small-4-119b-2603",
            "mistralai/mistral-medium-3.5-128b",
            "mistralai/ministral-14b-instruct-2512",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def _build_safety_chain(self) -> List[str]:
        """Safety(안전성) 역할"""
        candidates = [
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "mistralai/mistral-nemotron",
            "meta/llama-3.1-8b-instruct",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def _build_translate_chain(self) -> List[str]:
        """Translate(번역) 역할"""
        candidates = [
            "minimaxai/minimax-m3",
            "minimaxai/minimax-m2.7",
            "sarvamai/sarvam-m",
            "moonshotai/kimi-k2.6",
            "openai/gpt-oss-120b",
            "deepseek-ai/deepseek-v4-pro",
            "google/gemma-4-31b-it",
            "meta/llama-3.3-70b-instruct",
        ]
        return [m for m in candidates if m in self.models and self.is_usable(m)]

    def chain_for(self, role: str) -> List[str]:
        return self.fallback_chains.get(role, self.fallback_chains["fast"])

    def summary(self) -> dict:
        return {
            "total_models": len(self.models),
            "healthy": sum(1 for h in self.health.values() if h.get("status") == "HEALTHY"),
            "warn": sum(1 for h in self.health.values() if h.get("status") == "WARN"),
            "slow": sum(1 for h in self.health.values() if h.get("status") == "SLOW"),
            "dead": sum(1 for h in self.health.values() if h.get("status") == "DEAD"),
            "chains": {r: len(c) for r, c in self.fallback_chains.items()},
        }


if __name__ == "__main__":
    r = ModelRegistry()
    s = r.discover(force=True)
    print(f"✅ {s['models']}개 모델 발견 (cached={s['cached']})")
    print(f"\n📊 폴백 체인 길이:")
    for role, chain in r.fallback_chains.items():
        print(f"  {role:10s}: {len(chain):2d}개 → {chain[:3]}{'...' if len(chain)>3 else ''}")
    print(f"\n📈 상태 요약: {r.summary()}")
