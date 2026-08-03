#!/usr/bin/env python3
"""
health_monitor.py - Hermes 모델 건강 상태를 주기적으로 점검하고 리포트 생성
Usage: python3 health_monitor.py [--notify]
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path("/var/minis/skills/hybrid-orchestrator")
sys.path.insert(0, str(SKILL_DIR))

from core.model_registry import ModelRegistry
from core.executor import Executor

HEALTH_REPORT = SKILL_DIR / "config" / "health_report.json"

def quick_ping(executor: Executor, model_id: str, role: str = "fast") -> dict:
    """빠른 핑으로 모델 상태 확인"""
    t0 = time.time()
    try:
        r = executor.execute(
            "OK", role=role, max_tokens=3, prefer_model=model_id, timeout=20
        )
        elapsed = time.time() - t0
        return {
            "model": model_id,
            "ok": r.get("success", False),
            "latency_ms": int(elapsed * 1000),
            "output": r.get("output", "")[:20],
            "attempts": r.get("attempts", 0),
        }
    except Exception as e:
        return {
            "model": model_id,
            "ok": False,
            "error": str(e)[:100],
            "latency_ms": int((time.time() - t0) * 1000),
        }


def health_check(roles: list = None) -> dict:
    """전체 건강 체크"""
    if roles is None:
        roles = ["fast", "expert", "code"]

    registry = ModelRegistry()
    executor = Executor(registry)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(registry.models),
        "checked": {},
        "summary": {"ok": 0, "failed": 0, "slow": 0},
    }

    for role in roles:
        chain = registry.chain_for(role)[:2]  # 역할당 첫 2개만 확인
        for model_id in chain:
            if model_id in report["checked"]:
                continue
            result = quick_ping(executor, model_id, role)
            report["checked"][model_id] = result

            latency = result.get("latency_ms", 99999)
            if result["ok"] and latency < 5000:
                report["summary"]["ok"] += 1
            elif result["ok"] and latency >= 5000:
                report["summary"]["slow"] += 1
            else:
                report["summary"]["failed"] += 1

    # 저장
    HEALTH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    return report


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("🏥 Hermes 건강 검사 시작...")
    report = health_check()
    s = report["summary"]
    print(f"📊 결과: ✅ {s['ok']}개 정상, ⚠️ {s['slow']}개 느림, ❌ {s['failed']}개 실패")
    print(f"📁 리포트: {HEALTH_REPORT}")
    for mid, r in report["checked"].items():
        icon = "✅" if r["ok"] else "❌"
        print(f"  {icon} {mid:40s} {r['latency_ms']:5d}ms")