#!/usr/bin/env python3
"""Hermes natural-language sub-agent orchestrator for Minis.

Routes a Korean request to specialist workers, calls live NVIDIA NIM models,
and synthesizes a single Korean result.  No credentials are stored here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
CACHE = Path("/var/minis/workspace/nvidia_model_cache.json")

# Minis provider-group policy: primary recovery is Luna, final verification is Claude Opus.
LUNA_PROVIDER = "AeroLink — GPT-5.6"
LUNA_MODELS = ["gpt-5.6-luna"]
OPUS_PROVIDER = "AeroLink — Claude"
OPUS_MODELS = ["claude-opus-5", "claude-opus-4-8"]
DEFAULTS = {
    "planner": ["openai/gpt-oss-120b"],
    "sys-agent": ["openai/gpt-oss-20b", "meta/llama-3.1-8b-instruct"],
    "coder-agent": ["deepseek-ai/deepseek-v4-flash", "deepseek-ai/deepseek-v4-pro"],
    "fetch-agent": ["openai/gpt-oss-120b", "meta/llama-3.1-70b-instruct"],
    "reviewer": ["z-ai/glm-5.2", "deepseek-ai/deepseek-v4-flash"],
}
ROLE_DESCRIPTIONS = {
    "planner": "요청을 분해하고 필요한 서브 에이전트와 순서를 결정",
    "sys-agent": "iOS/Minis 네이티브 기능, 기기·파일·환경 작업 담당",
    "coder-agent": "코드 작성·수정·실행·테스트 담당",
    "fetch-agent": "웹·브라우저·외부 자료 수집 및 출처 정리 담당",
    "reviewer": "각 결과의 사실성·누락·충돌을 독립 검토하고 최종 통합",
}


def available_models() -> set[str]:
    try:
        data = json.loads(CACHE.read_text())
        return set(data.get("chat_models", []))
    except Exception:
        return set()


def call_model(prompt: str, model_candidates: list[str], system: str, max_tokens: int = 1200) -> dict[str, Any]:
    if not API_KEY:
        return {"success": False, "error": "NVIDIA_API_KEY not set"}
    valid = available_models()
    candidates = [m for m in model_candidates if not valid or m in valid]
    if not candidates:
        candidates = model_candidates
    payload = json.dumps({
        "model": candidates[0],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.35,
        "stream": False,
    }, ensure_ascii=False)
    for model in candidates[:3]:
        cmd = ["curl", "-s", "-m", "120", "-H", f"Authorization: Bearer {API_KEY}",
               "-H", "Content-Type: application/json", "-d", payload.replace(candidates[0], model, 1),
               f"{BASE_URL}/chat/completions"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
            data = json.loads(p.stdout or "{}")
            if p.returncode == 0 and data.get("choices"):
                msg = data["choices"][0].get("message", {})
                return {"success": True, "model": data.get("model", model),
                        "output": msg.get("content", ""), "usage": data.get("usage", {})}
            err = data.get("error", {}).get("message", p.stderr[:200] or "empty response")
        except Exception as exc:
            err = str(exc)
    return {"success": False, "model": candidates[0], "error": str(err)}


def call_provider_model(prompt: str, provider: str, models: list[str], system: str, max_tokens: int = 1200) -> dict[str, Any]:
    """Call a specific Minis provider through minis-model-use with provider disambiguation."""
    for model in models:
        payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.2}
        tmp = Path(f"/tmp/hermes_provider_{os.getpid()}.json")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            p = subprocess.run(["minis-model-use", "run", "--provider", provider, "--model", model,
                                "--input", str(tmp)], capture_output=True, text=True, timeout=130)
            data = json.loads(p.stdout or "{}")
            if p.returncode == 0 and data.get("ok"):
                result = data.get("data", {})
                return {"success": True, "provider": provider, "model": model,
                        "output": result.get("output_text", ""), "usage": result.get("usage", {})}
            err = data.get("error", {}).get("message", p.stderr[:300] or "provider call failed")
        except Exception as exc:
            err = str(exc)
        finally:
            try: tmp.unlink()
            except OSError: pass
    return {"success": False, "provider": provider, "model": models[0] if models else "", "error": str(err)}


def verify_with_opus(prompt: str, evidence: str, max_tokens: int = 1200) -> dict[str, Any]:
    verification_prompt = (
        f"검증 대상 요청:\n{prompt}\n\n실행 결과:\n{evidence}\n\n"
        "모든 결과를 독립적으로 검증하세요. 사실 오류, 누락, 모델/provider 불일치, 실행 완료 여부를 판정하고 "
        "검증 완료/검증 필요를 명시한 한국어 최종 검증 보고서를 작성하세요."
    )
    return call_provider_model(verification_prompt, OPUS_PROVIDER, OPUS_MODELS,
                               "당신은 Claude Opus 최종 검증자입니다. 근거 없이 성공을 주장하지 마세요.", max_tokens)


def recover_with_luna(prompt: str, max_tokens: int = 1200) -> dict[str, Any]:
    return call_provider_model(prompt, LUNA_PROVIDER, LUNA_MODELS,
                               "당신은 AeroLink GPT-5.6 Luna 복구 워커입니다. 오류를 해결할 실행안을 한국어로 제시하세요.", max_tokens)


def provider_fallback(prompt: str, evidence: str, max_tokens: int = 1200) -> dict[str, Any]:
    """Luna recovery first, then Claude Opus if Luna cannot recover."""
    luna = recover_with_luna(prompt + "\n\n기존 결과/오류:\n" + evidence, max_tokens)
    if luna.get("success") and luna.get("output"):
        return {"policy": "Luna first", "recovery": luna,
                "verification": verify_with_opus(prompt, luna.get("output", ""), max_tokens)}
    opus = verify_with_opus(prompt, evidence, max_tokens)
    return {"policy": "Luna failed → Claude Opus", "recovery": opus, "verification": opus}


def plan_roles(prompt: str) -> list[str]:
    """Korean/English natural-language role selection."""
    p = prompt.lower()
    explicit = any(x in p for x in ["서브 에이전트", "sub-agent", "subagent", "각각의 역할", "역할을 지정", "오케스트레", "모든 과정"])
    roles: list[str] = []
    if explicit:
        roles = ["sys-agent", "coder-agent", "fetch-agent"]
    else:
        if any(x in p for x in ["기기", "아이폰", "ios", "알람", "캘린더", "건강", "파일", "환경"]):
            roles.append("sys-agent")
        if any(x in p for x in ["코드", "개발", "파이썬", "swift", "디버그", "구현", "테스트", "스크립트"]):
            roles.append("coder-agent")
        if any(x in p for x in ["검색", "웹", "사이트", "뉴스", "자료", "url", "브라우저", "수집"]):
            roles.append("fetch-agent")
        if not roles:
            roles = ["planner"]
    return roles


def worker_prompt(role: str, user_prompt: str) -> str:
    return (
        f"사용자 요청:\n{user_prompt}\n\n"
        f"당신은 {role}입니다. 역할: {ROLE_DESCRIPTIONS[role]}.\n"
        "독립적으로 작업안을 작성하세요. 확인하지 못한 사실은 추정하지 말고 '검증 필요'라고 표시하세요.\n"
        "실행 가능한 단계, 필요한 입력, 결과와 제한사항을 한국어로 구조화하세요."
    )


def orchestrate(prompt: str, max_tokens: int = 1200) -> dict[str, Any]:
    roles = plan_roles(prompt)
    workers: list[dict[str, Any]] = []
    # Planning is included for explicit multi-agent requests; otherwise direct workers are enough.
    selected = (["planner"] + roles) if len(roles) > 1 else roles
    with ThreadPoolExecutor(max_workers=min(3, len(selected))) as pool:
        jobs = {pool.submit(call_model, worker_prompt(r, prompt), DEFAULTS[r],
                            f"당신은 Hermes의 {r} 워커입니다. {ROLE_DESCRIPTIONS[r]} 한국어로 답하세요.", max_tokens): r
                for r in selected}
        for future in as_completed(jobs):
            role = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"success": False, "error": str(exc)}
            workers.append({"role": role, **result})
    workers.sort(key=lambda x: selected.index(x["role"]))
    evidence = "\n\n".join(f"[{w['role']}]\n{w.get('output', w.get('error', '실패'))}" for w in workers)
    synthesis_prompt = (
        f"사용자 요청:\n{prompt}\n\n서브 에이전트 결과:\n{evidence}\n\n"
        "위 결과를 검토해 한국어 최종 답변을 작성하세요. 역할별 결과를 구분하지 말고, "
        "실행 완료/검증 완료/검증 필요를 명확히 구분하세요. "
        "충돌하면 사실을 꾸며내지 말고 충돌 내용을 밝혀 주세요."
    )
    final = call_model(synthesis_prompt,
                       ["z-ai/glm-5.2", "deepseek-ai/deepseek-v4-pro"],
                       "당신은 초안 통합자입니다. 한국어로 실행 결과를 요약하세요.", max_tokens)
    if not final.get("success"):
        fallback = provider_fallback(prompt, evidence, max_tokens)
        final = fallback.get("recovery", {})
        final["fallback_policy"] = fallback.get("policy")
        verification = fallback.get("verification", {})
    else:
        verification = verify_with_opus(prompt, final.get("output", ""), max_tokens)
    verification_ok = bool(verification.get("success"))
    return {"success": bool(final.get("success")) and verification_ok, "request": prompt, "roles": selected,
            "workers": workers, "final": final, "verification": verification,
            "routing": "NVIDIA workers → Luna recovery → Claude Opus final verification"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hermes Korean natural-language sub-agent orchestrator")
    ap.add_argument("prompt", nargs="?", help="한국어 자연어 요청")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--roles", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()
    if args.roles:
        print(json.dumps({"roles": ROLE_DESCRIPTIONS, "models": DEFAULTS}, ensure_ascii=False, indent=2))
        return 0
    if not args.prompt:
        ap.error("prompt required")
    result = orchestrate(args.prompt, args.max_tokens)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("final", {}).get("output") or json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
