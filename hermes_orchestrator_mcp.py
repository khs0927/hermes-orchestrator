#!/usr/bin/env python3
"""MCP stdio server for Korean natural-language Hermes orchestration."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/.hermes")
from hermes_orchestrator import (DEFAULTS, ROLE_DESCRIPTIONS, OPUS_PROVIDER, OPUS_MODELS,
                                  orchestrate, call_model, call_provider_model, verify_with_opus,
                                  available_models)


def response(req_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        out["error"] = {"code": -32603, "message": str(error)}
    else:
        out["result"] = result if result is not None else {}
    return out


def text_result(text):
    return {"content": [{"type": "text", "text": text}]}


def handle(req):
    rid = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}
    if method == "initialize":
        return response(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "hermes-korean-orchestrator", "version": "3.0"},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        tools = [
            {"name": "hermes_orchestrate", "description": "한국어 자연어 요청을 planner/sys-agent/coder-agent/fetch-agent/reviewer로 분배하고 NVIDIA 실시간 모델 결과를 통합합니다.", "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "max_tokens": {"type": "integer"}}, "required": ["prompt"]}},
            {"name": "hermes_chat", "description": "단일 Hermes 역할 워커를 호출합니다. role을 생략하면 fast를 사용합니다.", "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "role": {"type": "string", "enum": ["planner", "sys-agent", "coder-agent", "fetch-agent", "reviewer"]}}, "required": ["prompt"]}},
            {"name": "hermes_roles", "description": "서브 에이전트 역할과 기본 실시간 모델을 보여줍니다.", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "hermes_verify_opus", "description": "Claude Opus로 모든 과정과 결과를 최종 검증합니다.", "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["prompt", "evidence"]}},
        ]
        return response(rid, {"tools": tools})
    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if name == "hermes_orchestrate":
            result = orchestrate(args.get("prompt", ""), int(args.get("max_tokens", 1200)))
            final_text = result.get("final", {}).get("output", "")
            verification = result.get("verification", {}).get("output", "")
            text = final_text
            if verification:
                text += "\n\n## Claude Opus 최종 검증\n" + verification
            return response(rid, text_result(text or json.dumps(result, ensure_ascii=False, indent=2)))
        if name == "hermes_chat":
            role = args.get("role", "planner")
            if role not in DEFAULTS:
                return response(rid, error=f"unknown role: {role}")
            result = call_model(args.get("prompt", ""), DEFAULTS[role], f"당신은 Hermes의 {role}입니다. {ROLE_DESCRIPTIONS[role]} 한국어로 답하세요.")
            return response(rid, text_result(result.get("output", result.get("error", "실패"))))
        if name == "hermes_verify_opus":
            result = verify_with_opus(args.get("prompt", ""), args.get("evidence", ""), int(args.get("max_tokens", 1200)))
            return response(rid, text_result(result.get("output", result.get("error", "검증 실패"))))
        if name == "hermes_roles":
            return response(rid, text_result(json.dumps({
                "roles": ROLE_DESCRIPTIONS, "models": DEFAULTS,
                "recovery_policy": "Luna first → Claude Opus",
                "verification_provider": OPUS_PROVIDER,
                "verification_models": OPUS_MODELS,
            }, ensure_ascii=False, indent=2)))
        if name == "hermes_list_models":
            models = sorted(available_models())
            return response(rid, text_result(f"NVIDIA NIM live chat models: {len(models)}\n" + "\n".join(models)))
        return response(rid, error=f"unknown tool: {name}")
    return response(rid, {})


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            out = handle(req)
            if out is not None:
                sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps(response(None, error=exc), ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
