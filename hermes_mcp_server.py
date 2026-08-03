#!/usr/bin/env python3
"""
Hermes Agent MCP Server v2 - Minis Integration
==============================================
Exposes intelligent role-based AI tools via MCP stdio protocol.
Routes to 86 NVIDIA NIM models with auto-fallback.

Tools:
  hermes_chat      - Send prompt (auto-route or explicit model/role)
  hermes_list_models - List available models
  hermes_roles     - Show role configurations
  hermes_get_skills - Browse skills
  hermes_memory_search - Search memory
"""

import sys, os, json, subprocess, re

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
CACHE_FILE      = "/var/minis/workspace/nvidia_model_cache.json"

ROLE_MAP = {
    "fast": {
        "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "meta/llama-3.1-8b-instruct"],
        "fallback": "openai/gpt-oss-120b",
        "system": "You are a fast, efficient assistant. Be concise and direct."
    },
    "expert": {
        "models": ["z-ai/glm-5.2", "deepseek-ai/deepseek-v4-pro", "nvidia/llama-3.3-nemotron-super-49b-v1"],
        "fallback": "z-ai/glm-5.2",
        "system": "You are an expert analyst. Provide thorough, step-by-step reasoning."
    },
    "code": {
        "models": ["deepseek-ai/deepseek-v4-flash", "deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-ai/deepseek-v4-pro"],
        "fallback": "deepseek-ai/deepseek-v4-flash",
        "system": "You are an expert programmer. Write clean, working code with complete solutions."
    },
    "vision": {
        "models": ["meta/llama-3.2-90b-vision-instruct", "meta/llama-3.2-11b-vision-instruct"],
        "fallback": "meta/llama-3.2-90b-vision-instruct",
        "system": "You analyze images and visual content precisely."
    },
    "korean": {
        "models": ["z-ai/glm-5.2", "openai/gpt-oss-120b", "deepseek-ai/deepseek-v4-pro"],
        "fallback": "z-ai/glm-5.2",
        "system": "Always respond in natural Korean."
    }
}


def call_nvidia(prompt: str, model: str, system: str = None) -> dict:
    """Call NVIDIA NIM API via curl."""
    if not system:
        system = "You are a helpful AI assistant."
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": 2000, "stream": False
    })
    cmd = ["curl", "-s", "-m", "120", "-H", f"Authorization: Bearer {NVIDIA_API_KEY}",
           "-H", "Content-Type: application/json", "-d", payload,
           f"{NVIDIA_BASE_URL}/chat/completions"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    except:
        return {"text": "API timeout", "success": False, "model": model}

    if result.returncode != 0:
        return {"text": f"curl failed: {result.stderr}", "success": False, "model": model}
    try:
        data = json.loads(result.stdout)
        return {"text": data["choices"][0]["message"]["content"], "success": True,
                "model": data.get("model", model), "usage": data.get("usage", {})}
    except Exception as e:
        return {"text": f"Parse error: {e}", "success": False, "model": model}


def auto_detect_role(prompt: str) -> str:
    p = prompt.lower()
    if any(kw in p for kw in ["code","function","implement","debug","python","javascript","코드","함수","구현"]):
        return "code"
    if re.search(r'[가-힣]', p):
        if any(kw in p for kw in ["분석","비교","평가","연구","아키텍처","설계","리뷰"]):
            return "expert"
        return "korean"
    if any(kw in p for kw in ["analyze","compare","contrast","evaluate","research","architecture","complex","why"]):
        return "expert"
    if any(kw in p for kw in ["vision","image","picture","photo","이미지","사진"]):
        return "vision"
    return "fast"


def load_models() -> list:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            d = json.load(f)
        return d.get("chat_models", [])
    return []


def handle_request(req: dict) -> dict:
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        return {"jsonrpc":"2.0","id":req_id,"result":{
            "protocolVersion":"2024-11-05","capabilities":{"tools":{}},
            "serverInfo":{"name":"hermes-agent-mcp","version":"2.0"}}}

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc":"2.0","id":req_id,"result":{"tools":[
            {"name":"hermes_chat","description":"Send prompt to Hermes Agent. Auto-routes to best model or specify role/model.","inputSchema":{"type":"object","properties":{
                "prompt":{"type":"string","description":"The prompt"},
                "role":{"type":"string","description":"Worker role: fast/expert/code/vision/korean (default: auto)"},
                "model":{"type":"string","description":"Explicit model ID (overrides role)"}},
                "required":["prompt"]}},
            {"name":"hermes_list_models","description":"List available NVIDIA NIM models","inputSchema":{"type":"object","properties":{}}},
            {"name":"hermes_roles","description":"Show role-to-model configurations","inputSchema":{"type":"object","properties":{}}}
        ]}}

    if method == "tools/call":
        name = params.get("name","")
        args = params.get("arguments",{})

        if name == "hermes_chat":
            prompt = args.get("prompt","")
            role = args.get("role","auto")
            model = args.get("model","")
            system_prompt = "You are a helpful AI assistant."

            if model:
                models = [model]
            elif role and role != "auto":
                rc = ROLE_MAP.get(role)
                if not rc:
                    return {"jsonrpc":"2.0","id":req_id,"result":{"content":[{"type":"text","text":f"Unknown role: {role}. Available: {list(ROLE_MAP.keys())}"}]}}
                models = rc["models"]
                system_prompt = rc["system"]
            else:
                detected = auto_detect_role(prompt)
                rc = ROLE_MAP.get(detected, ROLE_MAP["fast"])
                models = rc["models"]
                system_prompt = rc["system"]

            # Try models
            used_model = models[0] if models else "openai/gpt-oss-120b"
            result = call_nvidia(prompt, models[0], system_prompt)
            if not result["success"]:
                fallback = ROLE_MAP.get(detected if role=="auto" else role, {}).get("fallback") if not model else None
                if fallback and fallback != models[0]:
                    result = call_nvidia(prompt, fallback, system_prompt)

            return {"jsonrpc":"2.0","id":req_id,"result":{"content":[{"type":"text","text":result["text"]}]}}

        if name == "hermes_list_models":
            models = load_models()
            text = f"Available Models ({len(models)}):\n" + "\n".join(f"  • {m}" for m in sorted(models)) if models else "No models found."
            return {"jsonrpc":"2.0","id":req_id,"result":{"content":[{"type":"text","text":text}]}}

        if name == "hermes_roles":
            lines = ["Role Configurations:", "="*40]
            for rn, rc in ROLE_MAP.items():
                lines.append(f"\n[{rn}] {rc['description']}" if "description" in rc else f"\n[{rn}]")
                lines.append(f"  Models: {', '.join(rc['models'])}")
            return {"jsonrpc":"2.0","id":req_id,"result":{"content":[{"type":"text","text":"\n".join(lines)}]}}

    return {"jsonrpc":"2.0","id":req_id,"result":{}}


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            pass
        except Exception as e:
            err = {"jsonrpc":"2.0","error":{"code":-32603,"message":str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()