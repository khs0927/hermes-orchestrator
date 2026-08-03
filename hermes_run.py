#!/usr/bin/env python3
"""
Hermes Agent Unified Runner (v2)
=================================
Intelligent role-based routing across all 86 NVIDIA NIM chat models.
Seamless integration with Minis, sub-agents, and orchestrator.

Usage:
    hermes-ai "prompt"                    # auto-routes to best model
    hermes-ai "prompt" --role fast        # fast extraction/summarization
    hermes-ai "prompt" --role expert      # deep analysis/reasoning
    hermes-ai "prompt" --role code        # code generation/debugging
    hermes-ai "prompt" --model z-ai/glm-5.2  # explicit model
    hermes-ai --list-models               # list all 86 models
    hermes-ai --roles                     # show role configurations
"""

import sys, os, json, subprocess, argparse, re

# ---- Configuration ----
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
HERMES_HOME     = os.environ.get("HERMES_HOME", "/root/.hermes")
CACHE_FILE      = "/var/minis/workspace/nvidia_model_cache.json"

# ---- Role-to-Model Mapping (intelligent defaults) ----
ROLE_MAP = {
    "fast": {
        "description": "빠른 추출, 요약, 일반 도구 작업",
        "models": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "meta/llama-3.1-8b-instruct",
            "mistralai/mistral-7b-instruct-v0.3"
        ],
        "fallback": "openai/gpt-oss-120b",
        "system": "You are a fast, efficient assistant. Be concise and direct. Respond in the same language as the user."
    },
    "expert": {
        "description": "심층 조사, 아키텍처, 복잡한 분석",
        "models": [
            "z-ai/glm-5.2",
            "deepseek-ai/deepseek-v4-pro",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "nvidia/nemotron-3-ultra-550b-a55b"
        ],
        "fallback": "z-ai/glm-5.2",
        "system": "You are an expert analyst and researcher. Provide thorough, well-reasoned analysis. Think step by step. Respond in the same language as the user."
    },
    "code": {
        "description": "코드 작성, 디버깅, 저장소 작업",
        "models": [
            "deepseek-ai/deepseek-v4-flash",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-ai/deepseek-v4-pro",
            "bigcode/starcoder2-15b"
        ],
        "fallback": "deepseek-ai/deepseek-v4-flash",
        "system": "You are an expert programmer. Write clean, efficient, well-documented code. Provide complete solutions. Respond in the same language as the user."
    },
    "vision": {
        "description": "이미지 분석 및 시각 작업",
        "models": [
            "meta/llama-3.2-90b-vision-instruct",
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/nemotron-nano-12b-v2-vl",
            "microsoft/phi-3-vision-128k-instruct"
        ],
        "fallback": "meta/llama-3.2-90b-vision-instruct",
        "system": "You analyze images and visual content. Be precise about what you see. Respond in the same language as the user."
    },
    "korean": {
        "description": "한국어 품질 우선",
        "models": [
            "z-ai/glm-5.2",
            "openai/gpt-oss-120b",
            "deepseek-ai/deepseek-v4-pro",
            "stepfun-ai/step-3.7-flash"
        ],
        "fallback": "z-ai/glm-5.2",
        "system": "You are a Korean-optimized assistant. Always respond in Korean. Use natural, fluent Korean expressions."
    }
}

DEFAULT_ROLE = "fast"


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {"chat_models": [], "models": []}


def get_model_list() -> list:
    cache = load_cache()
    return cache.get("chat_models", [m.get("model_id") for m in cache.get("models", [])])


def auto_detect_role(prompt: str) -> str:
    """Auto-detect the best role based on prompt content."""
    prompt_lower = prompt.lower()

    # Code detection
    code_keywords = ["code", "function", "implement", "debug", "python", "javascript",
                     "write a", "코드", "함수", "구현", "디버깅", "api"]
    if any(kw in prompt_lower for kw in code_keywords):
        return "code"

    # Korean detection
    if re.search(r'[가-힣]', prompt):
        # Check if it's a complex analysis question
        analysis_keywords = ["분석", "비교", "평가", "연구", "논문", "아키텍처",
                            "설계", "리뷰", "검토", "방법론"]
        if any(kw in prompt_lower for kw in analysis_keywords):
            return "expert"
        return "korean"

    # Expert detection
    expert_keywords = ["analyze", "compare", "contrast", "evaluate", "research",
                       "architecture", "design pattern", "complex", "why", "explain in detail"]
    if any(kw in prompt_lower for kw in expert_keywords):
        return "expert"

    # Vision detection (if file paths are mentioned)
    vision_keywords = ["vision", "image", "picture", "photo", "see", "look at",
                       "이미지", "사진"]
    if any(kw in prompt_lower for kw in vision_keywords):
        return "vision"

    return DEFAULT_ROLE


def call_nvidia(prompt: str, model: str, system: str = None, max_tokens: int = 2000,
                temperature: float = 0.7) -> dict:
    """Call NVIDIA NIM API via curl."""
    if not system:
        system = "You are a helpful AI assistant. Respond concisely and accurately in the user's language."

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    })

    cmd = [
        "curl", "-s", "-m", "120",
        "-H", f"Authorization: Bearer {NVIDIA_API_KEY}",
        "-H", "Content-Type: application/json",
        "-d", payload,
        f"{NVIDIA_BASE_URL}/chat/completions"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    except subprocess.TimeoutExpired:
        return {"error": "API timeout (120s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

    if result.returncode != 0:
        return {"error": f"curl failed: {result.stderr}", "success": False}

    try:
        data = json.loads(result.stdout)
        return {
            "success": True,
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
            "reasoning": data["choices"][0]["message"].get("reasoning_content", "")
        }
    except Exception as e:
        return {"error": f"Parse error: {e}\nRaw: {result.stdout[:300]}", "success": False}


def try_models(prompt: str, models: list, system: str, max_tokens: int, temperature: float) -> dict:
    """Try models in order until one succeeds."""
    errors = []
    for model in models:
        result = call_nvidia(prompt, model, system, max_tokens, temperature)
        if result.get("success"):
            result["model"] = model
            return result
        errors.append(f"  {model}: {result.get('error', 'unknown error')[:80]}")

    return {"error": f"All models failed:\n" + "\n".join(errors), "success": False}


def list_models():
    """List all available models with categorization."""
    models = get_model_list()
    if not models:
        print("❌ No models found. Run refresh first.")
        return []

    print(f"📋 Available NVIDIA NIM Chat Models ({len(models)})")
    print("=" * 65)

    # Group by provider
    groups = {}
    for mid in models:
        provider = mid.split("/")[0] if "/" in mid else "other"
        groups.setdefault(provider, []).append(mid)

    for provider in sorted(groups.keys()):
        mids = groups[provider]
        print(f"\n  [{provider}] ({len(mids)}):")
        for mid in mids:
            print(f"    • {mid}")

    print(f"\n{'=' * 65}")
    print("Roles: fast / expert / code / vision / korean")
    return models


def list_roles():
    """Show role configurations."""
    print("📋 Role-to-Model Mapping")
    print("=" * 65)
    for name, config in ROLE_MAP.items():
        print(f"\n  [{name}] {config['description']}")
        for m in config["models"]:
            mark = " ⭐" if m == config["fallback"] else ""
            print(f"    • {m}{mark}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Unified Runner v2")
    parser.add_argument("prompt", nargs="?", help="Prompt to send")
    parser.add_argument("--role", "-r", choices=list(ROLE_MAP.keys()) + ["auto"],
                        default="auto", help="Worker role (auto=detect)")
    parser.add_argument("--model", "-m", help="Explicit model ID (overrides role)")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--list-models", action="store_true", help="List all models")
    parser.add_argument("--roles", action="store_true", help="Show role configs")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return 0

    if args.roles:
        list_roles()
        return 0

    if not args.prompt:
        print("Usage:")
        print("  hermes-ai \"prompt\"")
        print("  hermes-ai --list-models")
        print("  hermes-ai --roles")
        return 0

    if not NVIDIA_API_KEY:
        print("❌ NVIDIA_API_KEY not set")
        return 1

    # --- Determine role & model ---
    if args.model:
        # Explicit model
        models = [args.model]
        role_config = {"system": "You are a helpful AI assistant. Respond in the user's language."}
        role_name = "custom"
    else:
        role_name = args.role if args.role != "auto" else auto_detect_role(args.prompt)
        role_config = ROLE_MAP.get(role_name, ROLE_MAP["fast"])
        models = role_config["models"]

    system_prompt = role_config["system"]
    fallback = role_config.get("fallback", models[0] if models else "")

    if args.verbose:
        print(f"🤖 [{role_name.upper()}] → {models[0]}", file=sys.stderr)
        print(f"📝 {args.prompt[:100]}{'...' if len(args.prompt)>100 else ''}", file=sys.stderr)

    # --- Execute ---
    result = try_models(args.prompt, models, system_prompt, args.max_tokens, args.temperature)

    if not result.get("success"):
        # Try fallback
        if fallback and fallback not in models:
            if args.verbose:
                print(f"⚠️ Primary failed, trying fallback: {fallback}", file=sys.stderr)
            result = call_nvidia(args.prompt, fallback, system_prompt, args.max_tokens, args.temperature)

    if not result.get("success"):
        print(f"❌ Error: {result.get('error')}")
        return 1

    if args.json:
        output = {
            "content": result["content"],
            "model": result.get("model", ""),
            "role": role_name,
            "usage": result.get("usage", {})
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        print(result["content"])

        if args.verbose:
            usage = result.get("usage", {})
            model_used = result.get("model", args.model or models[0])
            print(f"\n📊 [{role_name}] Model: {model_used}", file=sys.stderr)
            print(f"📊 Tokens: {usage.get('total_tokens', '?')}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())