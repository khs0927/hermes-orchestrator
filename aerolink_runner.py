#!/usr/bin/env python3
"""
AeroLink API Runner for Hermes Multi-Backend Integration
=========================================================
Bridges AeroLink API gateway with Hermes Agent routing system.

AeroLink Notes:
- Base URL: https://cgapi.aerolink.lat
- Auth: Bearer token via $AEROLINK_API_KEY
- Models use plain IDs (no provider/ prefix): gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra
- Supports OpenAI-compatible chat completions endpoint

Usage as Hermes backend:
    /root/.hermes/aerolink_runner.py --prompt "..." [--model gpt-5.6-luna]
"""

import sys, os, json, subprocess, argparse

AEROLINK_BASE = "https://cgapi.aerolink.lat/v1"
AEROLINK_KEY   = os.environ.get("AEROLINK_API_KEY", "")

# Verified AeroLink model catalog
AEROLINK_MODELS = ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"]

DEFAULT_MODEL = "gpt-5.6-luna"


def call_aerolink(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
    """Call AeroLink API via curl."""
    if not AEROLINK_KEY:
        return {"error": "AEROLINK_API_KEY not set", "success": False}

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Hermes Agent, a helpful AI assistant. Respond concisely in the user's language."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    })

    cmd = [
        "curl", "-s", "-m", "120",
        "-H", f"Authorization: Bearer {AEROLINK_KEY}",
        "-H", "Content-Type: application/json",
        "-d", payload,
        f"{AEROLINK_BASE}/chat/completions"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=130)
    except subprocess.TimeoutExpired:
        return {"error": "API timeout", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

    if result.returncode != 0:
        return {"error": f"curl failed: {result.stderr}", "success": False}

    try:
        data = json.loads(result.stdout)
        if "error" in data:
            return {"error": f"API error: {data['error']}", "success": False}
        return {
            "success": True,
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "usage": data.get("usage", {})
        }
    except Exception as e:
        return {"error": f"Parse error: {e}\nRaw: {result.stdout[:300]}", "success": False}


def main():
    parser = argparse.ArgumentParser(description="AeroLink API Runner")
    parser.add_argument("prompt", nargs="?", help="Prompt to send")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--list-models", action="store_true", help="List AeroLink models")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print(f"AeroLink Models ({len(AEROLINK_MODELS)}):")
        for m in AEROLINK_MODELS:
            print(f"  • {m}")
        return 0

    if not args.prompt:
        print("Usage: aerolink_runner.py \"prompt\" [--model MODEL]")
        print("       aerolink_runner.py --list-models")
        return 1

    if args.verbose:
        print(f"🌐 AeroLink → {args.model}", file=sys.stderr)

    result = call_aerolink(args.prompt, args.model, args.max_tokens, args.temperature)

    if not result.get("success"):
        print(f"❌ Error: {result.get('error')}")
        return 1

    print(result["content"])

    if args.verbose:
        usage = result.get("usage", {})
        print(f"\n📊 Model: {result.get('model')} | Tokens: {usage.get('total_tokens', '?')}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())