# Hermes Orchestrator — Multi-Model Sub-Agent Engine for Korean NLP

**A natural-language disassembler → parallel worker dispatch → Claude Opus final verification pipeline optimized for Korean/English multilingual environments.**

Runs on iOS (Minis + iSH) or any Linux host. 87 models, 5 ensemble strategies, 3-step recovery policy (NVIDIA workers → Luna recovery → Opus verification).

---

## "Why This Exists"

LLMs are strongest when given typed, scoped, well-defined sub-tasks — not when prompted with a long, unstructured request. The Hermes Orchestrator turns one natural-language sentence into several parallel `hermes_chat` calls:

```
"이 프로젝트를 GitHub에 올리고 문서도 쓰고 자동으로 배포도 하고 싶어."

┌─ planner: step 1: `github-sync` → step 2: `document-generator` → step 3: `deploy`
├─ sys-agent: git clone, branch create, push (iOS native)
├─ coder-agent: generate docs with type hints (Python)
├─ fetch-agent: scrape README template + commit log → include in docs
└─ reviewer: checks conflicts, `deploy` target URL verified, `github` push confirmed → final Korean report
```

The worker chain is fully parallel. The reviewer gates the output. Only "모든 과정" or multi-agent · `explicit` requests trigger this; regular quick commands (`hermes "weather"`) forward directly to NVIDIA without overhead.

---

## 5 Workers

| Worker | Role Purpose | Default NVIDIA Model |
|--------|------|---------|
| `planner` | Request dissassembly, dependencies, ordering | `openai/gpt-oss-120b` |
| `sys-agent` | iOS/Minis native functions — device, files, env | `openai/gpt-oss-20b` / `meta/llama-3.1-8b` |
| `coder-agent` | Write, modify, run, debug code | `deepseek-ai/deepseek-v4-flash` / `pro` |
| `fetch-agent` | Web, browser, external archive, cite | `openai/gpt-oss-120b` / `meta/llama-3.1-70b` |
| `reviewer` | Independent fact-check, synthesis in Korean | `z-ai/glm-5.2` / `deepseek-ai/deepseek-v4-pro` |

The `sys-agent` connects to apple-* commands:
```bash
$ hermes "오늘 걸은 거리와 심박수 데이터를 가져와서 분석해줘"
planner 분해 → sys-agent: "📊 healthKit healthkit batch --types step_count,heart_rate --days 1"…
fetching → result sent to coder-agent for processing → visual chart generation → final reviewer → final output
```

---

## Recovery & Verification Policy

### Luna First (AeroLink GPT-5.6)

If any worker fails, the result is redirected through Luna's recovery before marking failed:

```
Worker FAIL → Luna recovery worker
  ├─ successful
  │   └→ passes to ⤵
  └─ dead ──→ jump to ⤴
           ⤵ Claude Opus final verification
```

### Claude Opus (AeroLink Claude) Gate

```
NVIDIA results + Luna recovery → Claude Opus verdict chain
├─ factuality
├─ omissions
├─ model/provider mismatch
├─ execution completeness (external action verified?)
├─ conflict or placeholder reporting
└─ Final: "검증 완료" or "검증 필요"
```

If Opus returns `need_review`, the orchestrator loops (NOT silently marking pass):

```
retry → repair (code fix, re-fetch, conflict recheck) → Opus again → pass?
```

---

## 5 Ensemble Strategies

| Strategy | When to use |
|----------|-------------|
| `FAST_TO_EXPERT_TO_MERGE` | routine task: fast answer → expert review → merge |
| `INDEPENDENT_EXPERT_PANEL` | complex analysis, law/medical, high-stakes |
| `CODER_TO_ARCHITECT_TO_TEST` | code: fast coder → architect review → test pass |
| `LONG_CONTEXT_PANEL` | multilingual, extraction, articles, law |
| `ADVERSARIAL_REVIEW` | solutions vs. safety critique → revision |

The strategy is auto-selected based on:
1. User explicitly requests strategy
2. Prompt keywords (code, arch, multi-document, safety)
3. Default: `FAST_TO_EXPERT_TO_MERGE`

---

## CLI

```bash
# Quick: plain prompt → auto-route
hermes "오늘 날씨 어때?"

# Explicit multi-agent (detected keyword + Korean)
hermes "모든 과정을 진행해줘"

# Use orchestrator directly
hermes-ai "복잡한 한글 텍스트 분석" --role expert

# With model
hermes-ai "SwiftUI view design proposal" --model z-ai/glm-5.2

# List supported models
hermes-ai --list-models

# Show role configuration
hermes-ai --roles
```

---

## Architecture Map

```
$ hermes "한국어 요청"
→ python3 /root/.hermes/hermes_orchestrator.py
─────────────────────────────
plan_roles()
   ├→ if "서브 에이전트" pattern:
   │    planner + sys-agent + coder-agent + fetch-agent (parallel)
   │   else:
   │    check key-words (코드, 검색, 기기, 파일)
   └→ Dispatch

call_model()     ← /root/.hermes/hermes_run.py
   ├→ curl to NVIDIA NIM (OpenAI-compatible)
   │
FALLBACK: Luna recovery
   └→ curl to AeroLink GPT-5.6 / Luna

VERIFY: Opus via OPUS
   └→ curl to AeroLink Claude / claude-opus-5
```

---

## File Map

```
/root/.hermes/hermes_orchestrator.py       Korean sub-agent dispatcher
/root/.hermes/hermes_orchestrator_mcp.py  MCP v3 server wrapper
/root/.hermes/hermes_run.py             NVIDIA direct runner
/root/.hermes/config.yaml               Model + skill configuration
/root/.hermes/hermes_mcp_server.py      MCP server stdio wrapper
/usr/local/bin/hermes                   Orchestrator entry point (plain English)
/usr/local/bin/hermes-ai                Model-aware entry (role / model flags)
/usr/local/bin/hermes-mcp               MCP server entry
```

---

## Configuration (config.yaml)

```yaml
model:
  provider: nvidia
  default: nvidia/nemotron-3-super-120b-a12b
nvidia:
  api_key_env: NVIDIA_API_KEY
  base_url: https://integrate.api.nvidia.com/v1
skills:
  external_dirs: ['/var/minis/skills']
  template_vars: true
  inline_shell: false
mcp_servers:
  khs0927:
    url: https://mcp.smithery.run/khs0927
    headers:
      Authorization: Bearer $$SMITHERY_API_KEY
```

---

## Supported Models

| Provider | Count | Model families |
|----------|-------|----------------|
| NVIDIA NIM | 86 chat + 16 specialist | GPT-OSS, Nemotron, Llama, DeepSeek, Mistral, GLM, Phi, Qwen, StarCoder, Gemma, Cohere, IBM... |
| AeroLink GPT-5.6 | 3 | Luna, Sol, Terra |
| AeroLink Claude | 2 | Opus‑5, Opus‑4‑8 |

All models with auto-fallback for 410/EOL, 429 rate-limit backoff.

---

## Security

- API keys: env-var only (`NVIDIA_API_KEY`, `AEROLINK_API_KEY`, `SMITHERY_API_KEY`)
- No secret in output, trace, or memories
- OAuth pins to native app-based authentication or redirect URI (no file copy)

## License

MIT

---

**Related project**: [Hermes-Minis](https://github.com/user/hermes-minis) — full Hermes Agentelei on iOS with Telegram & dashboard.