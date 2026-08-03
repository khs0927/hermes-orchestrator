"""
executor.py — 모델 호출 및 자동 폴백 실행기
"""
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List
from .model_registry import ModelRegistry

WORKSPACE = Path("/var/minis/workspace")


class Executor:
    """단일 프롬프트에 대해 폴백 체인 전체를 시도하는 실행기"""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.attempt_log: List[dict] = []

    def execute(
        self,
        prompt: str,
        role: str = "fast",
        max_tokens: int = 256,
        temperature: float = 0.7,
        prefer_model: Optional[str] = None,
        timeout: int = 60,
    ) -> dict:
        """
        주어진 역할의 폴백 체인 전체를 순차 시도.
        우선 prefer_model이 있으면 맨 앞에 배치.
        """
        self.attempt_log = []
        chain = self.registry.chain_for(role)

        if prefer_model and prefer_model in chain:
            chain = [prefer_model] + [m for m in chain if m != prefer_model]
        elif prefer_model and prefer_model not in chain and prefer_model in self.registry.models:
            chain = [prefer_model] + chain

        if not chain:
            return {"success": False, "error": f"'{role}' 역할에 사용 가능한 모델이 없습니다."}

        for idx, model_id in enumerate(chain):
            t0 = time.time()
            self.attempt_log.append(
                {"idx": idx, "model": model_id, "role": role}
            )

            try:
                output, elapsed = self._call_model(
                    model_id, prompt, max_tokens, temperature, timeout
                )
                self.registry.mark_success(model_id, int(elapsed * 1000))
                self.attempt_log[-1].update(
                    {"status": "ok", "latency_ms": int(elapsed * 1000)}
                )
                return {
                    "success": True,
                    "model": model_id,
                    "role": role,
                    "output": output,
                    "latency_ms": int(elapsed * 1000),
                    "attempts": idx + 1,
                    "total_chain": len(chain),
                    "attempt_log": self.attempt_log,
                }
            except TimeoutError as e:
                self.registry.mark_error(model_id, "timeout")
                self.attempt_log[-1].update({"status": "timeout", "error": str(e)[:100]})
                continue
            except _ModelGoneError as e:
                # EOL (410) — 즉시 dead 표시
                self.registry.mark_error(model_id, "eol_410")
                self.attempt_log[-1].update({"status": "eol", "error": str(e)[:100]})
                continue
            except _ModelBadRequestError as e:
                self.registry.mark_error(model_id, f"bad_request: {str(e)[:50]}")
                self.attempt_log[-1].update({"status": "bad_request", "error": str(e)[:100]})
                continue
            except Exception as e:
                self.registry.mark_error(model_id, f"unknown: {str(e)[:50]}")
                self.attempt_log[-1].update({"status": "error", "error": str(e)[:100]})
                continue

        return {
            "success": False,
            "error": f"'{role}' 체인의 {len(chain)}개 모델 모두 실패",
            "attempt_log": self.attempt_log,
        }

    def _call_model(
        self, model_id: str, prompt: str, max_tokens: int,
        temperature: float, timeout: int
    ) -> Tuple[str, float]:
        """실제 minis-model-use 호출 — stdout/파싱 모두 처리"""
        tmp = WORKSPACE / f"_tmp_hermes_{int(time.time()*1000)}.json"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False))

        t0 = time.time()
        try:
            proc = subprocess.run(
                ["minis-model-use", "run", "--model", model_id, "--input", str(tmp)],
                capture_output=True, text=True, timeout=timeout,
            )
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

        elapsed = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"명령 실패 (rc={proc.returncode}): {proc.stderr[:200]}")

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON 파싱 실패: {e}")

        if not data.get("ok"):
            err = data.get("error", {})
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", 0) if isinstance(err, dict) else 0
            if "410" in str(msg) or "Gone" in str(msg) or "end of life" in str(msg).lower():
                raise _ModelGoneError(msg)
            if code == 400 or "400" in str(msg):
                raise _ModelBadRequestError(msg)
            raise RuntimeError(msg)

        output_text = data.get("data", {}).get("output_text", "")
        if not output_text:
            raise RuntimeError("빈 응답")

        return output_text, elapsed


class _ModelGoneError(Exception):
    """모델 EOL (410 Gone) — 재시도 불필요"""


class _ModelBadRequestError(Exception):
    """잘못된 요청 (400) — 프롬프트 문제 가능"""


if __name__ == "__main__":
    # 간단 테스트
    exec = Executor()
    print(f"🔗 풀 모델 수: {len(exec.registry.models)}")
    for role, chain in exec.registry.fallback_chains.items():
        print(f"  {role:10s}: {len(chain)}개")

    print("\n🧪 빠른 응답 테스트:")
    r = exec.execute(
        "한국어로 '연결 성공'이라고만 답하세요.", role="fast", max_tokens=20
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))