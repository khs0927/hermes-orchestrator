"""
task_router.py — sentence analysis to determine role & model chain
"""
import re
from typing import Tuple, List


class TaskRouter:
    """Keyword-based prompt classifier → role selector"""

    KEYWORDS = {
        "code": [
            r"\b(code|코드|function|함수|class|클래스|swift|swiftui|kotlin|java|python|javascript|typescript|api|bug|버그|compile|컴파일|error|오류|stack|stacktrace|sql|database|git|commit|merge|diff|refactor|리팩토링)\b",
            r"(작성해줘|구현해줘|수정해줘|디버깅|분석해줘|리뷰해줘|코드를)",
        ],
        "vision": [
            r"\b(image|이미지|picture|사진|screenshot|스크린샷|ocr|diagram|다이어그램|chart|차트|graph|그래프)\b",
        ],
        "translate": [
            r"\b(translate|번역|번역해줘|영어로|한국어로|일본어로|중국어로|스페인어로)\b",
            r"(translate|convert to|change language)",
        ],
        "expert": [
            r"\b(architecture|아키텍처|design|설계|analyze|분석|deep|심층|long.?context|장기|complex|복잡|plan|계획|strategy|전략|investigate|조사|document|문서)\b",
            r"(심층|깊이|자세히|전문가|expert|expert.?level)",
        ],
        "safety": [
            r"\b(safety|안전|toxic|harmful|jailbreak|injection|filter|audit|content.?policy|review|검토)\b",
            r"(안전성|검토|감사)",
        ],
    }

    DEFAULT_ROLE = "fast"

    def classify(self, prompt: str) -> Tuple[str, float, List[str]]:
        """
        (role, confidence, matched keywords)
        """
        if not prompt or not prompt.strip():
            return self.DEFAULT_ROLE, 0.0, []

        text = prompt.lower()
        scores = {}
        matches = {}

        for role, patterns in self.KEYWORDS.items():
            score = 0
            hit = []
            for p in patterns:
                found = re.findall(p, text, flags=re.IGNORECASE)
                if found:
                    score += len(found)
                    hit.extend(found if isinstance(found[0], str) else [f[0] for f in found])
            if score > 0:
                scores[role] = score
                matches[role] = hit

        if not scores:
            return self.DEFAULT_ROLE, 0.0, []

        best = max(scores.items(), key=lambda x: x[1])
        role, raw_score = best
        confidence = min(raw_score / 3.0, 1.0)

        return role, confidence, matches.get(role, [])

    def route(self, prompt: str) -> dict:
        role, confidence, hits = self.classify(prompt)
        return {
            "role": role,
            "confidence": round(confidence, 2),
            "matched_keywords": list(set(hits))[:10],
            "reasoning": self._explain(role, confidence),
        }

    def _explain(self, role, confidence):
        explanations = {
            "code": "코드/프로그래밍 관련 키워드 감지 — 코드 전문 모델 사용",
            "vision": "이미지/시각 콘텐츠 처리 — 멀티모달 모델 사용",
            "translate": "번역 작업 감지 — 번역 특화 모델 사용",
            "expert": "심층 분석/아키텍처 — 대형 추론 모델 사용",
            "safety": "안전성 검토 — 안전 특화 모델 사용",
            "fast": "일반 작업 — 빠른 경량 모델 사용",
        }
        return explanations.get(role, "")