"""
synthesis.py — 여러 모델의 출력을 종합
"""
from typing import List, Dict


class Synthesis:
    """다중 모델 결과를 합치는 간단한 종합기"""

    @staticmethod
    def merge_unique(results: List[Dict], field: str = "output") -> List[str]:
        """중복 제거된 결과 모음"""
        seen = set()
        out = []
        for r in results:
            val = r.get(field, "")
            if val and val not in seen:
                seen.add(val)
                out.append(val)
        return out

    @staticmethod
    def vote(results: List[Dict], field: str = "output") -> Dict:
        """
        빈도 기반 투표 — 가장 많이 나온 응답이 winner
        """
        if not results:
            return {"winner": "", "votes": {}, "total": 0}

        from collections import Counter
        outputs = [r.get(field, "") for r in results if r.get(field)]
        cnt = Counter(outputs)
        total = len(outputs)
        top = cnt.most_common(1)[0] if cnt else ("", 0)
        return {
            "winner": top[0],
            "winner_votes": top[1],
            "winner_ratio": round(top[1] / total, 2) if total else 0,
            "votes": dict(cnt),
            "total": total,
            "unique_outputs": len(cnt),
            "agreement_level": "high" if top[1] / total >= 0.6 else
                              "medium" if top[1] / total >= 0.4 else "low",
        }

    @staticmethod
    def best_by_score(results: List[Dict]) -> Dict:
        """latency_ms 짧고 true인 것 우선"""
        valid = [r for r in results if r.get("success")]
        if not valid:
            return {}
        return min(valid, key=lambda r: r.get("latency_ms", 1e9))

    @staticmethod
    def report(results: List[Dict], strategy: str = "vote") -> Dict:
        """전략에 따라 종합"""
        if strategy == "vote":
            return Synthesis.vote(results)
        elif strategy == "best":
            return Synthesis.best_by_score(results)
        else:
            return {"merged": Synthesis.merge_unique(results), "count": len(results)}