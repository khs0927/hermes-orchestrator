from __future__ import annotations

from pathlib import Path

from core.resource_hints import query_resource_hints


def test_resource_hints_are_disabled_without_router(monkeypatch):
    monkeypatch.delenv("HERMES_RESOURCE_ROUTER", raising=False)
    assert query_resource_hints("Ontology handoff") == []


def test_resource_hints_accept_only_bounded_read_only_candidates(tmp_path: Path, monkeypatch):
    router = tmp_path / "route.py"
    router.write_text(
        "import json\n"
        "print(json.dumps(["
        "{'score': 9.5, 'id': 'project_contract:aec-source-to-cad-executor', "
        "'type': 'project_contract', 'name': 'aec-source-to-cad-executor', "
        "'verification_status': 'verified_in_ci'},"
        "{'score': 4.0, 'id': 'project:khs0927/power-cad-mcp', "
        "'type': 'project', 'name': 'khs0927/power-cad-mcp'}"
        "]))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RESOURCE_ROUTER", str(router))

    hints = query_resource_hints("Ontology handoff to Power CAD", k=5)

    assert hints == [
        {
            "id": "project_contract:aec-source-to-cad-executor",
            "type": "project_contract",
            "name": "aec-source-to-cad-executor",
            "score": 9.5,
            "verification_status": "verified_in_ci",
        },
        {
            "id": "project:khs0927/power-cad-mcp",
            "type": "project",
            "name": "khs0927/power-cad-mcp",
            "score": 4.0,
        },
    ]


def test_resource_hints_fail_closed_on_invalid_json(tmp_path: Path, monkeypatch):
    router = tmp_path / "route.py"
    router.write_text("print('not-json')\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_RESOURCE_ROUTER", str(router))
    assert query_resource_hints("CAD") == []
