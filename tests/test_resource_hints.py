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
        "'verification_status': 'verified_in_ci', "
        "'verified_consumers': ['khs0927/power-cad-mcp'], "
        "'pending_consumers': [], 'real_cad_e2e': False},"
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
            "verified_consumers": ["khs0927/power-cad-mcp"],
            "pending_consumers": [],
            "real_cad_e2e": False,
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


def test_resource_hints_bound_untrusted_consumer_lists(tmp_path: Path, monkeypatch):
    router = tmp_path / "route.py"
    router.write_text(
        "import json\n"
        "print(json.dumps([{'score': 5, 'id': 'project_contract:x', "
        "'type': 'project_contract', 'name': 'x', "
        "'verified_consumers': ['ok', 3, '', 'x' * 500], "
        "'pending_consumers': 'not-a-list', 'real_cad_e2e': 'yes'}]))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RESOURCE_ROUTER", str(router))

    hints = query_resource_hints("x", k=5)

    assert hints == [
        {
            "id": "project_contract:x",
            "type": "project_contract",
            "name": "x",
            "score": 5,
            "verified_consumers": ["ok"],
            "pending_consumers": [],
            "real_cad_e2e": False,
        }
    ]
