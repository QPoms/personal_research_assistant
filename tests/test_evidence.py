import json

import pytest

from research_assistant.research.evidence import (
    EvidenceSource,
    PersonalEvidence,
    WebEvidence,
    evidence_from_payload,
)


def test_web_evidence_preserves_url_and_raw_payload():
    payload = {
        "title": "A primary source",
        "snippet": "  Supported finding. ",
        "link": "https://example.test/paper",
        "rank": 1,
    }
    evidence = WebEvidence.from_payload(payload)

    assert evidence.source is EvidenceSource.WEB
    assert evidence.text == "Supported finding."
    assert evidence.url == "https://example.test/paper"
    assert evidence.raw == payload
    assert evidence.to_dict()["source"] == "web"


def test_personal_evidence_preserves_identity_and_spec_locator():
    payload = {
        "evidence_id": "ev-17-001",
        "content": "Integrity protection is required.",
        "document_title": "3GPP TS 24.501",
        "doc_id": "24501-h00",
        "spec_release": 17,
        "spec_version": "17.10.0",
        "section": "4.4.4",
        "source_url": "specrag://documents/24501-h00#4.4.4",
        "verified": True,
    }
    evidence = PersonalEvidence.from_payload(payload)

    assert evidence.source is EvidenceSource.PERSONAL
    assert evidence.evidence_id == "ev-17-001"
    assert evidence.release == "17"
    assert evidence.version == "17.10.0"
    assert evidence.clause == "4.4.4"
    assert evidence.raw["verified"] is True


def test_personal_evidence_parses_verified_specrag_api_fields():
    payload = {
        "evidence_id": "E001",
        "task_id": "T001",
        "chunk_id": "chunk-1",
        "specification": "3GPP TS 24.501",
        "release": 17,
        "version": "17.10.0",
        "source_path": "specs/24501-h00.pdf",
        "document_sha256": "a" * 64,
        "heading_path": "5.4 > NAS security",
        "line_start": 120,
        "line_end": 128,
        "excerpt": "Verified specification text.",
    }
    evidence = PersonalEvidence.from_payload(payload)

    assert evidence.title == "3GPP TS 24.501"
    assert evidence.task_id == "T001"
    assert evidence.chunk_id == "chunk-1"
    assert evidence.clause == "5.4 > NAS security"
    assert evidence.line_start == "120"
    assert evidence.line_end == "128"
    assert evidence.document_sha256 == "a" * 64


def test_normalized_records_are_json_serializable_even_when_raw_has_types():
    evidence = WebEvidence(
        text="Finding",
        url="https://example.test",
        raw={"source": EvidenceSource.WEB, "tags": {"a", "b"}},
    )
    encoded = json.dumps(evidence.to_dict(), sort_keys=True)
    assert '"source": "web"' in encoded


@pytest.mark.parametrize(
    ("factory", "payload", "message"),
    [
        (WebEvidence.from_payload, {"url": "https://example.test"}, "text field"),
        (WebEvidence.from_payload, {"text": "finding"}, "URL field"),
        (PersonalEvidence.from_payload, {"evidence_id": "ev-1"}, "text field"),
        (PersonalEvidence.from_payload, {"text": "finding"}, "evidence ID field"),
    ],
)
def test_incomplete_payloads_are_rejected(factory, payload, message):
    with pytest.raises(ValueError, match=message):
        factory(payload)


def test_factory_dispatches_only_declared_sources():
    evidence = evidence_from_payload(
        "personal",
        {"id": "ev-1", "quote": "A verified statement"},
    )
    assert isinstance(evidence, PersonalEvidence)
    with pytest.raises(ValueError, match="Unsupported evidence source"):
        evidence_from_payload("hybrid", {"text": "x"})


def test_payload_aliases_are_explicit_not_fuzzy():
    with pytest.raises(ValueError, match="text field"):
        PersonalEvidence.from_payload(
            {"evidence_id": "ev-1", "answer_text": "not an approved alias"}
        )
