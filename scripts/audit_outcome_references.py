"""Compare named penalty references with an independently retrieved FIFA PDF.

The audit checks the official event record. It does not establish that a video
cut depicts that event, create human labels, or evaluate a video model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
LEGACY_URL = "https://img.fifa.com/image/upload/evdvpfdkueqrdlbbrrus.pdf"
PDF_URL = "https://digitalhub.fifa.com/m/649e84967b086928/original/evdvpfdkueqrdlbbrrus-pdf.pdf"
MATCHES = {
    "col_eng": {"page": 134, "table_index": 0, "match_number": 56,
                "score_order": ["col", "eng"], "first_taker": "col", "attempts": 10},
    "cro_den": {"page": 132, "table_index": 1, "match_number": 52,
                "score_order": ["cro", "den"], "first_taker": "den", "attempts": 10},
    "esp_rus": {"page": 132, "table_index": 0, "match_number": 51,
                "score_order": ["esp", "rus"], "first_taker": "esp", "attempts": 9},
    "rus_cro": {"page": 136, "table_index": 0, "match_number": 59,
                "score_order": ["rus", "cro"], "first_taker": "rus", "attempts": 10},
}
DETAIL_SOURCES = {
    "cro_den": "https://inside.fifa.com/tournaments/mens/worldcup/2018russia/news/schmeichel-s-hat-trick-not-enough",
    "esp_rus": "https://inside.fifa.com/tournaments/mens/worldcup/2018russia/news/russia-win-first-ever-world-cup-shootout-to-reach-quarters-2980138",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_tables(document) -> dict[str, list[dict]]:
    tables = {}
    pattern = re.compile(r"(\d+)-(\d+)\s+([A-Za-z][A-Za-z\s-]*?)\s*\((\d+)\)")
    for match_id, metadata in MATCHES.items():
        text = document[metadata["page"] - 1].get_text()
        segments = text.split("Penalty Shoot-out:")[1:]
        section = segments[metadata["table_index"]].split("Referee:")[0]
        rows = [{"score": [int(a), int(b)], "name": " ".join(name.split()), "shirt": int(shirt)}
                for a, b, name, shirt in pattern.findall(section)]
        if len(rows) != metadata["attempts"]:
            raise ValueError(f"Unexpected FIFA table membership: {match_id}")
        previous = [0, 0]
        for ordinal, row in enumerate(rows, 1):
            first = metadata["first_taker"]
            country = first if ordinal % 2 else next(c for c in metadata["score_order"] if c != first)
            side = metadata["score_order"].index(country)
            changes = [a - b for a, b in zip(row["score"], previous)]
            if changes[side] not in (0, 1) or changes[1 - side] != 0:
                raise ValueError("Official score progression does not support alternating takers")
            row.update(ordinal=ordinal, country=country, previous_score=previous,
                       outcome="goal" if changes[side] else "no_goal")
            previous = row["score"]
        tables[match_id] = rows
    return tables


def detail_support(reference: dict, match_id: str) -> tuple[str, str | None]:
    if reference["outcome"] == "goal" and reference["outcome_detail"] == "scored":
        return "supported_by_official_score_increment", PDF_URL
    if reference["clip_id"] == "col_eng_p06_henderson_eng8" and reference["outcome_detail"] == "saved":
        return "supported_by_individual_pdf_narrative", PDF_URL
    if match_id in DETAIL_SOURCES and reference["outcome_detail"] == "saved":
        return "supported_by_individually_named_official_fifa_article", DETAIL_SOURCES[match_id]
    return "unknown_not_individually_established_in_checked_sources", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-pdf", type=Path, required=True)
    parser.add_argument("--legacy-html-response", type=Path)
    parser.add_argument("--references", type=Path, default=ROOT / "data/annotations/penalty_outcomes.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs/OUTCOME_REFERENCE_AUDIT_2026-10-12.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/OUTCOME_REFERENCE_AUDIT_2026-10-12.md")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "outputs/reference_source_audit_2026-10-12")
    args = parser.parse_args()
    if args.official_pdf.read_bytes()[:5] != b"%PDF-":
        raise ValueError("The official source is not a PDF; an HTML redirect is not evidence")
    document = pymupdf.open(args.official_pdf)
    if len(document) != 140 or document.metadata.get("author") != "FIFA":
        raise ValueError("Unexpected official technical report identity")
    tables = parse_tables(document)
    with args.references.open(newline="", encoding="utf-8") as stream:
        references = list(csv.DictReader(stream))
    source_manifest = ROOT / "data/course_context_clips_2026-10-12/source_manifest.json"
    expected = {row["clip_id"] for row in json.loads(source_manifest.read_text())["clips"]}
    if len(references) != 39 or len(expected) != 39 or {row["clip_id"] for row in references} != expected:
        raise ValueError("Outcome references and canonical source manifest must describe exactly 39 events")
    records = []
    clip_pattern = re.compile(r"([a-z]+_[a-z]+)_p(\d+)_(.+)_([a-z]{3})(\d+)")
    for reference in references:
        match_id, ordinal, player, country, shirt = clip_pattern.fullmatch(reference["clip_id"]).groups()
        event = tables[match_id][int(ordinal) - 1]
        normalized_name = re.sub(r"[^a-z]", "", event["name"].lower())
        normalized_player = {"kuzyaev": "kuziaev", "fernandes": "mariofernandes"}.get(player, player)
        named_event_matches = (event["country"] == country and event["shirt"] == int(shirt)
                               and normalized_name.startswith(normalized_player))
        binary_matches = named_event_matches and reference["outcome"] == event["outcome"]
        detail_status, detail_url = detail_support(reference, match_id)
        if not binary_matches:
            detail_status, detail_url = "unknown_binary_reference_or_identity_conflict", None
        notes = []
        if reference["clip_id"] == "cro_den_p09_jorgensen_den9":
            notes.append("The PDF table says JORGENSEN Mathias (9), but its same-page substitution record names Nicolai (9). FIFA's named-save article also identifies Nicolai. The binary outcome uses ordinal 9, Denmark and shirt 9; the original clip ID supplies no first name.")
        if player == "kuzyaev":
            notes.append("The PDF spells the Russia number 7 name KUZIAEV; the clip ID uses the transliteration kuzyaev.")
        records.append({"clip_id": reference["clip_id"], "match_id": match_id,
            "reference_outcome": reference["outcome"], "reference_outcome_detail": reference["outcome_detail"],
            "source_accessible": True, "legacy_source_pdf_accessible": False,
            "reference_supported": binary_matches, "individual_checked": True,
            "check_type": "official_named_event_score_progression_audit",
            "human_verified": False, "clip_to_event_video_identity_verified": False,
            "official_record": {"match_number": MATCHES[match_id]["match_number"],
                "pdf_page_1_based": MATCHES[match_id]["page"], "printed_page": MATCHES[match_id]["page"],
                "attempt_ordinal": event["ordinal"], "player_name_as_printed": event["name"],
                "shirt_number": event["shirt"], "country": event["country"],
                "score_team_order": MATCHES[match_id]["score_order"],
                "score_before": event["previous_score"], "score_after": event["score"],
                "binary_outcome_from_score_change": event["outcome"], "url": PDF_URL},
            "outcome_detail_support_status": detail_status, "outcome_detail_source_url": detail_url,
            "notes": notes})
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = args.evidence_dir / "FIFA_2018_Technical_Report_official.pdf"
    shutil.copyfile(args.official_pdf, saved_pdf)
    for page in (132, 134, 136):
        document[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(args.evidence_dir / f"page_{page}.png")
    audit = {"schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor": "AI assistant comparing independently fetched primary-source records",
        "scope": "Named-event binary outcome reference audit, separate from video-model accuracy and human annotation",
        "references": {"path": str(args.references.relative_to(ROOT)), "sha256": digest(args.references)},
        "source_manifest_sha256": digest(source_manifest), "audit_script_sha256": digest(Path(__file__)),
        "official_pdf": {"url": PDF_URL, "source_accessible": True, "access_method": "HTTPS GET using curl; decoded as PDF",
            "path": str(saved_pdf.relative_to(ROOT)), "sha256": digest(saved_pdf), "bytes": saved_pdf.stat().st_size,
            "pages": len(document), "title": document.metadata["title"], "author": document.metadata["author"],
            "checked_pages_1_based": [132, 134, 136]},
        "legacy_url": {"url": LEGACY_URL, "pdf_source_accessible": False, "http_response_accessible": True,
            "observed_redirect_target": "https://inside.fifa.com/official-documents", "returned_content_type": "text/html",
            "response_sha256": digest(args.legacy_html_response) if args.legacy_html_response else None},
        "supplementary_official_articles_checked": [
            {"url": DETAIL_SOURCES["cro_den"], "source_accessible": True,
             "checked_named_facts": "Badelj and Pivaric saved by Schmeichel; Eriksen, Schone and Nicolai Jorgensen saved by Subasic"},
            {"url": DETAIL_SOURCES["esp_rus"], "source_accessible": True,
             "checked_named_facts": "Akinfeev saved Koke and Iago Aspas in the shootout"}],
        "summary": {"events": len(records), "source_accessible": sum(r["source_accessible"] for r in records),
            "reference_supported": sum(r["reference_supported"] for r in records),
            "individual_checked": sum(r["individual_checked"] for r in records),
            "outcome_counts": dict(Counter(r["reference_outcome"] for r in records)),
            "outcome_detail_not_individually_supported": [r["clip_id"] for r in records
                if r["outcome_detail_source_url"] is None],
            "human_verified_events": 0, "video_event_identities_verified": 0},
        "held_out_test": False, "accuracy_validated": False, "labels_modified": False,
        "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = audit["summary"]["outcome_counts"]
    supported = audit["summary"]["reference_supported"]
    lines = ["# Outcome Reference Audit (12 October 2026)", "",
        "This is an independent source-document audit by an AI assistant. It checks named penalty-event outcomes against FIFA's official report. It is not a new human video annotation, does not verify the identity of every video cut, and does not validate model accuracy or an unseen test set. No labels were modified.", "",
        f"The annotation table contains 39 distinct events. {supported}/39 binary references agree with official score progression. The stored table has {counts.get('goal', 0)} goals and {counts.get('no_goal', 0)} non-goals. The official PDF was retrieved from [FIFA Digital Hub]({PDF_URL}), contains 140 pages and has SHA-256 `{audit['official_pdf']['sha256']}` ({audit['official_pdf']['bytes']:,} bytes). PDF page numbers and printed page numbers coincide.", "",
        f"The [stored legacy URL]({LEGACY_URL}) currently redirects to FIFA's document directory and returns HTML. That response is not the technical-report PDF. The independently retrieved Digital Hub file provides the checked primary-source evidence.", "",
        "| Match | FIFA Match | Checked Page | Events | Goals | Non-goals |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for match_id, metadata in MATCHES.items():
        match_records = [r for r in records if r["match_id"] == match_id]
        goals = sum(r["reference_outcome"] == "goal" for r in match_records)
        lines.append(f"| {match_id} | {metadata['match_number']} | {metadata['page']} | {len(match_records)} | {goals} | {len(match_records)-goals} |")
    lines += ["", "The audit parses each penalty table, starts the shootout at 0-0, and checks whether the named taker's team total increases by one. It also checks event order, country, shirt number and the clip-ID surname. Source accessibility, reference support and individual comparison are separate JSON fields. The comparison concerns the official event represented by an ID; it does not prove that all source cuts show the intended event.", "",
        "The report's Denmark table prints `JORGENSEN Mathias (9)` for attempt 9. Its same-page substitution record instead identifies Nicolai Jorgensen as number 9. FIFA's [named-save account](" + DETAIL_SOURCES["cro_den"] + ") also identifies Nicolai. The audit records this source-name inconsistency and checks Denmark, shirt 9 and attempt order without changing the clip ID or outcome. `KUZIAEV` and `kuzyaev` are recorded as a transliteration difference.", "",
        "A binary non-goal does not by itself establish `saved`, `crossbar` or `wide`. The PDF explicitly describes Henderson's save. FIFA's named-save account supports the five Denmark-Croatia saves; its [Spain-Russia account](" + DETAIL_SOURCES["esp_rus"] + ") names the saves of Koke and Aspas. The checked sources do not individually establish the stored detailed mechanism for the following five events: Uribe (`crossbar`), Bacca (`saved`), Russia-Croatia Smolov (`saved`), Kovacic (`saved`) and Fernandes (`wide`). Their binary outcomes are supported; their detailed mechanisms remain unknown in this audit.", "",
        "| Event | Page | Official Score Before To After | Stored Outcome | Accessible | Supported | Individually Checked |",
        "| --- | ---: | --- | --- | --- | --- | --- |"]
    for r in records:
        event = r["official_record"]
        before, after = event["score_before"], event["score_after"]
        lines.append(f"| {r['clip_id']} | {event['pdf_page_1_based']} | {before[0]}-{before[1]} to {after[0]}-{after[1]} | {r['reference_outcome']} | yes | {'yes' if r['reference_supported'] else 'no'} | yes |")
    lines += ["", "The authoritative machine-readable record is [OUTCOME_REFERENCE_AUDIT_2026-10-12.json](OUTCOME_REFERENCE_AUDIT_2026-10-12.json). The retrieved PDF and three rendered evidence pages are retained in `outputs/reference_source_audit_2026-10-12/`. Reproduce the binary comparison with `scripts/audit_outcome_references.py --official-pdf <retrieved-official-PDF>`; supplementary article findings were independently read during this audit and remain explicitly identified as source-reading records.", ""]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()
