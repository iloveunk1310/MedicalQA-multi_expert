"""Contextual prefix and section title mappings per document type."""

import re
from typing import Any, Mapping, Optional

from .normalize import normalize_unicode

DRUG_SECTION_NAMES: dict[str, str] = {
    "composition_form": "Dạng thuốc và hàm lượng",
    "pharmacology": "Dược lý và cơ chế tác dụng",
    "indication": "Chỉ định",
    "contraindication": "Chống chỉ định",
    "precaution_warning": "Thận trọng",
    "pregnancy_lactation": "Thời kỳ mang thai và cho con bú",
    "side_effect": "Tác dụng không mong muốn (ADR)",
    "dosage_administration": "Liều lượng và cách dùng",
    "interaction": "Tương tác thuốc",
    "storage": "Bảo quản",
    "overdose": "Quá liều và xử trí",
}

GUIDELINE_COMMON_SECTION_NAMES: dict[str, str] = {
    "overview": "Đại cương",
    "clinical_paraclinical": "Lâm sàng - Cận lâm sàng",
    "diagnosis": "Chẩn đoán",
    "indication_contraindication": "Chỉ định - Chống chỉ định",
}

CLINICAL_GUIDELINE_SECTION_NAMES: dict[str, str] = {
    **GUIDELINE_COMMON_SECTION_NAMES,
    "treatment_procedure": "Điều trị",
    "monitoring_complications": "Theo dõi - Biến chứng",
}

PROCEDURE_SECTION_NAMES: dict[str, str] = {
    **GUIDELINE_COMMON_SECTION_NAMES,
    "treatment_procedure": "Chuẩn bị và các bước tiến hành",
    "monitoring_complications": "Theo dõi và xử trí tai biến",
}

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_LEAKED_DOSAGE_RE = re.compile(
    r"\bDạng\s+(?:thuốc|dùng|bào\s*chế)\s+và\s+hàm\s*l\s*ư\s*ợ\s*n\s*g\b",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", normalize_unicode(str(value))).strip()


def clean_drug_class(raw: Any, max_chars: int = 120) -> str:
    """Clean leaked dosage-form text from drug_class and cap its length."""
    cleaned = _clean_text(raw)
    if not cleaned:
        return ""
    cleaned = _LEAKED_DOSAGE_RE.split(cleaned, maxsplit=1)[0].rstrip(" .;:-")
    if not cleaned:
        return ""
    return truncate_heading(cleaned, max_chars=max_chars)


def truncate_heading(heading: str, max_chars: int = 60) -> str:
    """Strip Markdown heading markers and truncate cleanly to max_chars."""
    cleaned = _HEADING_PREFIX_RE.sub("", normalize_unicode(heading)).strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    limit = max_chars - 3
    cut = cleaned[:limit].rstrip()
    if " " in cleaned[: limit + 1]:
        word_cut = cleaned[:limit].rsplit(" ", 1)[0].rstrip()
        if word_cut:
            cut = word_cut
    return f"{cut}..."


def get_base_section_name(doc_type: str, section_key: str) -> str:
    """Map section_key to its Vietnamese display name for the given doc_type."""
    if doc_type == "drug_monograph":
        return DRUG_SECTION_NAMES.get(section_key, section_key)
    if doc_type == "clinical_guideline":
        return CLINICAL_GUIDELINE_SECTION_NAMES.get(section_key, section_key)
    if doc_type in ("surgical_procedure", "technical_procedure"):
        return PROCEDURE_SECTION_NAMES.get(section_key, section_key)
    return (
        CLINICAL_GUIDELINE_SECTION_NAMES.get(section_key)
        or DRUG_SECTION_NAMES.get(section_key)
        or section_key
    )


def get_section_name(
    doc_type: str,
    section_key: str,
    heading: Optional[str] = None,
    max_heading_chars: int = 60,
) -> str:
    """Return section display name, optionally suffixed with active Markdown heading."""
    base = get_base_section_name(doc_type, section_key)
    if heading:
        raw_heading = _HEADING_PREFIX_RE.sub("", normalize_unicode(heading)).strip()
        raw_heading = _WHITESPACE_RE.sub(" ", raw_heading).strip()
        if raw_heading and raw_heading.lower() != base.lower():
            short_h = truncate_heading(raw_heading, max_chars=max_heading_chars)
            if short_h and short_h.lower() != base.lower():
                return f"{base} > {short_h}"
    return base


def extract_entity_name(doc: Mapping[str, Any]) -> str:
    """Return primary entity name (drug_name_vn or title) normalized to NFC."""
    if doc.get("doc_type") == "drug_monograph":
        return _clean_text(doc.get("drug_name_vn") or doc.get("title") or "")
    return _clean_text(doc.get("title") or doc.get("drug_name_vn") or "")


def extract_source(doc: Mapping[str, Any]) -> str:
    """Return normalized document source string with sensible fallback by doc_type."""
    explicit = _clean_text(doc.get("source"))
    if explicit:
        return explicit
    doc_type = _clean_text(doc.get("doc_type"))
    if doc_type == "drug_monograph":
        return "Dược thư Quốc gia Việt Nam 2018"
    if doc_type == "clinical_guideline":
        return "Hướng dẫn chẩn đoán và điều trị Bộ Y tế"
    if doc_type in ("surgical_procedure", "technical_procedure"):
        return "Quy trình kỹ thuật Bộ Y tế"
    return "Tài liệu y khoa Bộ Y tế"


def extract_specialty_list(doc: Mapping[str, Any]) -> list[str]:
    """Normalize specialty into a list of non-empty strings."""
    if doc.get("doc_type") == "drug_monograph":
        raw = doc.get("specialty_vm14k")
        if raw is None:
            raw = doc.get("specialty")
    else:
        raw = doc.get("specialty")
        if raw is None:
            raw = doc.get("specialty_vm14k")

    if isinstance(raw, list):
        return [_clean_text(x) for x in raw if _clean_text(x)]
    cleaned = _clean_text(raw)
    return [cleaned] if cleaned else []


def build_contextual_prefix(
    doc: Mapping[str, Any],
    section_key: str,
    heading: Optional[str] = None,
    max_heading_chars: int = 60,
) -> str:
    """Build full contextual prefix bracketed header for a parent chunk."""
    doc_type = _clean_text(doc.get("doc_type"))
    section_name = get_section_name(
        doc_type=doc_type,
        section_key=section_key,
        heading=heading,
        max_heading_chars=max_heading_chars,
    )

    if doc_type == "drug_monograph":
        drug_name = _clean_text(doc.get("drug_name_vn"))
        inn_name = _clean_text(doc.get("inn_name"))
        if inn_name and inn_name.lower() != drug_name.lower():
            drug_part = f"{drug_name} ({inn_name})" if drug_name else inn_name
        else:
            drug_part = drug_name
        drug_part = truncate_heading(drug_part, max_chars=160)

        drug_class = clean_drug_class(doc.get("drug_class"), max_chars=120) or clean_drug_class(
            doc.get("group_monograph"), max_chars=120
        )
        parts = ["Nguồn: Dược thư QGVN 2018"]
        if drug_part:
            parts.append(f"Thuốc: {drug_part}")
        if drug_class:
            parts.append(f"Nhóm: {drug_class}")
        parts.append(f"Mục: {section_name}")
        return "[" + " | ".join(parts) + "]"

    specialty = truncate_heading(_clean_text(doc.get("specialty")), max_chars=100)
    title = truncate_heading(extract_entity_name(doc), max_chars=160)

    if doc_type in ("surgical_procedure", "technical_procedure"):
        parts = ["Nguồn: Quy trình kỹ thuật BYT"]
        if specialty:
            parts.append(f"Chuyên khoa: {specialty}")
        if title:
            parts.append(f"Kỹ thuật: {title}")
        parts.append(f"Mục: {section_name}")
        return "[" + " | ".join(parts) + "]"

    parts = ["Nguồn: Hướng dẫn chẩn đoán và điều trị BYT"]
    if specialty:
        parts.append(f"Chuyên khoa: {specialty}")
    if title:
        parts.append(f"Bệnh: {title}")
    parts.append(f"Mục: {section_name}")
    return "[" + " | ".join(parts) + "]"


def build_short_prefix(entity_name: str, section_name: str) -> str:
    """Build compact child prefix for 256-subword ViHealthBERT dense encoding."""
    entity = _clean_text(entity_name)
    section = _clean_text(section_name)
    if entity and section:
        return f"{entity} - {section}: "
    if entity:
        return f"{entity}: "
    if section:
        return f"{section}: "
    return ""

