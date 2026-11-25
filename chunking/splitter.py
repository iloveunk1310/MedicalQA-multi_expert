"""5-level recursive splitter and Parent-Child chunking algorithm."""

import re
from typing import Any, Callable, Optional

from .config import ChunkConfig
from .normalize import is_table_line, is_table_separator
from .prefix import build_contextual_prefix, build_short_prefix, extract_entity_name, get_section_name
from .tokenize_vi import (
    ViHealthBertCounter,
    count_vi_words,
    count_vi_words_from_segmented,
    segment_vi,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _greedy_merge(
    pieces: list[str],
    budget: int,
    length_fn: Callable[[str], int],
    sep: str = "\n\n",
) -> list[str]:
    """Greedily merge adjacent non-empty text pieces while staying within budget."""
    merged: list[str] = []
    cur = ""
    for piece in pieces:
        cleaned = piece.strip()
        if not cleaned:
            continue
        if not cur:
            cur = cleaned
            continue
        candidate = f"{cur}{sep}{cleaned}"
        if length_fn(candidate) <= budget:
            cur = candidate
        else:
            merged.append(cur)
            cur = cleaned
    if cur:
        merged.append(cur)
    return merged


def _split_by_headings(text: str) -> list[str]:
    """Split text at Markdown heading lines, keeping each heading with its following body."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        if HEADING_RE.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(sec).strip() for sec in sections if "\n".join(sec).strip()]


def _parse_blocks(text: str) -> list[tuple[str, Any]]:
    """Parse text into Level-2 blocks: ('table', [lines]) or ('prose', str)."""
    blocks: list[tuple[str, Any]] = []
    prose_buf: list[str] = []
    lines = text.split("\n")
    i = 0

    def flush_prose() -> None:
        if prose_buf:
            joined = "\n".join(prose_buf).strip()
            if joined:
                blocks.append(("prose", joined))
            prose_buf.clear()

    while i < len(lines):
        line = lines[i]
        if is_table_line(line):
            tbl_lines: list[str] = []
            while i < len(lines) and is_table_line(lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            if len(tbl_lines) >= 2 and is_table_separator(tbl_lines[1]):
                flush_prose()
                blocks.append(("table", tbl_lines))
            else:
                # Stray single pipe line or non-table pipe text: treat as prose without pipes
                for stray in tbl_lines:
                    cleaned = re.sub(r"\s*\|\s*", "; ", stray.strip().strip("|").strip())
                    if cleaned:
                        prose_buf.append(cleaned)
            continue

        if not line.strip():
            flush_prose()
        else:
            prose_buf.append(line)
        i += 1

    flush_prose()
    return blocks


def _flatten_table_lines(lines: list[str]) -> str:
    """Flatten a degenerate Markdown table (e.g. oversized header) into plain lines."""
    out_lines: list[str] = []
    for line in lines:
        if is_table_separator(line):
            continue
        inner = line.strip().strip("|").strip()
        cells = [c.strip() for c in inner.split("|") if c.strip()]
        if cells:
            out_lines.append("; ".join(cells))
    return "\n".join(out_lines)


def _split_hard_words_with_fn(
    text: str,
    budget: int,
    measure_fn: Callable[[str], int],
) -> list[str]:
    """Level 5 hard split by whitespace words (and char fallback if a single word > budget)."""
    words = text.split()
    if not words:
        return []

    pieces: list[str] = []
    idx = 0
    n = len(words)
    while idx < n:
        if measure_fn(words[idx]) > budget:
            # Extremely rare single oversized token: binary-split by characters
            tok = words[idx]
            c_idx = 0
            while c_idx < len(tok):
                lo, hi = c_idx + 1, len(tok)
                best = c_idx + 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if measure_fn(tok[c_idx:mid]) <= budget:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                pieces.append(tok[c_idx:best])
                c_idx = best
            idx += 1
            continue

        lo, hi = idx + 1, n
        best = idx + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = " ".join(words[idx:mid])
            if measure_fn(cand) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        pieces.append(" ".join(words[idx:best]))
        idx = best
    return pieces


def _split_sentences_and_words(
    text: str,
    budget: int,
    length_fn: Callable[[str], int],
) -> list[str]:
    """Level 4 (sentences) -> Level 5 (hard words) split with greedy merge."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if length_fn(cleaned) <= budget:
        return [cleaned]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if len(sentences) > 1:
        sub_pieces: list[str] = []
        for sent in sentences:
            if length_fn(sent) <= budget:
                sub_pieces.append(sent)
            else:
                sub_pieces.extend(_split_hard_words_with_fn(sent, budget, length_fn))
        return _greedy_merge(sub_pieces, budget, length_fn, sep=" ")

    return _split_hard_words_with_fn(cleaned, budget, length_fn)


def _split_table_block(
    lines: list[str],
    budget: int,
    length_fn: Callable[[str], int],
) -> list[str]:
    """Split a Markdown table by rows, repeating the 2-line header on every chunk."""
    full_tbl = "\n".join(lines)
    if length_fn(full_tbl) <= budget:
        return [full_tbl]

    header_block = f"{lines[0]}\n{lines[1]}"
    header_len = length_fn(header_block)
    if header_len > int(budget * 0.6) or budget - header_len < 12:
        flattened = _flatten_table_lines(lines)
        return split_text_recursive(flattened, budget, length_fn, _level=3)

    data_rows = lines[2:]
    if not data_rows:
        return [header_block]

    chunks: list[str] = []
    cur_rows: list[str] = []
    for row in data_rows:
        candidate = header_block + "\n" + "\n".join(cur_rows + [row])
        if length_fn(candidate) <= budget:
            cur_rows.append(row)
            continue

        if cur_rows:
            chunks.append(header_block + "\n" + "\n".join(cur_rows))
            cur_rows = []

        single = f"{header_block}\n{row}"
        if length_fn(single) <= budget:
            cur_rows = [row]
            continue

        # Single table row exceeds budget even with header: split row content safely
        inner = row.strip().strip("|").strip()
        inner_text = re.sub(r"\s*\|\s*", "; ", inner)
        wrapper_cost = length_fn(f"{header_block}\n|  |")
        frag_budget = max(1, budget - wrapper_cost - 2)
        frags = _split_sentences_and_words(inner_text, frag_budget, length_fn)
        for frag in frags:
            wrapped = f"{header_block}\n| {frag} |"
            if length_fn(wrapped) <= budget:
                chunks.append(wrapped)
            else:
                sub_frags = _split_hard_words_with_fn(
                    frag,
                    budget,
                    lambda f: length_fn(f"{header_block}\n| {f} |"),
                )
                for sf in sub_frags:
                    chunks.append(f"{header_block}\n| {sf} |")

    if cur_rows:
        chunks.append(header_block + "\n" + "\n".join(cur_rows))
    return chunks


def split_text_recursive(
    text: str,
    budget: int,
    length_fn: Callable[[str], int],
    _level: int = 1,
) -> list[str]:
    """Recursively split text across 5 structural levels while respecting budget."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if length_fn(cleaned) <= budget:
        return [cleaned]

    if _level <= 1:
        heading_parts = _split_by_headings(cleaned)
        if len(heading_parts) > 1:
            sub_pieces: list[str] = []
            for part in heading_parts:
                sub_pieces.extend(split_text_recursive(part, budget, length_fn, _level=2))
            return _greedy_merge(sub_pieces, budget, length_fn, sep="\n\n")

    if _level <= 2:
        blocks = _parse_blocks(cleaned)
        if len(blocks) > 1 or (len(blocks) == 1 and blocks[0][0] == "table"):
            sub_pieces = []
            for btype, bval in blocks:
                if btype == "table":
                    sub_pieces.extend(_split_table_block(bval, budget, length_fn))
                else:
                    sub_pieces.extend(split_text_recursive(bval, budget, length_fn, _level=3))
            return _greedy_merge(sub_pieces, budget, length_fn, sep="\n\n")
        if len(blocks) == 1 and blocks[0][0] == "prose":
            cleaned = blocks[0][1]

    if _level <= 3:
        lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
        if len(lines) > 1:
            sub_pieces = []
            for ln in lines:
                if length_fn(ln) <= budget:
                    sub_pieces.append(ln)
                else:
                    sub_pieces.extend(split_text_recursive(ln, budget, length_fn, _level=4))
            return _greedy_merge(sub_pieces, budget, length_fn, sep="\n")

    return _split_sentences_and_words(cleaned, budget, length_fn)


def _extract_leading_heading(text: str) -> Optional[str]:
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            if HEADING_RE.match(stripped):
                return stripped
            return None
    return None


def split_section_to_parents(
    doc: dict[str, Any],
    section_key: str,
    normalized_text: str,
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    """Split a normalized section into non-overlapping parent chunks <= parent_max_words."""
    cleaned = normalized_text.strip()
    if not cleaned:
        return []

    doc_type = str(doc.get("doc_type") or "")
    first_heading = _extract_leading_heading(cleaned)
    first_prefix = build_contextual_prefix(
        doc,
        section_key,
        heading=first_heading,
        max_heading_chars=config.max_heading_chars,
    )
    first_full = f"{first_prefix}\n{cleaned}"
    first_wc = count_vi_words(first_full)
    if first_wc <= config.parent_max_words:
        return [
            {
                "section_name": get_section_name(
                    doc_type, section_key, first_heading, config.max_heading_chars
                ),
                "contextual_prefix": first_prefix,
                "text": cleaned,
                "word_count": first_wc,
            }
        ]

    heading_blocks = _split_by_headings(cleaned)
    raw_items: list[tuple[Optional[str], str]] = []
    for h_block in heading_blocks:
        heading = _extract_leading_heading(h_block)
        prefix = build_contextual_prefix(
            doc,
            section_key,
            heading=heading,
            max_heading_chars=config.max_heading_chars,
        )
        prefix_words = count_vi_words(prefix)
        budget = max(16, config.parent_max_words - prefix_words)
        pieces = split_text_recursive(h_block, budget, count_vi_words)
        for piece in pieces:
            raw_items.append((heading, piece))

    # Merge tiny consecutive heading blocks if the combined text fits within the opening parent budget
    merged_items: list[tuple[Optional[str], str, str]] = []
    cur_heading: Optional[str] = None
    cur_prefix = ""
    cur_text = ""
    for heading, piece in raw_items:
        if not cur_text:
            cur_heading = heading
            cur_prefix = build_contextual_prefix(
                doc,
                section_key,
                heading=cur_heading,
                max_heading_chars=config.max_heading_chars,
            )
            cur_text = piece
            continue

        cand_text = f"{cur_text}\n\n{piece}"
        if count_vi_words(f"{cur_prefix}\n{cand_text}") <= config.parent_max_words:
            cur_text = cand_text
        else:
            merged_items.append((cur_heading, cur_prefix, cur_text))
            cur_heading = heading
            cur_prefix = build_contextual_prefix(
                doc,
                section_key,
                heading=cur_heading,
                max_heading_chars=config.max_heading_chars,
            )
            cur_text = piece

    if cur_text:
        merged_items.append((cur_heading, cur_prefix, cur_text))

    # Post-check & repair so no parent ever exceeds parent_max_words
    parents: list[dict[str, Any]] = []
    for heading, prefix, ptext in merged_items:
        full_wc = count_vi_words(f"{prefix}\n{ptext}")
        if full_wc <= config.parent_max_words:
            parents.append(
                {
                    "section_name": get_section_name(
                        doc_type, section_key, heading, config.max_heading_chars
                    ),
                    "contextual_prefix": prefix,
                    "text": ptext,
                    "word_count": full_wc,
                }
            )
            continue

        def _measure_with_prefix(s: str, _pfx: str = prefix) -> int:
            return count_vi_words(f"{_pfx}\n{s}")

        repaired = split_text_recursive(ptext, config.parent_max_words, _measure_with_prefix, _level=3)
        for rpiece in repaired:
            if _measure_with_prefix(rpiece) > config.parent_max_words:
                sub_repaired = _split_hard_words_with_fn(
                    rpiece, config.parent_max_words, _measure_with_prefix
                )
            else:
                sub_repaired = [rpiece]
            for sp in sub_repaired:
                parents.append(
                    {
                        "section_name": get_section_name(
                            doc_type, section_key, heading, config.max_heading_chars
                        ),
                        "contextual_prefix": prefix,
                        "text": sp,
                        "word_count": _measure_with_prefix(sp),
                    }
                )
    return parents


def _extract_trailing_unit_overlap(
    prev_text: str,
    curr_text: str,
    overlap_max: int,
    max_len: int,
    measure_content: Callable[[str], int],
    measure_full: Callable[[str], int],
) -> str:
    """Collect intact trailing sentences/lines from prev_text up to overlap_max, never breaking tables."""
    if overlap_max <= 0:
        return ""
    prev_lines = [ln for ln in prev_text.splitlines() if ln.strip()]
    curr_lines = [ln for ln in curr_text.splitlines() if ln.strip()]
    if not prev_lines or not curr_lines:
        return ""
    if is_table_line(prev_lines[-1]) or is_table_line(curr_lines[0]):
        return ""

    # Build atomic units from trailing non-table lines (sentences within each line)
    unit_groups: list[tuple[str, bool]] = []  # (unit_text, starts_new_line)
    for ln in prev_lines:
        if is_table_line(ln):
            unit_groups.clear()
            continue
        sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(ln.strip()) if s.strip()]
        for idx, s in enumerate(sents):
            unit_groups.append((s, idx == 0))

    if not unit_groups:
        return ""

    chosen: list[tuple[str, bool]] = []
    best_overlap = ""
    for unit_text, starts_line in reversed(unit_groups):
        trial_list = [(unit_text, starts_line)] + chosen
        parts: list[str] = []
        for idx, (ut, sl) in enumerate(trial_list):
            if idx == 0:
                parts.append(ut)
            elif sl:
                parts.append(f"\n{ut}")
            else:
                parts.append(f" {ut}")
        cand_ov = "".join(parts)
        if measure_content(cand_ov) > overlap_max:
            break
        if measure_full(f"{cand_ov}\n{curr_text}") > max_len:
            break
        chosen = trial_list
        best_overlap = cand_ov

    return best_overlap


def _repair_child_piece(
    text: str,
    max_len: int,
    measure_full: Callable[[str], int],
) -> list[str]:
    """Ensure every child piece strictly satisfies measure_full(piece) <= max_len."""
    if measure_full(text) <= max_len:
        return [text]

    pieces = split_text_recursive(text, max_len, measure_full, _level=2)
    final_pieces: list[str] = []
    for p in pieces:
        if measure_full(p) <= max_len:
            final_pieces.append(p)
        else:
            final_pieces.extend(_split_hard_words_with_fn(p, max_len, measure_full))
    return final_pieces


def split_parent_to_children(
    parent_text: str,
    short_prefix: str,
    config: ChunkConfig,
    counter: Optional[ViHealthBertCounter] = None,
) -> list[dict[str, Any]]:
    """Split a single parent's text into child chunks strictly within the child budget."""
    cleaned = parent_text.strip()
    if not cleaned:
        return []

    is_subword = config.child_length_mode == "subword"
    if is_subword:
        if counter is None:
            raise ValueError("ViHealthBertCounter is required when child_length_mode='subword'.")
        max_len = config.child_max_subwords
        overlap_max = config.child_overlap_subwords

        def measure_full(txt: str) -> int:
            return counter.count_subwords(f"{short_prefix}{txt}", add_special_tokens=True)

        def measure_content(txt: str) -> int:
            return counter.count_subwords(txt, add_special_tokens=False)

        prefix_cost = counter.count_subwords(short_prefix, add_special_tokens=True)
        content_budget = max(8, max_len - prefix_cost - 2)
    else:
        max_len = config.child_max_words
        overlap_max = config.child_overlap_words

        def measure_full(txt: str) -> int:
            return count_vi_words(f"{short_prefix}{txt}")

        def measure_content(txt: str) -> int:
            return count_vi_words(txt)

        prefix_cost = count_vi_words(short_prefix)
        content_budget = max(8, max_len - prefix_cost)

    def _build_child_record(ch_text: str) -> dict[str, Any]:
        dense = f"{short_prefix}{ch_text}"
        dense_seg = segment_vi(dense)
        sw_count = (
            counter.count_subwords_from_seg(dense_seg, add_special_tokens=True)
            if is_subword and counter is not None
            else None
        )
        return {
            "text": ch_text,
            "text_for_dense": dense,
            "text_for_dense_seg": dense_seg,
            "subword_count": sw_count,
            "word_count": count_vi_words_from_segmented(dense_seg),
        }

    # 1 Parent = 1 Child fast path
    if measure_full(cleaned) <= max_len:
        return [_build_child_record(cleaned)]

    base_chunks = split_text_recursive(cleaned, content_budget, measure_content)
    final_texts: list[str] = []
    for idx, base_ch in enumerate(base_chunks):
        cand_ch = base_ch
        if idx > 0 and overlap_max > 0:
            ov = _extract_trailing_unit_overlap(
                prev_text=base_chunks[idx - 1],
                curr_text=base_ch,
                overlap_max=overlap_max,
                max_len=max_len,
                measure_content=measure_content,
                measure_full=measure_full,
            )
            if ov:
                cand_ch = f"{ov}\n{base_ch}"

        if measure_full(cand_ch) <= max_len:
            final_texts.append(cand_ch)
        elif cand_ch != base_ch and measure_full(base_ch) <= max_len:
            final_texts.append(base_ch)
        else:
            final_texts.extend(_repair_child_piece(base_ch, max_len, measure_full))

    return [_build_child_record(t) for t in final_texts if t.strip()]

