"""Locate the model's excerpts in the checked text.

Every issue must point at text the UI can highlight. The model's excerpt is
located in the checked text and replaced by the exact source passage. Excerpts
that cannot be located are re-quoted once by the model and accepted only if
they match.
"""

import json
import re
import unicodedata
from typing import NamedTuple

import httpx

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
SHINGLE = 3
MIN_FUZZY_WORDS = 4
# A fuzzy match is accepted when at least half of the excerpt's words, and at
# least six of them, line up in order with one passage. Shorter overlaps are
# generic phrases that also occur in unrelated texts, so those excerpts go to
# the repair call instead.
FUZZY_ACCEPT_COVERAGE = 0.5
FUZZY_MIN_WORDS = 6

OMISSION_PATTERN = re.compile(r"(?:…|\.{3,})")
CHUNK_SEPARATOR_PATTERN = re.compile(r"(?:…|\.{3,}|\r?\n)+")
# Pasted links are kept as "anchor text (https://...)". Models quote the anchor
# with or without the URL (sometimes shortened with an ellipsis), so URLs never
# take part in matching.
URL_PATTERN = re.compile(r'\(?https?://[^\s<>"]+')
# Square-bracket insertions are the model's own words; image markers are real
# source text.
EDITORIAL_BRACKETS = re.compile(r"\[(?!image\b)[^\]]*\]", re.IGNORECASE)
# NFKC keeps the Romanian cedilla letters apart from the comma-below ones that
# the same word is often typed with.
LETTER_FOLDS = {"ş": "ș", "ţ": "ț"}


class FuzzyMatch(NamedTuple):
    start: int
    end: int
    coverage: float
    words: int


def canonicalize(value):
    """Punctuation- and whitespace-insensitive text with offsets back into `value`.

    Returns (text, starts, ends): for every character of `text`, the span of the
    source character it came from.
    """
    inside_url = [False] * len(value)
    for match in URL_PATTERN.finditer(value):
        for i in range(match.start(), match.end()):
            inside_url[i] = True

    text = []
    starts = []
    ends = []
    pending_separator = False
    pending_start = 0
    pending_end = 0

    for source_start, source_character in enumerate(value):
        source_end = source_start + 1
        lowered = " " if inside_url[source_start] else unicodedata.normalize("NFKC", source_character).lower()
        normalized = LETTER_FOLDS.get(lowered, lowered)

        for character in normalized:
            if unicodedata.category(character)[0] in "LN":
                if pending_separator and text and text[-1] != " ":
                    text.append(" ")
                    starts.append(pending_start)
                    ends.append(pending_end)
                text.append(character)
                starts.append(source_start)
                ends.append(source_end)
                pending_separator = False
            else:
                if not pending_separator:
                    pending_start = source_start
                pending_end = source_end
                pending_separator = True

    return "".join(text), starts, ends


def canonical_words(value):
    """Words of a text in canonical form as (word, start, end) in the source."""
    text, starts, ends = canonicalize(value)
    words = []
    word_start = -1
    for i in range(len(text) + 1):
        if i == len(text) or text[i] == " ":
            if word_start != -1:
                words.append((text[word_start:i], starts[word_start], ends[i - 1]))
                word_start = -1
        elif word_start == -1:
            word_start = i
    return words


def find_canonical_range(source, excerpt, from_source_offset=0):
    source_text, source_starts, source_ends = source
    needle = canonicalize(excerpt)[0].strip()
    if not needle:
        return None

    canonical_start = 0
    while canonical_start < len(source_starts) and source_starts[canonical_start] < from_source_offset:
        canonical_start += 1

    match = source_text.find(needle, canonical_start)
    while match != -1:
        before_is_boundary = match == 0 or source_text[match - 1] == " "
        after = match + len(needle)
        after_is_boundary = after == len(source_text) or source_text[after] == " "
        if before_is_boundary and after_is_boundary:
            return source_starts[match], source_ends[after - 1]
        match = source_text.find(needle, match + 1)

    return None


def find_excerpt_ranges(original_text, excerpt):
    """Locate an excerpt in the original text as a list of (start, end) ranges.

    Exact excerpts remain a single range. If the model joins non-contiguous
    quotations with an ellipsis or line break, each quoted part is returned as
    its own range.
    """
    if not original_text or not excerpt.strip():
        return []

    exact_start = original_text.find(excerpt)
    if exact_start != -1:
        return [(exact_start, exact_start + len(excerpt))]

    canonical_source = canonicalize(original_text)
    whole_range = find_canonical_range(canonical_source, excerpt)
    if whole_range:
        return [whole_range]

    without_urls = URL_PATTERN.sub(" ", excerpt)
    if not (OMISSION_PATTERN.search(without_urls) or re.search(r"\r?\n", without_urls)):
        return []

    chunks = [chunk.strip() for chunk in CHUNK_SEPARATOR_PATTERN.split(without_urls)]
    chunks = [chunk for chunk in chunks if len(canonicalize(chunk)[0].strip()) >= 2]
    if len(chunks) < 2:
        return []

    ranges = []
    next_source_offset = 0
    for chunk in chunks:
        exact_chunk_start = original_text.find(chunk, next_source_offset)
        if exact_chunk_start != -1:
            found = (exact_chunk_start, exact_chunk_start + len(chunk))
        else:
            found = find_canonical_range(canonical_source, chunk, next_source_offset)
        if not found:
            continue
        ranges.append(found)
        next_source_offset = found[1]

    # A single matched fragment is too weak to prove that a composite excerpt
    # refers to this location, so no highlight is invented from it.
    return ranges if len(ranges) >= 2 else []


def slice_ranges(text, ranges):
    return " … ".join(text[start:end] for start, end in ranges)


def locate_excerpt(text, excerpt):
    """Locate an excerpt without guessing.

    Verbatim, normalised (case, punctuation, whitespace, links), split at the
    model's ellipses, or without bracketed insertions. Returns (path, ranges)
    or None.
    """
    if not excerpt or not excerpt.strip():
        return None
    exact = text.find(excerpt)
    if exact != -1:
        return "exact", [(exact, exact + len(excerpt))]

    ranges = find_excerpt_ranges(text, excerpt)
    if ranges:
        return ("multi" if len(ranges) > 1 else "normalized"), ranges

    stripped = EDITORIAL_BRACKETS.sub(" ", excerpt)
    if stripped != excerpt:
        stripped_ranges = find_excerpt_ranges(text, stripped)
        if stripped_ranges:
            return "brackets", stripped_ranges
    return None


def build_fuzzy_index(text):
    words = canonical_words(text)
    shingles = {}
    for i in range(len(words) - SHINGLE + 1):
        key = " ".join(word for word, _, _ in words[i:i + SHINGLE])
        shingles.setdefault(key, []).append(i)
    return words, shingles


def fuzzy_locate(index, excerpt):
    """Find the passage that shares the most word triples with the excerpt, in order.

    Handles quotes whose tail drifts from the source, single changed words and
    inserted commentary. Returns a FuzzyMatch or None.
    """
    words, shingles = index
    needle = [word for word, _, _ in canonical_words(EDITORIAL_BRACKETS.sub(" ", excerpt))]
    if len(needle) < MIN_FUZZY_WORDS:
        return None

    votes = []
    for j in range(len(needle) - SHINGLE + 1):
        for i in shingles.get(" ".join(needle[j:j + SHINGLE]), []):
            votes.append((i - j, i, j))
    if len(votes) < 2:
        return None
    votes.sort(key=lambda vote: vote[0])

    # Votes whose offsets agree within the band describe one passage that has a
    # few words inserted or dropped relative to the excerpt.
    band = max(3, round(len(needle) * 0.3))
    best = None
    right = 0
    for left in range(len(votes)):
        while right < len(votes) and votes[right][0] - votes[left][0] <= band:
            right += 1
        group = votes[left:right]
        covered = {j + k for _, _, j in group for k in range(SHINGLE)}
        min_i = min(i for _, i, _ in group)
        max_i = max(i for _, i, _ in group) + SHINGLE - 1
        candidate = (len(covered), min_i, max_i, len(group))
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and max_i - min_i < best[2] - best[1]):
            best = candidate
    covered_count, min_i, max_i, count = best
    if count < 2 or max_i - min_i + 1 > len(needle) * 2:
        return None

    return FuzzyMatch(words[min_i][1], words[max_i][2], covered_count / len(needle), covered_count)


def request_excerpts(text, items, model, api_key, post):
    response = post(
        OPENROUTER_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You copy quotations out of a document. Reply with JSON only."},
                {
                    "role": "user",
                    "content": (
                        f"TEXT:\n<<<\n{text}\n>>>\n\n"
                        "Each issue below was found in TEXT. For each one, copy from TEXT the passage the issue is about: "
                        "one complete sentence or heading, character for character exactly as it appears in TEXT. "
                        "Do not shorten it with an ellipsis, do not join separate passages, do not add brackets, links or words of your own. "
                        "If the issue is about missing information, copy the sentence or heading where that information belongs.\n\n"
                        f"ISSUES:\n{json.dumps(items)}\n\n"
                        'Return {"fixes":[{"index":<issue index>,"excerpt":"<copied passage>"}]} with one entry per issue.'
                    ),
                },
            ],
            "reasoning": {"effort": "low"},
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "excerpt_fixes",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["fixes"],
                        "properties": {
                            "fixes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["index", "excerpt"],
                                    "properties": {"index": {"type": "integer"}, "excerpt": {"type": "string"}},
                                },
                            }
                        },
                    },
                },
            },
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Excerpt repair: OpenRouter HTTP {response.status_code}")
    data = response.json()
    fixes = json.loads(data["choices"][0]["message"]["content"])["fixes"]
    usage = data.get("usage", {})
    return fixes, {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")},
        "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
        "cost": usage.get("cost"),
    }


def align_issue_excerpts(text, issues, model, api_key, post=httpx.post):
    """Replace every issue's excerpt with the exact source passage it refers to.

    Returns {"issues", "paths", "repair_usage", "repair_error"}. A failing repair
    call never fails the check: the issues come back with the paths found so far.
    """
    index = build_fuzzy_index(text)
    aligned = [dict(issue) for issue in issues]
    paths = ["none"] * len(aligned)
    unresolved = []

    for i, issue in enumerate(aligned):
        excerpt = issue.get("excerpt", "")
        located = locate_excerpt(text, excerpt)
        if located:
            path, ranges = located
            issue["excerpt"] = slice_ranges(text, ranges)
            paths[i] = path
            continue
        fuzzy = fuzzy_locate(index, excerpt) if excerpt else None
        if fuzzy and fuzzy.coverage >= FUZZY_ACCEPT_COVERAGE and fuzzy.words >= FUZZY_MIN_WORDS:
            issue["excerpt"] = text[fuzzy.start:fuzzy.end]
            paths[i] = "fuzzy"
            continue
        unresolved.append(i)

    result = {"issues": aligned, "paths": paths, "repair_usage": None, "repair_error": None}
    if not unresolved or not api_key:
        return result

    items = [
        {"index": i, "type": aligned[i].get("type"), "issue": aligned[i].get("issue"), "excerpt": aligned[i].get("excerpt", "")}
        for i in unresolved
    ]
    try:
        fixes, usage = request_excerpts(text, items, model, api_key, post)
        for fix in fixes:
            if fix["index"] not in unresolved:
                continue
            located = locate_excerpt(text, fix["excerpt"])
            if not located:
                continue
            aligned[fix["index"]]["excerpt"] = slice_ranges(text, located[1])
            paths[fix["index"]] = "repaired"
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError) as error:
        result["repair_error"] = str(error)
        return result
    result["repair_usage"] = usage
    return result
