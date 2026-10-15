"""Split a trial's free-text eligibility block into individual criteria.

Rule-based on purpose: free, instant, deterministic and easy to test. Steps:
  1. find Inclusion / Exclusion headers and give each section its kind;
  2. split a section into items at bullets (-, *, •) and numbering (1. / 1) / a.);
  3. glue wrapped lines back onto the item they belong to;
  4. keep nested sub-bullets inside their parent item ("- child" lines).
Anything we cannot structure falls back safely and is reported in `notes`.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# Fallback / diagnostic flags reported in SplitResult.notes.
EMPTY_TEXT = "empty_text"  # nothing to split
NO_HEADERS = "no_headers"  # no Inclusion/Exclusion header; everything is kind=unknown
TEXT_BEFORE_HEADER = "text_before_first_header"  # kept as kind=unknown
NO_BULLETS = "no_bullets"  # section had several lines but no markers; split on blank lines
INLINE_NUMBERED = "inline_numbered"  # "1) ... 2) ... 3) ..." run on one line, split at the numbers
FLATTENED_WRAPPER = "flattened_wrapper"  # one intro bullet whose sub-items are the real criteria
UNSPLIT_LONG = "unsplit_long_section"  # one very long item; probably unstructured run-on text

_LONG_ITEM_CHARS = 400

_QUALIFIERS = (
    r"key|main|major|general|additional|other|basic|specific|core|study|protocol|the|"
    r"patient|patients|subject|subjects|participant|participants|eligibility|criteria|"
    r"for|of|are|is"
)
_HEADER = re.compile(
    rf"""^(?:(?:{_QUALIFIERS})\s+)*
        (?P<kind>inclusion|exclusion)s?
        (?:\s+(?:criteria|criterion))?
        (?:\s+(?:are|is))?
        (?:\s+for\s+[a-z0-9 ,/-]{{0,40}}|\s*\([^)]*\)|\s+-\s+[a-z0-9 ]{{1,30}}|\s*,\s*[a-z0-9 /-]{{1,40}})*
        \s*(?P<colon>[:：])?\s*(?P<rest>.*)$""",
    re.IGNORECASE | re.VERBOSE,
)
# "Cohort A (Breast Cohort) Inclusion Criteria" -- group label first, no colon.
_GROUP_HEADER = re.compile(
    r"^(?:cohort|arm|part|phase|group|stage|module|step)\b[^:]{0,50}?\b(?P<kind>inclusion|exclusion)\s+criteria\s*$",
    re.IGNORECASE,
)
_HEADER_LEAD = re.compile(r"^[\s*#_>•●▪✔✓☑\-–—:]+")
_HEADER_ENUM = re.compile(r"^(?:\d{1,2}|[ivx]+|[a-z])[.)]\s+", re.IGNORECASE)
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()<>#+\-.!~|])")  # the API returns markdown: "1\\)" -> "1)"
_INLINE_NUM = re.compile(r"(?:^|(?<=\s))(\d{1,2})[.)]\s+")
_HAS_WORD = re.compile(r"[^\W_]")
_DECORATION = re.compile(r"^[\s*_#]*$")

_BULLET = re.compile(
    r"""^(?P<indent>\ *)
        (?:(?P<sym>[-–—*•●▪◦·✔✓☐□]+)\s+
          |(?P<sym2>[•●▪◦·]+)
          |(?P<num>\(?\d{1,3}[.)])\s+
          |(?P<alpha>\(?[A-Za-z][.)])\s+)
        (?P<text>\S.*)$""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Criterion:
    kind: str  # 'inclusion' | 'exclusion' | 'unknown'
    position: int  # 1-based, across the whole trial
    text: str  # nested sub-bullets are appended as "\n- child" lines


@dataclass(frozen=True)
class SplitResult:
    criteria: list[Criterion]
    notes: frozenset[str]


def split_criteria(text: str | None) -> SplitResult:
    if not text or not text.strip():
        return SplitResult([], frozenset({EMPTY_TEXT}))

    notes: set[str] = set()
    text = _MD_ESCAPE.sub(r"\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    sections = _split_sections(text)

    if all(kind == "unknown" for kind, _ in sections):
        notes.add(NO_HEADERS)
    elif sections[0][0] == "unknown" and any(line.strip() for line in sections[0][1]):
        notes.add(TEXT_BEFORE_HEADER)

    criteria: list[Criterion] = []
    for kind, lines in sections:
        items, section_notes = _split_items(lines)
        notes |= section_notes
        for item in items:
            criteria.append(Criterion(kind, len(criteria) + 1, item))
    return SplitResult(criteria, frozenset(notes))


def summarize_notes(results: Iterable[SplitResult]) -> Counter:
    """How often each fallback fired, plus the number of trials seen."""
    counts: Counter = Counter()
    for result in results:
        counts["trials"] += 1
        counts.update(result.notes)
    return counts


# --- step 1: sections ----------------------------------------------------


def _match_header(line: str) -> tuple[str, str] | None:
    """Return (kind, text on the same line after the colon) if `line` is a header."""
    stripped = _HEADER_LEAD.sub("", line)
    stripped = _HEADER_ENUM.sub("", stripped)
    group = _GROUP_HEADER.match(stripped)
    if group:
        return group.group("kind").lower(), ""
    match = _HEADER.match(stripped)
    if not match:
        return None
    rest = match.group("rest").strip()
    if _DECORATION.match(rest):
        rest = ""
    if rest and not match.group("colon"):
        return None  # "Exclusion of patients who ..." is a criterion, not a header
    return match.group("kind").lower(), rest.strip("*_ ")


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    kind, current = "unknown", []
    for line in text.expandtabs(4).split("\n"):
        header = _match_header(line)
        if header is None:
            current.append(line)
            continue
        sections.append((kind, current))
        kind, current = header[0], [header[1]] if header[1] else []
    sections.append((kind, current))
    return sections


# --- steps 2-4: items ----------------------------------------------------


@dataclass
class _Item:
    text: str
    children: list[str]

    def render(self) -> str:
        return "\n".join([self.text, *(f"- {c}" for c in self.children)]).strip()


def _clean(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _marker_class(match: re.Match) -> str:
    return "num" if match.group("num") else "alpha" if match.group("alpha") else "sym"


def _split_items(lines: list[str]) -> tuple[list[str], set[str]]:
    notes: set[str] = set()
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return [], notes

    if not any(_BULLET.match(line) for line in nonblank):
        # No markers: items are blank-line-separated paragraphs; wrapped lines are glued.
        if len(nonblank) > 1:
            notes.add(NO_BULLETS)
        paragraphs, current = [], []
        for line in lines:
            if line.strip():
                current.append(_clean(line))
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return _finish(paragraphs, notes)

    items: list[_Item] = []
    base: tuple[int, str] | None = None  # (indent, marker class) of the first bullet
    after_blank = False
    last_was_child = False
    for line in lines:
        if not line.strip():
            after_blank = True
            continue
        match = _BULLET.match(line)
        if match:
            indent, cls, text = len(match.group("indent")), _marker_class(match), _clean(match.group("text"))
            if base is None:
                base = (indent, cls)
            nested = bool(items) and (
                indent >= base[0] + 2
                or (base[1] == "num" and cls in ("sym", "alpha"))
                or (base[1] == "sym" and cls == "alpha")
            )
            if nested:
                items[-1].children.append(text)
            else:
                items.append(_Item(text, []))
            last_was_child = nested
        elif not items or after_blank:
            items.append(_Item(_clean(line), []))  # unbulleted paragraph / intro line
            last_was_child = False
        elif last_was_child:
            items[-1].children[-1] += " " + _clean(line)  # wrapped line
        else:
            items[-1].text += " " + _clean(line)  # wrapped line
        after_blank = False
    if len(items) == 1 and len(items[0].children) >= 2:
        # "* Subjects who meet all of the following:" + indented list: the list is the content.
        notes.add(FLATTENED_WRAPPER)
        only = items[0]
        return _finish([only.text, *only.children], notes)
    return _finish([item.render() for item in items], notes)


def _split_inline_numbers(text: str) -> list[str] | None:
    """'1) a; 2) b; 3) c' -> ['a;', 'b;', 'c']. Needs numbers in order, at least 3 items in all.

    The first marker may already have been eaten as a bullet, so a run may start at 2.
    """
    chain: list[re.Match] = []
    for m in _INLINE_NUM.finditer(text):
        n = int(m.group(1))
        if n == (int(chain[-1].group(1)) + 1 if chain else (1 if m.start() == 0 else 2)):
            chain.append(m)
    if not chain or int(chain[-1].group(1)) < 3:
        return None
    parts = [text[: chain[0].start()]]
    for m, nxt in zip(chain, chain[1:] + [None]):
        parts.append(text[m.end() : nxt.start() if nxt else len(text)])
    return [_clean(p) for p in parts if _clean(p)]


def _finish(items: list[str], notes: set[str]) -> tuple[list[str], set[str]]:
    items = [i for i in items if _HAS_WORD.search(i)]  # drops "-", "<!-- -->", etc.
    if len(items) == 1:
        parts = _split_inline_numbers(items[0])
        if parts:
            notes.add(INLINE_NUMBERED)
            return [p for p in parts if p], notes
    if len(items) == 1 and len(items[0]) > _LONG_ITEM_CHARS:
        notes.add(UNSPLIT_LONG)
    return items, notes
