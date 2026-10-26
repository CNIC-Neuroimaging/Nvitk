"""Rewrite comments and documentation strings of Python files without touching code.

What it changes, and only inside comments and bare string statements (docstrings):

* decorative rule lines (``# ─────``, ``# ------``) are dropped, keeping any title;
* em dashes become a colon, comma, semicolon or parentheses chosen from the context;
* typographic arrows and symbols become ASCII (``->``, ``>=``, ``...``);
* filler adverbs (actually, genuinely, simply, essentially) are dropped and a few
  rhetorical phrases ("the honest answer", "quietly") get plain wording.

Every rewritten file is checked before it is written: its AST with the documentation strings
blanked must equal the original's, so no statement, literal or name can change. Files that fail
the check are reported and left alone.

Usage::

    python audit/scripts/deslop.py [--write] [--report edits.tsv] FILE...
    python audit/scripts/deslop.py --text [--write] pyproject.toml install.sh   # '#' comment lines only
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize

CHAR_MAP = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "…": "...",
    "–": "-",    # en dash (ranges)
    "−": "-",    # minus sign
    "≥": ">=",
    "≤": "<=",
}

RULE_RE = re.compile(r"^(?P<lead>[─━═\-=*~#_]{3,})?\s*(?P<title>.*?)\s*(?P<trail>[─━═\-=*~#_]{3,})?\s*$")
DIAGRAM_CHARS = set("│├└┌┐┘┬┴┼►▼▲◄╭╮╯╰")

COMMA_WORDS = {
    "which", "where", "when", "whenever", "while", "so", "but", "and", "or", "nor", "not", "unless",
    "since", "because", "as", "though", "although", "even", "except", "including", "plus",
    "rather", "then", "yet", "e.g.", "i.e.", "whichever", "whatever", "if", "otherwise",
    "especially", "typically", "usually", "never", "only", "just", "hence", "thus", "instead",
    "with", "without", "before", "after", "until", "once", "whether", "who", "whose", "now",
    "for", "in", "on", "at", "by", "to", "from", "via", "per", "into", "over", "under",
}
CLOSE_COMMA_WORDS = {
    "which", "where", "when", "whenever", "while", "so", "but", "and", "or", "nor", "unless",
    "since", "because", "though", "although", "even", "except", "including", "plus", "rather",
    "then", "yet", "e.g.", "i.e.", "otherwise", "hence", "thus", "whereas", "whether", "not",
}
SUBORDINATE_START = {"when", "if", "although", "though", "because", "since", "while", "unless",
                     "as", "once", "after", "before", "until", "whenever", "where", "even"}
LIST_MARK_RE = re.compile(r"\n[ \t]*(?:[*+-]|\d+[.)])[ \t]+")
ROLE_RE = re.compile(r":[a-zA-Z_]+:`[^`]*`|``[^`]*``|`[^`]*`|::|noqa:|https?:")
DASH_RE = re.compile(r"[ \t]*—[ \t]*")


def _first_word(text: str) -> str:
    m = re.match(r"\s*([A-Za-z][A-Za-z.]*)", text)
    return m.group(1).lower() if m else ""


def _colon_already(sentence_left: str) -> bool:
    cleaned = ROLE_RE.sub("", sentence_left)
    return ":" in cleaned


def _sentence_start(text: str, pos: int) -> int:
    """Index where the sentence (or list item) containing *pos* starts."""
    best = 0
    for m in re.finditer(r"[.!?](?:\s+|$)|\n\s*\n|\n[ \t]*(?:[*+-]|\d+[.)])[ \t]+", text[:pos]):
        best = m.end()
    return best


def _line_left(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    left = text[start:pos].strip()
    return re.sub(r"^(?:[*+-]|\d+[.)])\s+", "", left)


def _is_label(left_line: str) -> bool:
    """True when the text before the dash on its line reads as a list label / term."""
    if not left_line:
        return False
    words = left_line.split()
    if left_line.endswith(("``", "`", "**", "*", ")", "'", '"')) and len(words) <= 6:
        return True
    return len(words) <= 3 and not left_line.endswith((",", ";", ":"))


def _sentence_end(text: str, pos: int) -> int:
    m = re.compile(r"[.!?](?=\s|$)|\n\s*\n").search(text, pos)
    return m.start() if m else len(text)


def replace_dashes(text: str, *, noqa: bool = False) -> str:
    """Replace em dashes with plain punctuation chosen from the surrounding words."""
    if "—" not in text:
        return text
    if noqa:
        # A noqa reason after a dash goes in parentheses: CODE (reason).
        m = re.match(r"(.*?noqa(?::\s*[A-Z0-9, ]+?)?)[ \t]*—[ \t]*(.*)$", text, re.S)
        if m and "—" not in m.group(2):
            return f"{m.group(1)} ({m.group(2).rstrip()})"
    # A dash quoted as a literal (``—``, "—") is a value, not punctuation: never touched.
    def _literal(m):
        return (m.group(0) == "—" and m.start() > 0 and m.end() < len(text)
                and text[m.start() - 1] in "`\"'" and text[m.end()] in "`\"'")
    matches = [m for m in DASH_RE.finditer(text) if not _literal(m)]
    if not matches:
        return text
    # Pair dashes that fall in the same sentence: "A — B — C" -> "A (B) C".
    roles: dict[int, str] = {}
    i = 0
    while i < len(matches):
        cur = matches[i]
        if i + 1 < len(matches):
            nxt = matches[i + 1]
            same_sentence = _sentence_end(text, cur.end()) >= nxt.start()
            inner = text[cur.end():nxt.start()]
            ok_inner = (
                inner.strip()
                and len(inner) < 180
                and ";" not in inner
                and inner.count("(") == inner.count(")")
                and inner.count("\n") <= 1
                and not LIST_MARK_RE.search(inner)
            )
            if same_sentence and ok_inner:
                roles[i] = "open"
                roles[i + 1] = "close"
                i += 2
                continue
        roles[i] = "single"
        i += 1
    out = []
    last = 0
    for idx, m in enumerate(matches):
        out.append(text[last:m.start()])
        right = text[m.end():]
        left = text[:m.start()]
        role = roles[idx]
        at_line_start = (not left) or left.endswith("\n")
        at_line_end = right.startswith("\n") or not right
        paren_inner = False
        if role in ("open", "close"):
            o = m if role == "open" else matches[idx - 1]
            c = matches[idx + 1] if role == "open" else m
            inner_txt = text[o.end():c.start()].strip()
            paren_inner = inner_txt.startswith("(") and inner_txt.endswith(")")
        nested = False
        if role in ("open", "close") and not paren_inner:
            nested = "(" in inner_txt or ")" in inner_txt
        if role == "open" and nested:
            rep = "" if at_line_start else ("," if at_line_end else ", ")
        elif role == "close" and nested:
            rep = "," if at_line_end else ", "
        elif role == "open" and paren_inner:
            rep = "" if at_line_start else " "
        elif role == "close" and paren_inner:
            word = _first_word(right)
            rep = "" if at_line_end else (", " if word in CLOSE_COMMA_WORDS else " ")
        elif role == "open":
            rep = "(" if at_line_start else " ("
        elif role == "close":
            word = _first_word(right)
            open_pos = matches[idx - 1].start()
            sent_start = _sentence_start(text, open_pos)
            sub = _first_word(text[sent_start:open_pos]) in SUBORDINATE_START
            comma = word in CLOSE_COMMA_WORDS or (sub and word not in CLOSE_COMMA_WORDS)
            rep = ")" if at_line_end else ("), " if comma else ") ")
            if at_line_end:
                rep = "),"  if sub else ")"
        else:
            word = _first_word(right)
            sent_left = text[_sentence_start(text, m.start()):m.start()]
            if right.lstrip().startswith("see ") or word == "see":
                sep = ";"
            elif (_sentence_start(text, m.start()) >= text.rfind("\n", 0, m.start()) + 1
                  and _is_label(_line_left(text, m.start()))):
                sep = ":"
            elif word in COMMA_WORDS:
                sep = ","
            elif _colon_already(sent_left):
                sep = ";"
            else:
                sep = ":"
            if at_line_start:
                rep = ""
                if out and out[-1].endswith("\n"):
                    prev = out[-1][:-1].rstrip(" \t")
                    if prev and prev[-1] not in ".:;,!?(\n":
                        out[-1] = prev + sep + "\n"
            else:
                rep = sep if at_line_end else sep + " "
        out.append(rep)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


FILLER_RE = re.compile(r"(?<![\w-])(genuinely|actually|simply|essentially)(,?)[ \t]+(?=\S)", re.I)
PHRASES = [
    (re.compile(r"\bquietly\b"), "silently"),
    (re.compile(r"\bQuietly\b"), "Silently"),
    (re.compile(r"\bthe honest answer\b"), "the correct answer"),
    (re.compile(r"\bthe honest number\b"), "the unbiased number"),
    (re.compile(r"\bthe honest default\b"), "the safer default"),
    (re.compile(r"\bthe honest representation\b"), "the accurate representation"),
    (re.compile(r"\bthe honest way\b"), "the sound way"),
    (re.compile(r"\bthe honest fallback\b"), "the safe fallback"),
    (re.compile(r"\bthe honest one\b"), "the right one"),
    (re.compile(r"\bthe more honest\b"), "the more accurate"),
]


def replace_words(text: str) -> str:
    """Drop filler adverbs and swap a few rhetorical phrases for plain ones."""
    out = []
    last = 0
    for m in FILLER_RE.finditer(text):
        out.append(text[last:m.start()])
        last = m.end()
        if m.group(1)[0].isupper() and last < len(text):
            # The sentence started with the dropped word: capitalise the next one.
            text = text[:last] + text[last].upper() + text[last + 1:]
    out.append(text[last:])
    result = "".join(out)
    for rx, rep in PHRASES:
        result = rx.sub(rep, result)
    return result


def replace_chars(text: str) -> str:
    for a, b in CHAR_MAP.items():
        text = text.replace(a, b)
    return text


def _doc_positions(tree: ast.AST) -> set[tuple[int, int]]:
    pos = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pos.add((node.value.lineno, node.value.col_offset))
    return pos


class _Blank(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.copy_location(ast.Expr(value=ast.Constant(value="<doc>")), node)
        return self.generic_visit(node)


def code_fingerprint(src: str) -> str:
    tree = ast.parse(src)
    return ast.dump(_Blank().visit(tree), include_attributes=False)


def _is_banner_candidate(comment: str) -> bool:
    if comment.startswith(("#!", "#:")) or "-*-" in comment:
        return False
    body = comment[1:].strip()
    low = body.lower()
    if low.startswith(("noqa", "type:", "pragma", "fmt:", "pylint", "isort")):
        return False
    return bool(re.match(r"^[─━═\-=*~#_]{3,}", body)) or bool(re.search(r"[─━═]{3,}\s*$", body)) or bool(re.search(r"[\-=]{4,}\s*$", body) and re.match(r"^[\-=]{2,}", body))


def _banner_rewrite(comment: str) -> str | None:
    """Return the replacement comment text, '' to delete the line, or None to leave it."""
    if not _is_banner_candidate(comment):
        return None
    body = comment[1:].strip()
    m = RULE_RE.match(body)
    if not m:
        return None
    title = (m.group("title") or "").strip()
    title = title.strip("─━═-=*~#_ ").strip()
    if not title:
        return ""
    return "# " + title


def _rewrite(text: str, *, words: bool, noqa: bool = False) -> str:
    out = replace_chars(replace_dashes(text, noqa=noqa))
    return replace_words(out) if words else out


def process_source(src: str, *, banners: bool = True, words: bool = True) -> tuple[str, list[tuple[int, str, str]]]:
    """Return ``(new_source, changes)``, with ``changes`` as ``(line, old, new)`` tuples."""
    tree = ast.parse(src)
    docpos = _doc_positions(tree)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.splitlines(keepends=True)

    # (start_row, start_col, end_row, end_col, new_text)
    edits: list[tuple[int, int, int, int, str]] = []
    comment_tokens = []
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            comment_tokens.append(tok)
        elif tok.type == tokenize.STRING and (tok.start[0], tok.start[1]) in docpos:
            new = _rewrite(tok.string, words=words)
            if new != tok.string:
                edits.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1], new))

    # Group full-line comments into blocks (same column, consecutive rows).
    full_line = {}
    for tok in comment_tokens:
        row, col = tok.start
        is_full = lines[row - 1][:col].strip() == ""
        full_line[row] = (tok, is_full)

    delete_rows: set[int] = set()
    rows = sorted(full_line)
    blocks: list[list] = []
    cur: list = []
    for row in rows:
        tok, is_full = full_line[row]
        if not is_full:
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([tok])
            continue
        if cur and (cur[-1].start[0] == row - 1) and cur[-1].start[1] == tok.start[1] and full_line[cur[-1].start[0]][1]:
            cur.append(tok)
        else:
            if cur:
                blocks.append(cur)
            cur = [tok]
    if cur:
        blocks.append(cur)

    for block in blocks:
        is_full = full_line[block[0].start[0]][1]
        texts = [t.string for t in block]
        # Comment blocks drawing a diagram keep their rule characters.
        diagram = any(ch in DIAGRAM_CHARS for t in texts for ch in t)
        new_texts = list(texts)
        if banners and is_full and not diagram:
            for k, t in enumerate(texts):
                rep = _banner_rewrite(t)
                if rep is None:
                    continue
                if rep == "":
                    delete_rows.add(block[k].start[0])
                    new_texts[k] = None
                else:
                    new_texts[k] = rep
        # Dashes are rewritten on the joined block, so a pair can span lines.
        keep_idx = [k for k, t in enumerate(new_texts) if t is not None]
        if keep_idx:
            prefixes = []
            bodies = []
            for k in keep_idx:
                t = new_texts[k]
                m = re.match(r"(#[:!]?\s?)(.*)", t, re.S)
                prefixes.append(m.group(1))
                bodies.append(m.group(2))
            joined = "\n".join(bodies)
            noqa = len(keep_idx) == 1 and "noqa" in joined
            joined2 = _rewrite(joined, words=words, noqa=noqa)
            parts = joined2.split("\n")
            assert len(parts) == len(bodies), (texts, joined2)
            for k, p, pre in zip(keep_idx, parts, prefixes):
                new_texts[k] = (pre + p).rstrip()
        for tok, old, new in zip(block, texts, new_texts):
            if new is None:
                continue
            if new != old:
                edits.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1], new))

    # Apply edits from the end so earlier positions stay valid.
    changes = []
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for (sr, sc, er, ec, new) in edits:
        if sr == er:
            line = lines[sr - 1]
            old = line[sc:ec]
            lines[sr - 1] = line[:sc] + new + line[ec:]
            changes.append((sr, old, new))
        else:
            first = lines[sr - 1]
            last = lines[er - 1]
            old = first[sc:] + "".join(lines[sr:er - 1]) + last[:ec]
            merged = first[:sc] + new + last[ec:]
            new_lines = merged.splitlines(keepends=True)
            assert len(new_lines) == er - sr + 1, "line count changed inside a docstring"
            lines[sr - 1:er] = new_lines
            changes.append((sr, old, new))

    # Delete pure rule lines and collapse the blank runs they leave behind to two lines.
    def blank(i):
        return 0 <= i < len(lines) and lines[i].strip() == ""

    for row in sorted(delete_rows, reverse=True):
        idx = row - 1
        del lines[idx]
        start = idx
        while start - 1 >= 0 and blank(start - 1):
            start -= 1
        end = idx
        while end < len(lines) and blank(end):
            end += 1
        run = end - start
        if run > 2:
            del lines[start + 2:end]
        changes.append((row, "<rule line>", "<deleted>"))

    return "".join(lines), changes


def process_text(text: str, *, words: bool = True) -> str:
    """Full-line ``#`` comments of a TOML / YAML / shell file; everything else is kept verbatim."""
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            indent = line[: len(line) - len(stripped)]
            rep = _banner_rewrite(stripped)
            if rep == "":
                continue
            stripped = rep if rep is not None else stripped
            line = (indent + "#" + _rewrite(stripped[1:], words=words)).rstrip()
        out.append(line)
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write", action="store_true", help="rewrite the files in place")
    ap.add_argument("--no-banners", action="store_true", help="leave rule lines alone")
    ap.add_argument("--no-words", action="store_true", help="leave filler words alone")
    ap.add_argument("--text", action="store_true", help="treat FILEs as TOML/YAML/shell, not Python")
    ap.add_argument("--report", help="write every edit to this TSV file")
    args = ap.parse_args(argv)
    if args.text:
        for path in args.files:
            text = open(path, encoding="utf-8").read()
            new = process_text(text, words=not args.no_words)
            if new != text:
                print("changed", path)
                if args.write:
                    open(path, "w", encoding="utf-8").write(new)
        return 0
    report = []
    bad = 0
    for path in args.files:
        src = open(path, encoding="utf-8").read()
        try:
            new, changes = process_source(src, banners=not args.no_banners, words=not args.no_words)
        except SyntaxError as exc:
            print(f"SKIP (syntax) {path}: {exc}", file=sys.stderr)
            continue
        if new == src:
            continue
        if code_fingerprint(new) != code_fingerprint(src):
            print(f"FINGERPRINT MISMATCH {path}", file=sys.stderr)
            bad += 1
            continue
        for row, old, newt in changes:
            report.append(f"{path}\t{row}\t{old!r}\t{newt!r}")
        if args.write:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report) + "\n")
    print(f"files changed: {len({r.split(chr(9))[0] for r in report})}, edits: {len(report)}, fingerprint mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
