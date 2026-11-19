"""Deterministic multi-format specification example extraction (Cycle AA2).

Z2 could read only assert statements and classic >>> doctests.  Many public
prompts state their examples as prose or as arrow lines.  This module extracts
executable (args, expected) pairs from four deterministic formats and nothing
else:

1. name(...) == value and name(...) -> value lines;
2. >>> name(...) doctest blocks (the caller already reads these);
3. Input: ... / Output: ... prose blocks, positional or named;
4. Example: name(...) == value blocks.

Every value must be a Python literal.  An example that names a non-literal
value is dropped, never guessed.  The extractor reports which formats fired so
a receipt can show that no external model and no task identity entered the
parse.  If nothing is verifiable, the caller abstains with the reason
no_verifiable_io_examples; this module never invents a post-condition.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LITERAL = object()

_CALL_EQUALS = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\(([^()\n]*)\)\s*==\s*([^\n#]+)")
_CALL_ARROW = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\(([^()\n]*)\)\s*->\s*([^\n#]+)")
_INPUT = re.compile(r"(?im)^\s*Input\s*:\s*(.+)$")
_OUTPUT = re.compile(r"(?im)^\s*Output\s*:\s*(.+)$")
_EXAMPLE_LINE = re.compile(r"(?im)^\s*Example\s*[:\-]\s*(.+)$")


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(str(text).strip().rstrip("."))
    except Exception:  # noqa: BLE001 - a non-literal is not an example
        return _LITERAL


def _split_arguments(text: str) -> List[str]:
    """Split a call argument list on top-level commas."""
    pieces: List[str] = []
    depth = 0
    current: List[str] = []
    quote: Optional[str] = None
    for char in str(text):
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if ord(char) in (34, 39):
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            pieces.append("".join(current))
            current = []
            continue
        current.append(char)
    if "".join(current).strip():
        pieces.append("".join(current))
    return [piece.strip() for piece in pieces if piece.strip()]


def _render_call(name: str, args: Sequence[Any], keywords: Dict[str, Any]) -> str:
    pieces = [repr(value) for value in args]
    pieces.extend("%s=%r" % (key, value) for key, value in keywords.items())
    return "%s(%s)" % (name, ", ".join(pieces))


def _positional(text: str, params: Sequence[str]) -> Tuple[List[Any], Dict[str, Any]]:
    """Read an argument list as positional values or name=value pairs."""
    args: List[Any] = []
    keywords: Dict[str, Any] = {}
    for piece in _split_arguments(text):
        if "=" in piece and not piece.strip().startswith(("[", "(", "{")):
            key, _, raw = piece.partition("=")
            key = key.strip()
            value = _literal(raw)
            if value is _LITERAL:
                raise ValueError("non-literal")
            if key in params:
                keywords[key] = value
            else:
                # A named value the signature does not declare is a literal
                # fact, but it cannot be bound to a parameter.
                raise ValueError("unknown_parameter")
        else:
            value = _literal(piece)
            if value is _LITERAL:
                raise ValueError("non-literal")
            args.append(value)
    return args, keywords


def extract(
    prompt: str,
    name: str,
    params: Sequence[str] = (),
) -> Dict[str, Any]:
    """Extract verifiable example assertions from one prompt.

    Returns {"assertions": [...], "formats": {...}, "dropped": int}.  The
    assertions are plain assert statements in the caller's namespace, so the
    existing sandbox verification consumes them unchanged.
    """
    text = str(prompt)
    name = str(name)
    params = [str(param) for param in params]
    assertions: List[str] = []
    formats: Dict[str, int] = {
        "call_equals": 0,
        "call_arrow": 0,
        "input_output": 0,
        "example_block": 0,
    }
    dropped = 0

    def emit(call: str, expected: Any) -> None:
        line = "assert (%s) == (%r)" % (call, expected)
        if line not in assertions:
            assertions.append(line)

    for pattern, key in ((_CALL_EQUALS, "call_equals"), (_CALL_ARROW, "call_arrow")):
        for match in pattern.finditer(text):
            if str(match.group(1)) != name:
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            if text[line_start : match.start()].strip().startswith("assert"):
                # An assert statement is already read by the caller; the
                # prose channel must not take credit for it.
                continue
            expected = _literal(match.group(3))
            if expected is _LITERAL:
                dropped += 1
                continue
            try:
                args, keywords = _positional(match.group(2), params)
            except ValueError:
                dropped += 1
                continue
            emit(_render_call(name, args, keywords), expected)
            formats[key] += 1

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        output_match = _OUTPUT.match(line)
        if output_match:
            # Look back at the nearest Input line inside the same block.
            back = index - 1
            while back >= 0 and index - back <= 4:
                input_match = _INPUT.match(lines[back])
                if input_match:
                    expected = _literal(output_match.group(1))
                    if expected is not _LITERAL:
                        try:
                            args, keywords = _positional(input_match.group(1), params)
                        except ValueError:
                            args, keywords = [], {}
                        if args or keywords:
                            emit(_render_call(name, args, keywords), expected)
                            formats["input_output"] += 1
                        else:
                            dropped += 1
                    else:
                        dropped += 1
                    break
                back -= 1
        index += 1

    for match in _EXAMPLE_LINE.finditer(text):
        body = str(match.group(1))
        for inner in _CALL_EQUALS.finditer(body):
            if str(inner.group(1)) != name:
                continue
            expected = _literal(inner.group(3))
            if expected is _LITERAL:
                dropped += 1
                continue
            try:
                args, keywords = _positional(inner.group(2), params)
            except ValueError:
                dropped += 1
                continue
            emit(_render_call(name, args, keywords), expected)
            formats["example_block"] += 1

    return {
        "assertions": assertions,
        "formats": formats,
        "dropped": int(dropped),
        "reason": "" if assertions else "no_verifiable_io_examples",
    }
