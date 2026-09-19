"""Bounded lexical JSON scanner for metadata-only provenance verification.

Excluded values are never decoded. Top-level keys and retained values alone
use json.loads. The legacy scanner's accepted language/errors are preserved
within a 64 MiB file fast-path limit. Larger files use the unchanged legacy
stream; no new metadata size acceptance limit is introduced. Input buffering
is capped at 64 Mi characters even if a file grows after its size check.
"""
import json
from pathlib import Path
import re

from nhis_fairbias.benchmark.result_catalog import CatalogError, _require, _unique_object, _JSONStream as _LEGACY_STREAM

MAX_DOCUMENT_CHARS = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_SPACE = re.compile(r'[ \t\r\n]*')
_SPECIAL = re.compile(r'["\\\x00-\x1f]')
_HEX4 = re.compile(r'[0-9a-fA-F]{4}')
_TOKEN = re.compile(r'[^,\]} \t\r\n]+')
_NUMBER = r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_SCALAR = re.compile(r'(?:true|false|null|' + _NUMBER + r')\Z')
# At most 2048 primitive elements per C-regex match; never materialize values.
# The lookahead enforces the legacy 1024-character scalar limit independently
# of numeric grammar. A trailing comma is consumed only with an expectation
# that another value follows, so malformed [1,] still reaches the normal error.
_ATOM = r'(?=[^,\]} \t\r\n]{1,1024}(?:[,\]} \t\r\n]))(?:true|false|null|' + _NUMBER + r')'
_ARRAY_RUN = re.compile(r'(?:[ \t\r\n]*' + _ATOM + r'[ \t\r\n]*,){1,2048}')


class FastJSONStream:
    def __init__(self, handle, *, max_document_chars=MAX_DOCUMENT_CHARS):
        pieces, total = [], 0
        while True:
            piece = handle.read(min(1024 * 1024, max_document_chars + 1 - total))
            if not piece:
                break
            total += len(piece)
            _require(total <= max_document_chars, 'METADATA_DOCUMENT_LIMIT')
            pieces.append(piece)
        self.text, self.pos = ''.join(pieces), 0

    def peek(self):
        return self.text[self.pos:self.pos + 1]

    def take(self):
        value = self.peek()
        _require(bool(value), 'TRUNCATED_JSON')
        self.pos += 1
        return value

    def space(self):
        self.pos = _SPACE.match(self.text, self.pos).end()

    def value(self, keep=False, depth=0):
        _require(depth < 100, 'JSON_NESTING_LIMIT')
        self.space()
        start, token = self.pos, self.peek()
        if token == '"':
            self.pos += 1
            while True:
                match = _SPECIAL.search(self.text, self.pos)
                if match is None:
                    self.pos = len(self.text)
                    raise CatalogError('TRUNCATED_JSON')
                self.pos = match.end()
                char = match.group()
                if char == '"':
                    break
                _require(char == '\\', 'INVALID_JSON_STRING')
                escaped = self.take()
                _require(escaped in '"\\/bfnrtu', 'INVALID_JSON_ESCAPE')
                if escaped == 'u':
                    # Preserve truncated-vs-invalid escape error order.
                    for offset in range(4):
                        if self.pos + offset >= len(self.text):
                            raise CatalogError('TRUNCATED_JSON')
                        _require(self.text[self.pos + offset] in '0123456789abcdefABCDEF', 'INVALID_JSON_ESCAPE')
                    self.pos += 4
        elif token in ('{', '['):
            opening = self.take()
            closing = '}' if opening == '{' else ']'
            self.space()
            if self.peek() != closing:
                while True:
                    if opening == '{':
                        _require(self.peek() == '"', 'INVALID_JSON_OBJECT')
                        self.value(False, depth + 1)
                        self.space()
                        _require(self.take() == ':', 'INVALID_JSON_OBJECT')
                    elif not keep and depth + 1 < 100:
                        # Only lexical scalar runs: no key/value decoding.
                        while True:
                            run = _ARRAY_RUN.match(self.text, self.pos)
                            if run is None:
                                break
                            self.pos = run.end()
                    self.value(False, depth + 1)
                    self.space()
                    if self.peek() == closing:
                        break
                    _require(self.take() == ',', 'INVALID_JSON_COLLECTION')
                    self.space()
            self.take()
        else:
            match = _TOKEN.match(self.text, self.pos)
            if match is None:
                raise CatalogError('INVALID_JSON_SCALAR')
            self.pos = match.end()
            _require(self.pos - start <= 1024, 'INVALID_JSON_SCALAR')
            _require(_SCALAR.fullmatch(match.group()) is not None, 'INVALID_JSON_SCALAR')
        return self.text[start:self.pos] if keep else None

    def object(self, fields=None):
        self.space()
        _require(self.take() == '{', 'JSON_OBJECT_REQUIRED')
        result, seen = {}, set()
        self.space()
        if self.peek() != '}':
            while True:
                _require(self.peek() == '"', 'INVALID_JSON_OBJECT')
                key = json.loads(self.value(True))
                _require(key not in seen, 'DUPLICATE_JSON_KEY')
                seen.add(key)
                self.space()
                _require(self.take() == ':', 'INVALID_JSON_OBJECT')
                keep = fields is None or key in fields
                raw = self.value(keep)
                if keep:
                    result[key] = json.loads(raw, object_pairs_hook=_unique_object,
                        parse_constant=lambda _: (_ for _ in ()).throw(CatalogError('NONFINITE_JSON')))
                self.space()
                if self.peek() == '}':
                    break
                _require(self.take() == ',', 'INVALID_JSON_OBJECT')
                self.space()
        self.take()
        self.space()
        _require(not self.peek(), 'TRAILING_JSON_CONTENT')
        return result


def read_metadata(path, fields=None):
    try:
        path = Path(path)
        with path.open(encoding='utf-8') as handle:
            if path.stat().st_size > MAX_DOCUMENT_BYTES:
                return _LEGACY_STREAM(handle).object(fields)
            try:
                return FastJSONStream(handle).object(fields)
            except UnicodeError:
                # Match the legacy first-error order for malformed UTF-8 far
                # beyond an earlier JSON syntax error or chunk boundary.
                handle.seek(0)
                return _LEGACY_STREAM(handle).object(fields)
            except CatalogError as exc:
                if str(exc) != 'METADATA_DOCUMENT_LIMIT':
                    raise
                # A concurrently growing file can exceed the initial stat.
                # Rewind to the unchanged streaming implementation; do not
                # introduce a new acceptance limit on metadata file size.
                handle.seek(0)
                return _LEGACY_STREAM(handle).object(fields)
    except CatalogError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise CatalogError('UNREADABLE_METADATA') from None
