"""Generated differential tests only; no real data, metrics, or model loading."""
import io
import json
from pathlib import Path
import random

import pytest

from nhis_fairbias.benchmark import result_catalog as original
from nhis_fairbias.benchmark import catalog_json_reader as fast


def legacy_read_metadata(path, fields=None):
    try:
        with Path(path).open(encoding='utf-8') as handle:
            return original._JSONStream(handle).object(fields)
    except original.CatalogError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise original.CatalogError('UNREADABLE_METADATA') from None


def outcome(scanner, text, fields):
    try:
        return ('accepted', scanner(io.StringIO(text)).object(fields))
    except Exception as exc:
        return ('error', type(exc).__name__, str(exc))


def same(text, fields):
    assert outcome(fast.FastJSONStream, text, fields) == outcome(original._JSONStream, text, fields)


@pytest.mark.parametrize('fields', [None, set(), {'status'}, {'x'}, {'diagnostics'}])
def test_explicit_adversarial_and_depth_cases(fields):
    cases = ['[]', '', '{', ' ', '{}tail', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
        '{"x":1,"x":2}', '{"x":1,"\\u0078":2}', '{"diagnostics":{"a":1,"a":2}}',
        '{"x":1e9999}', '{"x":' + '1'*1024 + '}', '{"x":' + '1'*1025 + '}',
        '{"diagnostics":[1,]}', '{"diagnostics":"bad\\q"}', '{"x":"\\u12"}',
        '{"x":"\\u12', '{"x":"\\u123X"}', '{"x":"\\ud800"}', '{"x":"\\ud800\\udfff"}',
        '{"x":"a\x00b"}', '{"x":"\\"}', '{"x":01}', '{"x":-0}', '{"x":1.}', '{"x":.2}',
        '{"x":1e}', '{"x":--1}', '{"x":nullx}', '{"x":true false}', '{"x":1\v}',
        '{"x":{1:2}}', '{"x":{"x" 2}}', '{"x":[1 2]}', '{"x":[1,2}', '{"x":[]}',
        '{"x":{}}', '{"x": [ true,false,null,-1.2e+3, "a", [4] ]}', '{"x":[,,]}',
        '{"x":[1,2,3,4,5,6,7,8,9,10,11,12,]}']
    for n in (97, 98, 99, 100, 101):
        cases += ['{"x":' + '['*n + '0' + ']'*n + '}', '{"x":' + '['*n + ']'*n + '}',
            '{"x":' + '{"k":'*n + '0' + '}'*n + '}']
    for text in cases:
        same(text, fields)


def test_generated_valid_invalid_and_every_truncation():
    rng = random.Random(20260917)
    def value(depth=0):
        if depth >= 5 or rng.random() < .6:
            return rng.choice([None, True, False, rng.randint(-10**12, 10**12), rng.uniform(-1e20, 1e20),
                ''.join(rng.choice('abc汉字\\\"\t\n012') for _ in range(rng.randrange(30)))])
        if rng.random() < .5:
            return [value(depth+1) for _ in range(rng.randrange(8))]
        return {str(i): value(depth+1) for i in range(rng.randrange(8))}
    count = 0
    for i in range(500):
        text = json.dumps({'status': 'VALID', 'x': value(), 'diagnostics': value()}, ensure_ascii=i % 2 == 0,
            indent=2 if i % 3 == 0 else None)
        for fields in (None, set(), {'status'}, {'x'}):
            same(text, fields); count += 1
            for _ in range(4):
                where = rng.randrange(len(text))
                damaged = text[:where] + rng.choice([',', ':', '\\', '\x01', '[', '}', 'N', ' ', '"']) + text[where+1:]
                same(damaged, fields); count += 1
        if i < 10:
            for end in range(len(text)):
                same(text[:end], {'status'}); count += 1
    assert count > 10000


@pytest.mark.parametrize('chunk', [1, 2, 3, 7, 65535, 65536, 65537])
def test_chunk_boundaries(chunk):
    class Chunk(io.StringIO):
        def read(self, n=-1):
            return super().read(min(n, chunk))
    for boundary in (65535, 65536, 65537):
        text = '{"status":"VALID","diagnostics":"' + 'a' * boundary + '\\u0041\\\\\\\"尾"}'
        assert fast.FastJSONStream(Chunk(text)).object({'status'}) == original._JSONStream(Chunk(text)).object({'status'})


def test_excluded_large_string_numbers_and_no_decode(monkeypatch):
    calls = []
    actual = json.loads
    def restricted(text, *a, **k):
        calls.append(text)
        assert text in ('"status"', '"VALID"', '"diagnostics"')
        return actual(text, *a, **k)
    monkeypatch.setattr(json, 'loads', restricted)
    excluded = '{"blob":"' + 'z' * 2_000_000 + '","array":[' + ','.join(['1.234e-10'] * 100_000) + ']}'
    text = '{"status":"VALID","diagnostics":' + excluded + '}'
    assert fast.FastJSONStream(io.StringIO(text)).object({'status'}) == {'status': 'VALID'}
    assert calls == ['"status"', '"VALID"', '"diagnostics"']


def test_explicit_memory_cap():
    with pytest.raises(original.CatalogError, match='METADATA_DOCUMENT_LIMIT'):
        fast.FastJSONStream(io.StringIO('{"x":"' + 'a' * 100 + '"}'), max_document_chars=20)


def test_file_size_fallback_with_small_policy_cap(tmp_path, monkeypatch):
    path = tmp_path / 'generated.json'
    text = '{"status":"VALID","diagnostics":[' + ','.join(['0.2'] * 2000) + ']}'
    path.write_text(text)
    seen = []
    legacy = fast._LEGACY_STREAM
    class Recorder(legacy):
        def __init__(self, handle):
            seen.append(True)
            super().__init__(handle)
    monkeypatch.setattr(fast, '_LEGACY_STREAM', Recorder)
    monkeypatch.setattr(fast, 'MAX_DOCUMENT_BYTES', 128)
    assert fast.read_metadata(path, {'status'}) == legacy_read_metadata(path, {'status'})
    assert seen == [True]
    for text in ('{"x":[1,]}', '{"x":' + '[' * 100 + '0' + ']' * 100 + '}'):
        path.write_text(' ' * 129 + text)
        with pytest.raises(original.CatalogError) as first:
            legacy_read_metadata(path)
        with pytest.raises(original.CatalogError) as second:
            fast.read_metadata(path)
        assert str(first.value) == str(second.value)


def test_adversarial_regex_time_bound():
    import time
    cases = ['{"x":[' + '9' * 2_000_000 + ']}',
        '{"x":[' + ','.join(['1.2e-4'] * 200_000) + ',]}',
        '{"x":[' + '9' * 1023 + 'e' + '9' * 2_000_000 + ']}',
        '{"x":"' + 'z' * 2_000_000 + '\x00"}',
        '{"x":[' + ','.join(['true'] * 200_000) + ',falsefalse]}']
    start = time.perf_counter()
    for text in cases:
        outcome(fast.FastJSONStream, text, set())
    assert time.perf_counter() - start < 5.0


def test_primitive_run_cannot_bypass_depth_limit():
    for depth in (98, 99, 100):
        for ending in ('0', ','.join(['0'] * 5000), '[]', '[0,1,2]'):
            text = '{"diagnostics":' + '[' * depth + ending + ']' * depth + '}'
            for fields in (None, {'status'}):
                same(text, fields)


def test_invalid_utf8_keeps_legacy_first_error(tmp_path):
    cases = [b'{"x":!', b'{"x":"', b'{"status":"VALID","diagnostics":"']
    for prefix in cases:
        for boundary in (65535, 65536, 65537, 1024 * 1024):
            path = tmp_path / 'invalid_utf8.json'
            path.write_bytes(prefix + b'x' * boundary + b'\xff"}')
            results = []
            for reader in (legacy_read_metadata, fast.read_metadata):
                try:
                    reader(path, {'status'})
                except original.CatalogError as exc:
                    results.append(str(exc))
            assert len(results) == 2 and results[0] == results[1]
