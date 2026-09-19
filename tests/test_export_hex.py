"""Tests for the Intel HEX normalization module and /api/v1/export-hex."""

import pytest
from fastapi.testclient import TestClient

from app.hex_export import (
    MAX_DATA_BYTES_LIMIT,
    WINDOW_SIZE,
    export_intel_hex,
    format_record,
)
from app.hex_parser import Segment, parse_intel_hex
from app.main import app

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def linear_base(value: int) -> str:
    return rec(0x04, 0, value.to_bytes(2, "big"))


def linear_entry(value: int) -> str:
    return rec(0x05, 0, value.to_bytes(4, "big"))


def decode_line(line: str) -> tuple[int, int, int, bytes]:
    """Decode one record into (byte_count, offset, rectype, data)."""
    assert line.startswith(":")
    raw = bytes.fromhex(line[1:])
    assert len(raw) == raw[0] + 5
    assert sum(raw) & 0xFF == 0, "checksum must sum to 0 mod 256"
    return raw[0], (raw[1] << 8) | raw[2], raw[3], raw[4:-1]


def export_lines(export) -> list[str]:
    lines = export.hex_text.splitlines()
    assert len(lines) == export.record_count
    assert export.hex_text.endswith("\n")
    return lines


# ---------------------------------------------------------------------------
# pure export logic
# ---------------------------------------------------------------------------

def test_single_segment_chunked_by_max_data_bytes():
    payload = bytes(range(40))
    export = export_intel_hex([Segment(0x0000, payload)], None, max_data_bytes=16)
    lines = export_lines(export)
    # the first window is established explicitly, then 16/16/8 data, then EOF
    assert [decode_line(line)[0] for line in lines] == [2, 16, 16, 8, 0]
    assert decode_line(lines[0])[2] == 0x04
    assert int.from_bytes(decode_line(lines[0])[3], "big") == 0
    assert export.data_record_count == 3
    assert export.linear_base_record_count == 1
    assert export.record_count == 5
    assert export.total_bytes == 40
    # offsets advance within the single 64 KiB window
    offsets = [decode_line(line)[1] for line in lines[1:-1]]
    assert offsets == [0x0000, 0x0010, 0x0020]
    assert lines[-1] == EOF


def test_every_record_checksum_is_valid_two_complement():
    payload = bytes(range(64))
    export = export_intel_hex(
        [Segment(0x1234, payload)], 0x08001234, max_data_bytes=7
    )
    for line in export_lines(export):
        raw = bytes.fromhex(line[1:])
        assert sum(raw) & 0xFF == 0
        assert raw[3] in (0x00, 0x04, 0x05, 0x01)


def test_record_never_crosses_64k_window_and_switch_emits_linear_base():
    # 32 bytes straddling the 0xFFFF/0x10000 boundary.
    payload = bytes(range(32))
    export = export_intel_hex([Segment(0xFFF0, payload)], None, max_data_bytes=16)
    lines = export_lines(export)
    kinds = [decode_line(line)[2] for line in lines]
    assert kinds == [0x04, 0x00, 0x04, 0x00, 0x01]
    first = decode_line(lines[1])
    assert first[1] == 0xFFF0 and first[0] == 16  # stops exactly at the boundary
    base = decode_line(lines[2])
    assert int.from_bytes(base[3], "big") == 1  # upper 16 bits of 0x10000
    second = decode_line(lines[3])
    assert second[1] == 0x0000 and second[0] == 16
    assert export.linear_base_record_count == 2


def test_unaligned_window_boundary_splits_mid_chunk():
    # 4 bytes starting 2 bytes before the boundary: 2 + 2, not 4.
    export = export_intel_hex([Segment(0xFFFE, b"\x01\x02\x03\x04")], None, 16)
    chunks = [decode_line(line) for line in export_lines(export)]
    assert chunks[0][2] == 0x04
    assert (chunks[1][1], chunks[1][0]) == (0xFFFE, 2)
    assert chunks[2][2] == 0x04
    assert (chunks[3][1], chunks[3][0]) == (0x0000, 2)


def test_linear_base_record_emitted_once_per_window():
    segments = [
        Segment(0x0000, b"\x01" * 4),
        Segment(0x0008, b"\x02" * 4),  # same window: no new base record
        Segment(0x1_0000, b"\x03" * 4),
        Segment(0x2_0000, b"\x04" * 4),
    ]
    export = export_intel_hex(segments, None, max_data_bytes=16)
    kinds = [decode_line(line)[2] for line in export_lines(export)]
    assert kinds == [0x04, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x01]
    assert export.linear_base_record_count == 3
    assert export.data_record_count == 4


def test_entry_record_emitted_before_eof_when_requested():
    export = export_intel_hex(
        [Segment(0x08000000, b"\xAA")], 0x080001C1, max_data_bytes=16
    )
    lines = export_lines(export)
    kinds = [decode_line(line)[2] for line in lines]
    assert kinds == [0x04, 0x00, 0x05, 0x01]
    entry = decode_line(lines[2])
    assert int.from_bytes(entry[3], "big") == 0x080001C1
    assert export.entry_address == 0x080001C1
    assert export.include_entry is True


def test_include_entry_false_omits_entry_record():
    export = export_intel_hex(
        [Segment(0, b"\xAA")], 0x080001C1, max_data_bytes=16, include_entry=False
    )
    kinds = [decode_line(line)[2] for line in export_lines(export)]
    assert 0x05 not in kinds
    assert export.entry_address == 0x080001C1  # still reported
    assert export.include_entry is False


def test_empty_image_exports_only_eof():
    export = export_intel_hex([], None, max_data_bytes=16)
    assert export.hex_text == EOF + "\n"
    assert export.record_count == 1
    assert export.data_record_count == 0
    assert export.total_bytes == 0


def test_empty_image_with_entry_exports_entry_and_eof():
    export = export_intel_hex([], 0x08000000, max_data_bytes=16)
    kinds = [decode_line(line)[2] for line in export_lines(export)]
    assert kinds == [0x05, 0x01]


def test_max_data_bytes_bound_validated():
    with pytest.raises(ValueError):
        export_intel_hex([Segment(0, b"\x01")], None, max_data_bytes=0)
    with pytest.raises(ValueError):
        export_intel_hex(
            [Segment(0, b"\x01")], None, max_data_bytes=MAX_DATA_BYTES_LIMIT + 1
        )
    # the single-byte count limit itself is accepted
    export = export_intel_hex(
        [Segment(0, bytes(300))], None, max_data_bytes=MAX_DATA_BYTES_LIMIT
    )
    sizes = [
        decode_line(line)[0]
        for line in export_lines(export)
        if decode_line(line)[2] == 0x00
    ]
    assert sizes == [255, 45]


def test_out_of_range_addresses_rejected():
    with pytest.raises(ValueError):
        export_intel_hex([Segment(0xFFFF0000, bytes(0x10001))], None, 16)
    with pytest.raises(ValueError):
        export_intel_hex([], 0x1_0000_0000, 16)


def test_segments_sorted_defensively():
    export = export_intel_hex(
        [Segment(0x0010, b"\x03"), Segment(0x0000, b"\x01\x02")], None, 16
    )
    offsets = [
        decode_line(line)[1]
        for line in export_lines(export)
        if decode_line(line)[2] == 0x00
    ]
    assert offsets == [0x0000, 0x0010]


def test_format_record_matches_known_vector():
    assert format_record(0x00, 0x0000, bytes(range(16))) == (
        ":10000000000102030405060708090A0B0C0D0E0F78"
    )
    assert format_record(0x01, 0) == ":00000001FF"


# ---------------------------------------------------------------------------
# determinism / round-trip through the real parser
# ---------------------------------------------------------------------------

def _export(text: str, max_data_bytes: int = 16, include_entry: bool = True):
    parsed = parse_intel_hex(text)
    assert parsed.ok
    return export_intel_hex(
        parsed.segments,
        parsed.entry_address,
        max_data_bytes=max_data_bytes,
        include_entry=include_entry,
    )


def test_record_splitting_does_not_change_export():
    payload = bytes(range(32))
    one_record = "\n".join([linear_base(0x0800), data_rec(0x0000, payload), EOF])
    many_records = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, payload[:1]),
        data_rec(0x0001, payload[1:7]),
        data_rec(0x0007, payload[7:]),
        EOF,
    ])
    assert _export(one_record).hex_text == _export(many_records).hex_text


def test_segment_base_input_normalizes_to_linear_base_output():
    # A type-02 based input exports the same text as the type-04 equivalent.
    segment_based = "\n".join([
        rec(0x02, 0, (0x0800).to_bytes(2, "big")),  # base 0x8000
        data_rec(0x0000, b"\x01\x02"),
        EOF,
    ])
    linear_based = "\n".join([
        linear_base(0x0000),
        data_rec(0x8000, b"\x01\x02"),
        EOF,
    ])
    assert _export(segment_based).hex_text == _export(linear_based).hex_text


def test_round_trip_preserves_image_and_entry():
    payload = bytes(range(48))
    source = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, payload[:20]),
        data_rec(0x1000, payload[20:]),
        linear_entry(0x08000101),
        EOF,
    ])
    exported = _export(source, max_data_bytes=16).hex_text
    again = parse_intel_hex(exported)
    assert again.ok
    original = parse_intel_hex(source)
    assert [(s.start, s.data) for s in again.segments] == [
        (s.start, s.data) for s in original.segments
    ]
    assert again.entry_address == original.entry_address == 0x08000101
    # exporting the round-tripped text is a fixed point
    assert _export(exported).hex_text == exported


def test_round_trip_without_entry_omits_entry_record():
    source = "\n".join([data_rec(0x0000, b"\x01\x02"), linear_entry(0x08000101), EOF])
    exported = _export(source, include_entry=False).hex_text
    again = parse_intel_hex(exported)
    assert again.ok
    assert again.entry_address is None


def test_export_spans_multiple_windows_round_trip():
    segments = [Segment(0xFFF8, bytes(range(32)))]  # crosses into window 1
    export = export_intel_hex(segments, None, max_data_bytes=16)
    again = parse_intel_hex(export.hex_text)
    assert again.ok
    assert [(s.start, s.data) for s in again.segments] == [(0xFFF8, bytes(range(32)))]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_export_happy_path():
    text = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, bytes(range(20))),
        linear_entry(0x080001C1),
        EOF,
    ])
    resp = client.post(
        "/api/v1/export-hex",
        json={"hex_text": text, "max_data_bytes": 8, "include_entry": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["error_count"] == 0
    assert body["entry_address"] == 0x080001C1
    assert body["include_entry"] is True
    assert body["max_data_bytes"] == 8
    assert body["total_bytes"] == 20
    assert body["segment_count"] == 1
    # 20 bytes at 8/record -> 3 data records, +1 linear base, +1 entry, +1 EOF
    assert body["data_record_count"] == 3
    assert body["linear_base_record_count"] == 1
    assert body["record_count"] == 6
    lines = body["hex_text"].splitlines()
    assert len(lines) == body["record_count"]
    assert lines[0] == linear_base(0x0800)
    assert lines[-1] == EOF
    assert decode_line(lines[-2])[2] == 0x05
    for line in lines:
        decode_line(line)  # raises unless count/checksum are consistent
    # every data record respects the 8-byte cap and the 64 KiB window
    for line in lines:
        count, offset, rectype, _ = decode_line(line)
        if rectype == 0x00:
            assert count <= 8
            assert offset + count <= WINDOW_SIZE


def test_api_export_defaults():
    text = "\n".join([data_rec(0x0000, b"\x01\x02"), EOF])
    body = client.post("/api/v1/export-hex", json={"hex_text": text}).json()
    assert body["ok"] is True
    assert body["max_data_bytes"] == 16
    assert body["include_entry"] is True
    assert body["hex_text"] == (
        linear_base(0x0000) + "\n" + data_rec(0x0000, b"\x01\x02") + "\n" + EOF + "\n"
    )
    assert body["record_count"] == 3


def test_api_export_include_entry_false():
    text = "\n".join([
        data_rec(0x0000, b"\x01"),
        linear_entry(0x08000100),
        EOF,
    ])
    body = client.post(
        "/api/v1/export-hex", json={"hex_text": text, "include_entry": False}
    ).json()
    assert body["ok"] is True
    assert body["include_entry"] is False
    assert body["entry_address"] == 0x08000100  # still echoed
    kinds = [decode_line(line)[2] for line in body["hex_text"].splitlines()]
    assert 0x05 not in kinds


def test_api_export_same_image_same_text_regardless_of_input_split():
    payload = bytes(range(16))
    one = "\n".join([linear_base(0x0800), data_rec(0x0000, payload), EOF])
    two = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, payload[:3]),
        data_rec(0x0003, payload[3:10]),
        data_rec(0x000A, payload[10:]),
        EOF,
    ])
    a = client.post("/api/v1/export-hex", json={"hex_text": one}).json()
    b = client.post("/api/v1/export-hex", json={"hex_text": two}).json()
    assert a["hex_text"] == b["hex_text"]
    assert a["record_count"] == b["record_count"]


def test_api_parse_failure_returns_diagnostics_and_no_text():
    resp = client.post(
        "/api/v1/export-hex", json={"hex_text": ":00000001FE\n"}
    )  # bad checksum, no EOF
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_count"] >= 1
    assert body["hex_text"] is None
    assert body["record_count"] == 0
    assert body["data_record_count"] == 0
    assert body["linear_base_record_count"] == 0
    assert body["total_bytes"] == 0
    assert body["segment_count"] == 0
    assert body["entry_address"] is None
    assert len(body["diagnostics"]) == body["error_count"] + body["warning_count"]
    # diagnostics match the shared parser exactly
    parsed = parse_intel_hex(":00000001FE\n")
    assert [d["message"] for d in body["diagnostics"]] == [
        d.message for d in parsed.diagnostics
    ]


def test_api_max_data_bytes_validation():
    text = "\n".join([data_rec(0x0000, b"\x01"), EOF])
    for bad in (0, -1, 256):
        resp = client.post(
            "/api/v1/export-hex", json={"hex_text": text, "max_data_bytes": bad}
        )
        assert resp.status_code == 422
    resp = client.post(
        "/api/v1/export-hex", json={"hex_text": text, "max_data_bytes": 255}
    )
    assert resp.status_code == 200
