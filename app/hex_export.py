"""Deterministic Intel HEX normalization/export.

The parser assembles any Intel HEX text into canonical, address-ordered
contiguous segments plus an optional entry address. This module renders
that assembled image back into Intel HEX text with a fully deterministic
record layout, so the *same logical image* always exports the *same
text* no matter how the input bytes were split across records or how
the base address was carried (02/04 records):

* segments are walked in ascending address order and emitted as type-00
  data records of at most ``max_data_bytes`` bytes;
* a data record never crosses a 64 KiB address window (its 16-bit
  offset field cannot address beyond it), so chunks are split at window
  boundaries and every window switch emits a fresh type-04 extended
  linear address record — the first window included, so the text never
  relies on an implicit initial base and type-02 bases never appear;
* when ``include_entry`` is true and the image defines an entry, a
  type-05 start linear address record is appended after the data;
* every record carries a freshly computed two's-complement checksum and
  the text ends with the mandatory type-01 end-of-file record, one
  record per line.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hex_parser import Segment

WINDOW_SIZE = 0x10000  # 64 KiB window addressable by a record's 16-bit offset
MAX_DATA_BYTES_LIMIT = 0xFF  # the record byte-count field is a single byte
ADDRESS_SPACE_LIMIT = 0x1_0000_0000  # exported addresses must stay 32-bit

RECORD_TYPE_DATA = 0x00
RECORD_TYPE_EOF = 0x01
RECORD_TYPE_LINEAR_BASE = 0x04
RECORD_TYPE_ENTRY = 0x05


@dataclass
class HexExport:
    """The normalized Intel HEX rendering of an assembled image."""

    hex_text: str  # one record per line, trailing newline included
    record_count: int  # every emitted record: data + linear-base + entry + EOF
    data_record_count: int  # type-00 records only
    linear_base_record_count: int  # type-04 records only
    total_bytes: int  # data bytes carried by the type-00 records
    entry_address: int | None  # entry of the source image, if any
    include_entry: bool  # whether the entry record was emitted
    max_data_bytes: int  # the chunking bound that was applied


def format_record(rectype: int, offset: int, data: bytes = b"") -> str:
    """Render one Intel HEX record with a fresh two's-complement checksum."""
    body = bytes([len(data), (offset >> 8) & 0xFF, offset & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


def export_intel_hex(
    segments: list[Segment],
    entry_address: int | None,
    max_data_bytes: int = 16,
    include_entry: bool = True,
) -> HexExport:
    """Normalize assembled segments into deterministic Intel HEX text.

    Segments are sorted defensively: the parser already returns them in
    ascending address order, but the exported text must never depend on
    caller-supplied ordering. Raises ``ValueError`` when
    ``max_data_bytes`` is outside 1..255 or any address exceeds the
    32-bit space the record format can express.
    """
    if not 1 <= max_data_bytes <= MAX_DATA_BYTES_LIMIT:
        raise ValueError(
            f"max_data_bytes must be within 1..{MAX_DATA_BYTES_LIMIT}, "
            f"got {max_data_bytes}"
        )
    ordered = sorted(segments, key=lambda s: s.start)
    for segment in ordered:
        if segment.start < 0 or segment.end > ADDRESS_SPACE_LIMIT:
            raise ValueError(
                f"segment [{segment.start:#x}, {segment.end:#x}) exceeds the "
                "32-bit address space"
            )
    if entry_address is not None and not 0 <= entry_address < ADDRESS_SPACE_LIMIT:
        raise ValueError(f"entry address {entry_address:#x} exceeds the 32-bit space")

    lines: list[str] = []
    data_records = 0
    linear_base_records = 0
    current_window: int | None = None

    for segment in ordered:
        address = segment.start
        pos = 0
        while pos < len(segment.data):
            window = address >> 16
            if window != current_window:
                lines.append(
                    format_record(RECORD_TYPE_LINEAR_BASE, 0, window.to_bytes(2, "big"))
                )
                linear_base_records += 1
                current_window = window
            # Never let a record cross the 64 KiB window its offset lives in.
            window_remaining = WINDOW_SIZE - (address & 0xFFFF)
            chunk_len = min(max_data_bytes, window_remaining, len(segment.data) - pos)
            lines.append(
                format_record(
                    RECORD_TYPE_DATA,
                    address & 0xFFFF,
                    segment.data[pos : pos + chunk_len],
                )
            )
            data_records += 1
            pos += chunk_len
            address += chunk_len

    if include_entry and entry_address is not None:
        lines.append(
            format_record(RECORD_TYPE_ENTRY, 0, entry_address.to_bytes(4, "big"))
        )
    lines.append(format_record(RECORD_TYPE_EOF, 0))

    return HexExport(
        hex_text="\n".join(lines) + "\n",
        record_count=len(lines),
        data_record_count=data_records,
        linear_base_record_count=linear_base_records,
        total_bytes=sum(len(s.data) for s in ordered),
        entry_address=entry_address,
        include_entry=include_entry,
        max_data_bytes=max_data_bytes,
    )
