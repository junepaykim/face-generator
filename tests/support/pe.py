"""A minimal PE32+ file for the launcher-copy tests: headers, one section, then what follows."""

from __future__ import annotations

import struct

PE_AT = 0x40
OPTIONAL_AT = PE_AT + 24
OPTIONAL_SIZE = 112 + 16 * 8
SECURITY_AT = OPTIONAL_AT + 112 + 4 * 8
"""The security (Authenticode) entry of the data directories."""
SECTION_AT = 0x200
SECTION_SIZE = 0x200
IMAGE_SIZE = SECTION_AT + SECTION_SIZE


def synthetic_pe(*, extra: bytes = b"", cert: bytes = b"") -> bytes:
    """The headers and one section, then `extra` (data appended after the sections), then
    `cert` (named by the security entry, as a signature is)."""
    data = bytearray(IMAGE_SIZE)
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, PE_AT)
    data[PE_AT : PE_AT + 4] = b"PE\0\0"
    struct.pack_into("<HH", data, PE_AT + 4, 0x8664, 1)  # x64, one section
    struct.pack_into("<H", data, PE_AT + 20, OPTIONAL_SIZE)
    struct.pack_into("<H", data, OPTIONAL_AT, 0x20B)  # PE32+
    struct.pack_into("<I", data, OPTIONAL_AT + 108, 16)  # data directories
    table = OPTIONAL_AT + OPTIONAL_SIZE
    data[table : table + 8] = b".text\0\0\0"
    struct.pack_into("<II", data, table + 16, SECTION_SIZE, SECTION_AT)
    data[SECTION_AT : SECTION_AT + 4] = b"code"
    if cert:
        struct.pack_into("<II", data, SECURITY_AT, IMAGE_SIZE + len(extra), len(cert))
    return bytes(data) + extra + cert
