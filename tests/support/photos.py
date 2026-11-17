"""Synthetic Addressables catalog keys and facepack XML for photo discovery tests."""

from __future__ import annotations

import struct


def key(text: str) -> bytes:
    """One Addressables key as the catalog stores it: a little-endian length, then the text."""
    raw = text.encode()
    return struct.pack("<I", len(raw)) + raw


def catalog(*ids: int) -> bytes:
    guid = key("0003aebe2daea3c43bde98df9cb3e29e")
    body = b"".join(key(f"person_{i}") + guid + b"\x19t\x05\x00" for i in ids)
    # The silhouettes, a club key and stray digits elsewhere must not count.
    return body + key("person_default") + key("person_default_w") + key("club_1234567") + b"99"


def config_xml(*ids: int, quote: str = '"') -> bytes:
    records = "".join(
        f"\t\t<record from={quote}{i}{quote} to={quote}graphics/pictures/person/{i}/portrait{quote}/>\n"
        for i in ids
    )
    return (
        '<record>\n\t<boolean id="preload" value="false"/>\n\t<list id="maps">\n'
        f"{records}\t</list>\n</record>\n"
    ).encode()
