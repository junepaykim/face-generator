"""The bytes of facegen-app.exe's resources: version information, icons and the launcher
without its signature (desktop/app_exe.py)."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

import pytest
from support.pe import IMAGE_SIZE, SECURITY_AT, synthetic_pe

from facegen.desktop import app_exe
from facegen.desktop.app_exe import AppExeError


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("0.1.2", (0, 1, 2, 0)),
        ("1.2.3.4.5", (1, 2, 3, 4)),
        ("0.2.0.dev1", (0, 2, 0, 0)),
        ("70000.1", (0xFFFF, 1, 0, 0)),
        ("", (0, 0, 0, 0)),
        ("unknown", (0, 0, 0, 0)),
    ],
)
def test_the_version_as_four_numbers(text: str, numbers: tuple[int, ...]) -> None:
    assert app_exe.numeric_version(text) == numbers


@dataclass
class Node:
    key: str
    wtype: int
    value: bytes
    length: int
    children: list[Node] = field(default_factory=list)

    def child(self, key: str) -> Node:
        return next(c for c in self.children if c.key == key)


def _aligned(at: int) -> int:
    return at + (-at % 4)


def parse(data: bytes, at: int = 0) -> Node:
    """A VS_VERSIONINFO node as Windows reads it."""
    length, value_length, wtype = struct.unpack_from("<HHH", data, at)
    end = at + 6
    while data[end : end + 2] != b"\0\0":
        end += 2
    key = data[at + 6 : end].decode("utf-16-le")
    pos = _aligned(end + 2)
    size = value_length * 2 if wtype == 1 else value_length
    node = Node(key, wtype, data[pos : pos + size], length)
    pos = _aligned(pos + size)
    while pos < at + length:
        child = parse(data, pos)
        node.children.append(child)
        pos = _aligned(pos + child.length)
    return node


def test_the_version_information_names_kitshot() -> None:
    data = app_exe.version_info("0.1.2")
    root = parse(data)
    assert (root.key, root.length, root.wtype) == ("VS_VERSION_INFO", len(data), 0)
    fixed = struct.unpack("<13I", root.value)
    assert fixed[0] == 0xFEEF04BD
    assert fixed[2:6] == (1, 2 << 16, 1, 2 << 16)  # 0.1 / 2.0, file and product
    table = root.child("StringFileInfo").child("040904B0")
    strings = {s.key: s.value.decode("utf-16-le").rstrip("\0") for s in table.children}
    assert strings["FileDescription"] == strings["ProductName"] == "Kitshot"
    assert strings["CompanyName"] == "Kitshot"
    assert strings["FileVersion"] == strings["ProductVersion"] == "0.1.2"
    assert strings["OriginalFilename"] == "facegen-app.exe"
    assert "Python Software Foundation" in strings["LegalCopyright"]
    translation = root.child("VarFileInfo").child("Translation")
    assert translation.value == struct.pack("<HH", 0x0409, 1200)


def test_an_icon_file_becomes_pictures_and_their_directory() -> None:
    from facegen.desktop.app_icon import icon_image

    buffer = io.BytesIO()
    icon_image(None, 256).save(buffer, format="ICO", sizes=[(16, 16), (32, 32), (256, 256)])
    pictures, group = app_exe.icon_resources(buffer.getvalue())
    assert struct.unpack_from("<HHH", group) == (0, 1, 3) and len(group) == 6 + 14 * 3
    entries = [struct.unpack_from("<BBBBHHIH", group, 6 + 14 * i) for i in range(3)]
    assert [(e[0], e[1]) for e in entries] == [(16, 16), (32, 32), (0, 0)]  # 0 is 256
    assert [e[7] for e in entries] == [1, 2, 3]
    assert [e[6] for e in entries] == [len(p) for p in pictures]
    assert all(e[4] == 1 and e[5] == 32 for e in entries)


@pytest.mark.parametrize("data", [b"", b"nope", struct.pack("<HHH", 0, 1, 2) + bytes(20)])
def test_what_is_not_an_icon_is_refused(data: bytes) -> None:
    with pytest.raises(AppExeError):
        app_exe.icon_resources(data)


def test_the_signature_is_dropped_and_other_appended_data_kept() -> None:
    signed = synthetic_pe(extra=b"appended", cert=b"S" * 16)
    image, tail = app_exe.split_signature(signed)
    assert len(image) == IMAGE_SIZE and tail == b"appended"
    assert image[SECURITY_AT : SECURITY_AT + 8] == bytes(8)
    assert image[:SECURITY_AT] == signed[:SECURITY_AT]

    image, tail = app_exe.split_signature(synthetic_pe(cert=b"S" * 16))
    assert len(image) == IMAGE_SIZE and tail == b""
    plain = synthetic_pe()
    assert app_exe.split_signature(plain) == (plain, b"")
    assert app_exe.split_signature(plain + bytes(7)) == (plain, b"")  # padding only


@pytest.mark.parametrize(
    "data", [b"", b"ZM" + bytes(100), b"MZ" + bytes(100), synthetic_pe()[: IMAGE_SIZE - 10]]
)
def test_what_is_not_a_whole_program_is_refused(data: bytes) -> None:
    with pytest.raises(AppExeError):
        app_exe.split_signature(data)


def test_the_stamp_changes_with_each_input() -> None:
    key = app_exe.stamp_key(b"launcher", b"icon", "0.1.2")
    assert key == app_exe.stamp_key(b"launcher", b"icon", "0.1.2")
    assert key != app_exe.stamp_key(b"launcher2", b"icon", "0.1.2")
    assert key != app_exe.stamp_key(b"launcher", b"icon2", "0.1.2")
    assert key != app_exe.stamp_key(b"launcher", b"icon", "0.1.3")


def test_a_new_product_name_makes_the_copy_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stamp holds the version information, so the rename to Kitshot (Phase 17c) remade
    copies that said facegen."""
    key = app_exe.stamp_key(b"launcher", b"icon", "0.1.2")
    monkeypatch.setattr(app_exe, "NAME", "facegen")
    assert app_exe.stamp_key(b"launcher", b"icon", "0.1.2") != key
