"""Fast, deterministic test of the plate normalizer + validator (no OCR)."""
from modules.anpr_video import _clean_plate_text, _is_plausible_plate

cases = [
    ("HR26AB1234", True),
    ("HR 26 AB 1234", True),
    ("DL01AB1234", True),
    ("UP32AB1234", True),
    ("PB10AB1234", True),
    ("HR2GAB12B4", True),   # common OCR confusion G->6, B->8
    ("hr26ab1234", True),   # lowercase -> uppercase
    ("HR26AB123", True),    # generic 7-char mix
    ("IIIII", False),       # repeated char
    ("0000", False),        # pure digits, too short
    ("", False),
    ("ABC", False),         # too short
]

all_ok = True
for raw, expected in cases:
    norm = _clean_plate_text(raw)
    got = _is_plausible_plate(norm)
    ok = got == expected
    all_ok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {raw!r:20} -> {norm!r:14} plausible={got} (expected {expected})")

print("\nRESULT:", "PASS" if all_ok else "FAIL")
raise SystemExit(0 if all_ok else 2)
