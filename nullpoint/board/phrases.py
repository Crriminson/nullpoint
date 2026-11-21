"""Phrase pages. Most-used phrases first (they sit nearest the focus card / null point).

A caregiver can replace them from the caregiver page (PIN-protected over Wi-Fi). Custom pages are
stored in data/phrases.json (git-ignored: they can be personal) and loaded at start. PAGES is
updated in place, so every module that imported it sees the new pages.
"""
from __future__ import annotations

import json

from nullpoint.config import DATA

NEXT = "More options"
EMERGENCY_TEXT = "Help! I need help now!"
MAX_PAGES, MAX_PER_PAGE, MAX_LEN = 6, 7, 40
CUSTOM = DATA / "phrases.json"

DEFAULT_PAGES = [
    ("Core", ["Yes", "No", "I need help", "Thank you", "Wait, please", "I'm in pain", "I'm thirsty"]),
    ("Needs", ["Water, please", "I'm hungry", "Bathroom, please", "I'm cold", "I'm hot", "Adjust my position",
               "Turn off the light"]),
    ("Feelings", ["I'm okay", "I'm tired", "I'm happy", "I'm worried", "I feel sick", "I want to rest",
                  "Please stay with me"]),
    ("Talk", ["Hello", "Good morning", "How are you?", "Call my family", "Please repeat that", "I don't understand",
              "Let's talk later"]),
]


def clean(pages) -> list:
    """Validate pages given as [[name, [phrase, ...]], ...]; raises ValueError with a readable reason."""
    if not isinstance(pages, list) or not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError(f"between 1 and {MAX_PAGES} pages")
    out = []
    for i, page in enumerate(pages):
        if not (isinstance(page, (list, tuple)) and len(page) == 2 and isinstance(page[1], list)):
            raise ValueError(f"page {i + 1}: expected a name and a list of phrases")
        name = str(page[0]).strip()[:20] or f"Page {i + 1}"
        phr = [str(p).strip() for p in page[1] if str(p).strip()]
        phr = [p for p in phr if p != NEXT]
        if not 1 <= len(phr) <= MAX_PER_PAGE:
            raise ValueError(f"page '{name}': between 1 and {MAX_PER_PAGE} phrases")
        long = [p for p in phr if len(p) > MAX_LEN]
        if long:
            raise ValueError(f"page '{name}': '{long[0][:20]}…' is longer than {MAX_LEN} characters")
        out.append((name, phr))
    return out


def _with_next(pages):
    return [(name, phr + [NEXT]) for name, phr in pages]


def load() -> list:
    if CUSTOM.exists():
        try:
            return _with_next(clean(json.loads(CUSTOM.read_text(encoding="utf-8"))))
        except (ValueError, json.JSONDecodeError) as e:
            print(f"ignoring {CUSTOM.name}: {e}")
    return _with_next(DEFAULT_PAGES)


def editable() -> list:
    """Current pages without the 'More options' tile, for the editor."""
    return [[name, [p for p in phr if p != NEXT]] for name, phr in PAGES]


def set_pages(pages, save: bool = True) -> list:
    new = clean(pages)
    if save:
        CUSTOM.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM.write_text(json.dumps([[n, p] for n, p in new], indent=1, ensure_ascii=False), encoding="utf-8")
    PAGES[:] = _with_next(new)
    return PAGES


def reset_defaults() -> list:
    CUSTOM.unlink(missing_ok=True)
    PAGES[:] = _with_next(DEFAULT_PAGES)
    return PAGES


PAGES = load()
