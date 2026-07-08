"""Speech-only text normalization.

Local TTS engines (Kokoro, Piper) read `$20,000` back digit-by-digit
("dollar two zero zero zero zero") because their espeak-based phonemizers
don't expand currency. This expands currency amounts to words (e.g.
"twenty thousand dollars") *only for the audio sent to synthesize()* —
`scene.narration` itself is left untouched since captions display the
numeral form, which is what you want on screen.
"""

from __future__ import annotations

import re

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]
_SCALES = ["", "thousand", "million", "billion", "trillion"]

_CURRENCY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*|\d+)(\.\d{1,2})?")


def _below_thousand_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, remainder = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[remainder] if remainder else "")
    hundreds, remainder = divmod(n, 100)
    words = _ONES[hundreds] + " hundred"
    if remainder:
        words += " " + _below_thousand_to_words(remainder)
    return words


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    groups = []
    while n > 0:
        n, group = divmod(n, 1000)
        groups.append(group)
    parts = []
    for index in reversed(range(len(groups))):
        group = groups[index]
        if group == 0:
            continue
        words = _below_thousand_to_words(group)
        if _SCALES[index]:
            words += " " + _SCALES[index]
        parts.append(words)
    return " ".join(parts)


def _currency_replacement(match: re.Match[str]) -> str:
    dollars = int(match.group(1).replace(",", ""))
    cents_str = match.group(2)
    words = _int_to_words(dollars) + (" dollar" if dollars == 1 else " dollars")
    if cents_str:
        cents = int(cents_str[1:].ljust(2, "0"))
        if cents:
            words += " and " + _int_to_words(cents) + (
                " cent" if cents == 1 else " cents"
            )
    return words


def normalize_for_speech(text: str) -> str:
    """Expand `$`-prefixed currency amounts into spoken words."""
    return _CURRENCY_RE.sub(_currency_replacement, text)
