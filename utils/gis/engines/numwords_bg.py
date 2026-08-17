"""
Bulgarian number-to-words spelling for legal land-plot descriptions.

Ported from the user's own chsi-app project (github.com/Kamend1/chsi-app,
numwords.py) — that tool generates ЗКИР-standard property descriptions for
Bulgarian judicial executors (ЧСИ) and already has correct, tested
Bulgarian grammar (declensions, "и" insertion, thousands handling) for
exactly this purpose. Reused rather than re-derived to avoid introducing
new grammar bugs in legal-document text.

Two conventions from that project applied to our legal descriptions:
  - `spell_identifier`: a cadastral number written digit-by-digit in words
    (e.g. "68134.905.1462" -> "шест, осем, едно, три, четири, точка, ...")
    immediately after the numeral form, in slashes — standard Bulgarian
    notarial practice for identification numbers in formal documents.
  - `area_words`: an area spelled out in words alongside the numeral
    (e.g. 1625.0 -> "хиляда шестстотин двадесет и пет квадратни метра").
"""
from __future__ import annotations

_EDINICI_M = {
    0: "нула", 1: "един", 2: "два", 3: "три", 4: "четири",
    5: "пет", 6: "шест", 7: "седем", 8: "осем", 9: "девет",
}
_EDINICI_SR = {  # среден род — за "стотни"
    1: "едно", 2: "две", 3: "три", 4: "четири", 5: "пет",
    6: "шест", 7: "седем", 8: "осем", 9: "девет",
}
_EDINICI_F = {  # женски род — за брой "хиляди"
    1: "една", 2: "две", 3: "три", 4: "четири", 5: "пет",
    6: "шест", 7: "седем", 8: "осем", 9: "девет",
}
_NADESET = {
    10: "десет", 11: "единадесет", 12: "дванадесет", 13: "тринадесет",
    14: "четиринадесет", 15: "петнадесет", 16: "шестнадесет",
    17: "седемнадесет", 18: "осемнадесет", 19: "деветнадесет",
}
_DESETICI = {
    20: "двадесет", 30: "тридесет", 40: "четиридесет", 50: "петдесет",
    60: "шестдесет", 70: "седемдесет", 80: "осемдесет", 90: "деветдесет",
}
_STOTICI = {
    100: "сто", 200: "двеста", 300: "триста", 400: "четиристотин",
    500: "петстотин", 600: "шестстотин", 700: "седемстотин",
    800: "осемстотин", 900: "деветстотин",
}
_DIGIT_SPELL = {
    "0": "нула", "1": "едно", "2": "две", "3": "три", "4": "четири",
    "5": "пет", "6": "шест", "7": "седем", "8": "осем", "9": "девет",
}


def _do_999(n: int, edinici=_EDINICI_M) -> list[str]:
    parts: list[str] = []
    h = (n // 100) * 100
    rest = n % 100
    if h:
        parts.append(_STOTICI[h])
    if rest:
        if rest < 10:
            parts.append(edinici[rest])
        elif rest < 20:
            parts.append(_NADESET[rest])
        else:
            tens = (rest // 10) * 10
            ones = rest % 10
            parts.append(_DESETICI[tens])
            if ones:
                parts.append("и")
                parts.append(edinici[ones])
    return parts


def _join_i(parts: list[str]) -> str:
    if len(parts) >= 2 and parts[-2] != "и":
        parts = parts[:-1] + ["и", parts[-1]]
    return " ".join(parts)


def integer_words(n: int, edinici=_EDINICI_M) -> str:
    if n == 0:
        return "нула"
    if n < 0:
        return "минус " + integer_words(-n, edinici)

    thousands = n // 1000
    rest = n % 1000
    parts: list[str] = []

    if thousands:
        if thousands == 1:
            parts.append("хиляда")
        else:
            tw = _do_999(thousands, _EDINICI_F)
            parts.append(_join_i(tw) if len(tw) > 1 else tw[0])
            parts.append("хиляди")

    if rest:
        rw = _do_999(rest, edinici)
        if thousands and len(rw) == 1:
            parts.append("и")
        parts.extend(rw)

    return " ".join(parts)


def _needs_singular(n: int) -> bool:
    return n % 10 == 1 and n % 100 != 11


def area_words(value: float) -> str:
    """1625.0 -> 'хиляда шестстотин двадесет и пет квадратни метра';
    524.75 -> 'петстотин двадесет и четири цяло и седемдесет и пет стотни
    квадратни метра'."""
    whole = int(value)
    cents = round((value - whole) * 100)
    if cents == 100:
        whole += 1
        cents = 0

    whole_words = integer_words(whole, _EDINICI_M)

    if cents == 0:
        unit = "квадратен метър" if _needs_singular(whole) else "квадратни метра"
        return f"{whole_words} {unit}"

    cent_words = integer_words(cents, _EDINICI_SR)
    return f"{whole_words} цяло и {cent_words} стотни квадратни метра"


def spell_identifier(s: str) -> str:
    """Cadastral identifier spelled digit-by-digit, e.g. "68134.905.1462"
    -> "шест, осем, едно, три, четири, точка, девет, нула, пет, точка, ..."."""
    parts = []
    for ch in str(s):
        if ch == ".":
            parts.append("точка")
        elif ch in _DIGIT_SPELL:
            parts.append(_DIGIT_SPELL[ch])
    return ", ".join(parts)


def format_area(value: float) -> str:
    """Numeral form matching the ЗКИР-standard convention: no decimals if
    whole, else 2 decimals."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"
