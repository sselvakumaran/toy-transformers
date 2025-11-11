import sys
import re

NOPRINT_TRANSLATION_TABLE = {
    i: None for i in range(sys.maxunicode + 1) if not chr(i).isprintable()
}

def filter_unprintable(s: str) -> str:
    return s.translate(NOPRINT_TRANSLATION_TABLE)

def remove_extra_newlines(s: str) -> str:
    return re.sub(r'\n{2,}', '\n\n', s)