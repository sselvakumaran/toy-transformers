import sys
import re
from functools import reduce

NOPRINT_IGNORE = set(["\n"])
NOPRINT_TRANSLATION_TABLE = {
	i: None for i in range(sys.maxunicode + 1) 
	if not chr(i).isprintable() and chr(i) not in NOPRINT_IGNORE
}

def filter_unprintable(s: str) -> str:
	return s.translate(NOPRINT_TRANSLATION_TABLE)

def strip(s: str) -> str:
	return re.sub(r'^[ \t]+|[ \t]+$', '', s, flags=re.MULTILINE)

def reduce_extra_newlines(s: str) -> str:
	return re.sub(r'\n{2,}', '\n', s)

def collapse_paragraphs(s: str) -> str:
	return re.sub(r'(?<!\n)\n(?!\n)', ' ', s)

def reduce_whitespace(s: str) -> str:
	return re.sub(r' {2,}', ' ', s)

def basic_cleaning(s: str) -> str:
	funcs = [
		filter_unprintable,
		strip,
		collapse_paragraphs,
		reduce_extra_newlines,
		reduce_whitespace
	]
	return reduce(lambda x, f: f(x), funcs, s)