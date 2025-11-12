import sys
import re
from functools import reduce
import unicodedata

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

def gutenberg_cleaning(s: str) -> str:
	funcs = [
		filter_unprintable,
		strip,
		collapse_paragraphs,
		reduce_extra_newlines,
		reduce_whitespace
	]
	return reduce(lambda x, f: f(x), funcs, s)

# wikitext-103 cleaning

def normalize_unicode(s: str) -> str:
	return unicodedata.normalize("NFC", s)

_TOKEN_REPLACEMENTS = {
    r'@-@': '-',
    r'@\.@': '.',
    r"@'@": "'",
    r'@,@': ',',
    r'@;@': ';',
    r'@:@': ':',
    r'@\(@': '(',
    r'@\)@': ')',
    r'@\[@': '[',
    r'@\]@': ']',
    r'@/@': '/',
    r'@\\@': '\\',
    r'@\?@': '?',
    r'@!@': '!',
    r'@%@': '%',
    r'@\*@': '*',
    # add more if you see other wired tokens
}

def remove_wiki_headings(s: str) -> str:
	return re.sub(r'^\s*={1,6}\s.*?={1,6}\s*$', '', s, flags=re.MULTILINE)

def fix_token_artifacts(s: str) -> str:
	s = re.sub(r'\s*@-@\s*', '-', s)
	s = re.sub(r'\s*@\*@\s*', '*', s)
	s = re.sub(r'\s*@,@\s*', ',', s)
	s = re.sub(r'\s*@;@\s*', ';', s)
	s = re.sub(r'\s*@\(@\s*', '(', s)
	s = re.sub(r'\s*@\)@\s*', ')', s)
	s = re.sub(r'\s*@\[@\s*', '[', s)
	s = re.sub(r'\s*@\]@\s*', ']', s)
	s = re.sub(r'\s*@\'@', "'", s)
	s = re.sub(r'\s*@\.@\s*', '.', s)
	return s

def fix_spacing(s: str) -> str:
    # Fix spacing around punctuation
    s = re.sub(r'\s+([,.;:!?])', r'\1', s)
    s = re.sub(r'([(\["])\s+', r'\1', s)
    s = re.sub(r'\s+([\]\")])', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s)
    s = re.sub(r'(\w)"', r'\1 "', s)
    s = re.sub(r'"(\w)', r'" \1', s)
    s = re.sub(r'\\\'', "'", s)
    return s


def split_subjects(s: str) -> str:
	s = re.sub(r'\n\n([A-Z][^\n]+)', r'\n\n\1\n', s)
	return s

def basic_cleaning_wikitext(s: str) -> str:
	funcs = [
		filter_unprintable,
		remove_wiki_headings,
		fix_token_artifacts,
		fix_spacing,
		strip,
		collapse_paragraphs,
		reduce_extra_newlines,
		split_subjects,
		reduce_whitespace,
	]
	return reduce(lambda x, f: f(x), funcs, s)