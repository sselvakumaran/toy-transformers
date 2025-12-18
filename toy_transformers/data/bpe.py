from typing import List, Optional, Union, ByteString, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

Token = str | bytes

class TokenizationMode(Enum):
	STR = "str"
	BYTES = "bytes"

@dataclass(frozen=True)
class Vocabulary():
	tokens: list[Token]
	pattern: str | None = None
	
	@cached_property
	def token_to_idx(self) -> dict[Token, int]:
		return {t: i for i, t in enumerate(self.tokens)}