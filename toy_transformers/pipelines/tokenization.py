from dataclasses import dataclass
from io import BufferedIOBase, TextIOBase
from toy_transformers.data import bpe as tokenizer
from typing import Optional, Union
import torch
import argparse

from toy_transformers.utilities import io
from toy_transformers.utilities.io import TorchTensorRef
from toy_transformers.utilities.version import get_obj_metadata

class TokenizedData:
	vocab: Optional[tokenizer.Vocabulary]
	data: torch.Tensor

	def __init__(self,
		raw_data_handle: Optional[Union[TextIOBase, BufferedIOBase]] = None,
		vocab: Optional[tokenizer.Vocabulary] = None,
		data_tensor: Optional[torch.Tensor] = None,
	):
		assert (raw_data_handle is None) ^ (data_tensor is None), \
			"either raw_data_handle or data_tensor must be non-None"
		
		self.vocab = vocab
		if raw_data_handle is not None:
			self.data = torch.tensor(
				vocab.encode(raw_data_handle.read())
			)
		else:
			self.data = data_tensor

	def to_state_dict(self):
		assert self.vocab is not None, "vocab must be stored to save"
		return {
			"metadata": get_obj_metadata(self, 
				include_timestamp=False,
				include_hash = False
			),
			"mode": self.vocab.config.mode.value,
			"vocab_hash": hash(self.vocab),
			"data": TorchTensorRef("data", self.data)
		}
	
	@classmethod
	def from_state_dict(cls, obj,
		vocab: Optional[tokenizer.Vocabulary] = None
	):
		if vocab:
			assert hash(vocab) == obj["vocab_hash"], "vocab hash does not match"

		return cls(
			vocab=vocab,
			data_tensor=obj["data"]
		)

## EXAMPLE NEW FUNCTIONS

def tokenize_from_scratch(
	in_path: str,
	vocab_path: str,
	out_path: str,
	vocab_size: int,
	mode: Optional[tokenizer.TokenizationMode] = tokenizer.TokenizationMode.STR,
	**kwargs
):
	file_mode = 'rb' if mode == tokenizer.TokenizationMode.BYTES else 'r'

	with open(in_path, file_mode) as data_handle:
		vocab = tokenizer.create_bpe(
			data_handle,
			vocab_size,
			**kwargs
		)

	io.save(vocab.to_state_dict(), vocab_path)

	with open(in_path, file_mode) as data_handle:
		tokenized_data = TokenizedData(
			raw_data_handle=data_handle,
			vocab=vocab
		)

	io.save(tokenized_data.to_state_dict(), out_path)

def tokenize_from_vocab(
	in_path: str,
	vocab_path: str,
	out_path: str,
):
	vocab_data = io.load(vocab_path)
	vocab = tokenizer.Vocabulary.from_state_dict(vocab_data)

	file_mode = 'rb' if vocab.config.mode == tokenizer.TokenizationMode.BYTES else 'r'

	with open(in_path, file_mode) as data_handle:
		tokenized_data = TokenizedData(
			raw_data_handle=data_handle,
			vocab=vocab
		)

	io.save(tokenized_data.to_state_dict(), out_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="utility for applying / training tokenizers",
		formatter_class=argparse.RawTextHelpFormatter
	)
	subparsers = parser.add_subparsers(
		title='subcommands',
		dest='command',
		required=True,
		help="'train' for training new encoding, 'apply' for applying existing encoding"
	)
	parser_apply = subparsers.add_parser('apply', help="tokenize using existing vocabulary")
	parser_apply.add_argument('input_file', type=str, help="[1] path to raw input file")
	parser_apply.add_argument('output_file', type=str, help="[2] path to save tokenized data")
	parser_apply.add_argument('vocab_file', type=str, help="[3] path to vocabulary directory")

	parser_train = subparsers.add_parser('train', help='train new vocabulary and tokenize')
	parser_train.add_argument('input_file', type=str, help="[1] path to raw input file")
	parser_train.add_argument('output_file', type=str, help="[2] path to save tokenized data")
	parser_train.add_argument('vocab_file', type=str, help="[3] path to save vocabulary")
	parser_train.add_argument('vocab_size', type=int, help="[4] vocabulary size")
	parser_train.add_argument('--mode', type=str, choices=['str', 'bytes'], default='str', help="tokenization mode (default: str)")
	parser_train.add_argument('--pattern', type=str, default=None, help="regex pattern for splitting boundaries")
	parser_train.add_argument('--special-tokens', type=str, nargs='*', default=None, help="list of special tokens")
	parser_train.add_argument('--verbose', action='store_true', help="enable verbose output")
	parser_train.add_argument('--chunk-size', type=int, default=None, help="read chunk size for training")

	args = parser.parse_args()

	if args.command == 'apply':
		tokenize_from_vocab(
			in_path=args.input_file,
			vocab_path=args.vocab_file,
			out_path=args.output_file
		)
	elif args.command == 'train':
		tokenize_from_scratch(
			in_path=args.input_file,
			vocab_path=args.vocab_file,
			out_path=args.output_file,
			mode=args.mode,
			vocab_size=args.vocab_size,
			verbose=args.verbose,
			pattern=args.pattern,
			special_tokens=args.special_tokens,
			read_chunk_size=args.chunk_size
		)