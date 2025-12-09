import sys
import os
import data.bpe as tokenizer
from typing import Optional, NamedTuple
from collections import namedtuple
import json
import torch
import argparse

from toy_transformers.utilities import io
def _td_save(td: tokenizer.TokenDictionary):
	return {
		'vocab_size': len(td.token_set),
	}
def _td_load(obj):
	pass


def write_tokenization(td: tokenizer.TokenDictionary, fn: str) -> bool:
	assert isinstance(td, tokenizer.TokenDictionary), "write_tokenization requires TokenDictionary"

	if not os.path.exists(fn) and os.path.dirname(fn) != '':
		try:
			os.makedirs(os.path.dirname(fn), exist_ok=True)
		except OSError as e:
			print(f"error creating directory for {fn}: {e}")

	try:
		with open(fn, "w+") as file:
			json.dump(td, file, indent=4)
	except (IOError, OSError) as e:
		print(f"error writing to file '{fn}': {e}")
		return False
	except Exception as e:
		print(e)
		return False

	return True

def read_tokenization(fn: str) -> Optional[tokenizer.TokenDictionary]:
	try:
		with open(fn, "r") as file:
			d = json.load(file)
			td = tokenizer.TokenDictionary(*d)
			# json converts keys to strs, must convert back
			new_idx_to_token =dict(map(lambda t: (int(t[0]), t[1]), td.idx_to_token.items()))
			return td._replace(idx_to_token=new_idx_to_token)
	except FileNotFoundError:
		print(f"file not found: {fn}"); return None
	except (IOError, OSError) as e:
		print(f"error reading file '{fn}': {e}"); return None
	except json.decoder.JSONDecodeError as e:
		print(f"error decoding JSON: {e}"); return None
	except Exception as e:
		print(e); return None

def tokenize_from_json(
	in_fn: str, 
	out_fn: str,
	tokenization_json_fn: str
) -> bool:
	# open in file, read, close
	# create tokenization
	# open out file, write, close

	try:
		with open(in_fn, 'r') as file:
			data = file.read()
	except FileNotFoundError:
		print("file not found: {fn}"); return False
	except (IOError, OSError) as e:
		print(f"error reading file '{in_fn}': {e}"); return False
	
	td = read_tokenization(tokenization_json_fn)
	if not td:
		exit(1)
	
	encode = tokenizer.get_encoder(td, verbose=True)
	out = torch.tensor(encode(data), dtype=torch.long)
	try:
		if not os.path.exists(out_fn) and os.path.dirname(out_fn) != '':
			os.makedirs(os.path.dirname(out_fn), exist_ok=True)
		torch.save(out, out_fn)
	except OSError as e:
		print(f"error creating directory for {out_fn}: {e}")
	except Exception as e:
		print(f"error writing tokenized data: {e}"); return False
	
	return True

def tokenize_from_scratch(
	in_fn: str, 
	out_fn: str,
	tokenization_json_fn: str,
	num_tokens: int,
	pattern: Optional[str] = None,
	predefined: Optional[list[str]] = None,
) -> bool:
	try:
		with open(in_fn, 'r') as file:
			data = file.read()
	except FileNotFoundError:
		print("file not found: {fn}"); return False
	except (IOError, OSError) as e:
		print(f"error reading file '{in_fn}': {e}"); return False
	td = tokenizer.create_tokenizer(data, num_tokens, pattern=pattern, predefined=predefined, verbose=True)

	ok = write_tokenization(td, tokenization_json_fn)
	if not ok:
		return False
	
	encode = tokenizer.get_encoder(td, pattern=pattern, verbose=True)
	out = torch.tensor(encode(data), dtype=torch.long)
	try:
		if not os.path.exists(out_fn) and os.path.dirname(out_fn) != '':
			os.makedirs(os.path.dirname(out_fn), exist_ok=True)
		torch.save(out, out_fn)
	except OSError as e:
		print(f"error creating directory for {out_fn}: {e}")
	except Exception as e:
		print(f"error writing tokenized data: {e}"); return False
	
	return True

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
	parser_apply = subparsers.add_parser('apply', help="tokenize using existing encoding")
	parser_apply.add_argument('input_file', type=str, help="[1] path to raw text input file")
	parser_apply.add_argument('output_file', type=str, help="[2] path to save output data tensor")
	parser_apply.add_argument('encoding_file', type=str, help="[3] path to load token encoding JSON")

	parser_train = subparsers.add_parser('train', help='train new encoding')
	parser_train.add_argument('input_file', type=str, help="[1] path to raw text input file")
	parser_train.add_argument('output_file', type=str, help="[2] path to save output data tensor")
	parser_train.add_argument('encoding_file', type=str, help="[3] path to save token encoding JSON")
	parser_train.add_argument('num_tokens', type=int, help="[4] number of total tokens to process")
	parser_train.add_argument('pattern', type=str, nargs='?', default=None, help="[5] (optional) regex pattern to define merge boundaries")
	parser_train.add_argument('predefined', type=str, nargs='*', default=None, help="[6] (optional) list of predefined tokens")

	args = parser.parse_args()

	if args.command == 'apply':
		tokenize_from_json(
			in_fn=args.input_file, 
			out_fn=args.output_file, 
			tokenization_json_fn=args.encoding_file
		)
	elif args.command == 'train':
		tokenize_from_scratch(
			in_fn=args.input_file, 
			out_fn=args.output_file, 
			tokenization_json_fn=args.encoding_file,
			num_tokens=args.num_tokens,
			pattern=args.pattern,
			predefined=args.predefined
		)