# TODO
## Model Architectures
- GPTv2 (flash attention)
- Deepseek (multi-headed latent attention)
- gated linear attention
- recurrent transformers
- GPT v3 (transformer-XL caching memory, alibi)
- perceiver
## Model Additions
- finetuning (LoRA / prefix tuning)
- save/load model values
- checkpointing
## Utilities
- C++ tokenizer
- tokenization script
  - get data and info, dump processed data + token info into files to use later
## Repo Cleaning / Usability
- clean up having data in repo (not needed)
- easier interfaces to process text
- make easier model switchout / dropin
- ~~`<UNK>` should be built into tokenizer~~ (could be cleaner)
- fix tokenization script name
- tokenization should add checksum to verify correct token dictionary
- add used pattern to stored token dictionary (since changes encoder)
- tokenization argument defaults + file location defaults
- fix "verbose"

# LIST
1. Tokenization script
2. C++ tokenizer (if incredibly slow)
3. GPTv3 with transformer-XL and alibi
4. Deepseek
