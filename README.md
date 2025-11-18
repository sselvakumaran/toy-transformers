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
- `<UNK>` should be built into tokenizer
- fix tokenization script name
- tokenization should add checksum to verify correct token dictionary
- tokenization defaults 
- fix "verbose"

# LIST
1. Figure out issue with GPTv2
2. Tokenization script
3. C++ tokenizer (if incredibly slow)
4. GPTv3 with transformer-XL and alibi
5. Deepseek
