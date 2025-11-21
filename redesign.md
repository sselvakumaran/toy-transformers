# Redesign

## Things needed
- `/src` - for most code
  - `/models` - for holding models
    - `base.py` - base abstract model
    - `gpt1.py`
    - `gpt2.py`
    - etc.
  - `/optimizers` - optimizers
    - `adamw.py`
    - `muon.py`
  - `/preprocess` - for preprocess scripts
    - `bpe.py` - module with bpe code
    - `dataset.py` - class for handling pytorch dataset / dataloader
    - `tokenizer.py` - script for applying tokenizer to text, training / loading vocabs
    - `cleaning.py` - basic methods to clean some text datasets
  - `/util` - for utilities
    - `plotting.py` - plotting loss
    - `???.py` - handling tracking of model training / logs / etc. metadata
    - `???.py` - handling reading / writing model and optimizer checkpoints
  - `train.py` - script for training a model / 
- `/notebooks` - for running and verifying models
- `???` - storage for vocabs, tokenized/processed text, metadata, model checkpoints (git ignored)
- `/raw?` - place to keep raw text files (git ignored)


NEW
```python3
src/
	__init__.py
	models/
		base.py
		gpt1.py
		gpt2.py
		...
	optim/
		adamw.py
		muon.py
	data/
		__init__.py
		bpe.py
		dataset.py
		cleaning.py
	utils/
		__init__.py
		plotting.py
		logging.py
		io.py
		config.py
	pipeline/
		preprocess.py (BPE)
		train.py
		inference.py (maybe)
		evaluate.py (maybe)
scripts/
	train.sh
	preprocess.sh
	infer.sh
notebooks/
	test1.ipynb
	...
data/
	raw/
	processed/
runs/
	run1/
		metadata.json
		weights.py
		train/
			train.json
			checkpoints/
			logs/
```
