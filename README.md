# Decision boundary aware text augmentation (by IIPL Latent-variable NLP Team)

### Dependencies

This code is written in Python. Dependencies include

* Python == 3.6.13
* PyTorch == 1.10.0
* Transformers (Huggingface) == 4.12.0
* Datasets == 2.4.0
* NLG-Eval == 2.3.0 (https://github.com/Maluuba/nlg-eval)

## Preprocessing

Before training the model, it needs to go through a preprocessing step. Preprocessing is performed through the '--preprocessing' option and the pickle file of the set data is saved in the preprocessing path (--preprocessing_path).

```
python main.py --preprocessing
```

## Training

To train the model, add the training (--training) option. Currently, only the Transformer model is available, but RNN and Pre-trained Language Model will be added in the future.

```
python main.py --training
```
