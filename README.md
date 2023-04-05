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

## Augmenter Training

To train the augmentation model, add the augmenter training (--augmenter_training) option. Currently, only the Transformer-based pre-trained language model is available, but RNN will be added in the future.

```
python main.py --augmenter_training
```
