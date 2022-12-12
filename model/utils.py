def return_model_name(model_type):
    if model_type == 'bert':
        out = 'bert-base-cased'
    if model_type == 'albert':
        out = 'textattack/albert-base-v2-imdb'
    if model_type == 'deberta':
        out = 'microsoft/deberta-v3-base'
    return out