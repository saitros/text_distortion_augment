from transformers import PretrainedConfig, AutoModel, AutoTokenizer

def model_setting(model_name, isPreTrain):
    model_config = PretrainedConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    encoder = basemodel.encoder
    decoder = basemodel.decoder

    return encoder, decoder, model_config