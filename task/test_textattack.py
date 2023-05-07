# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
# Import PyTorch
import torch
# Import HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import TextAttack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import AttackRecipe
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019, PWWSRen2019, CLARE2020
from textattack.datasets import HuggingFaceDataset
# Import Custom Modules
from utils import TqdmLoggingHandler, write_log
from task.utils import data_load

def test_textattack(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start textattack testing!')

    # Load data - even though we do not need to load data for textattack,
    # we need to load data to get the number of labels
    _, total_trg_list = data_load(args)
    num_labels = len(set(total_trg_list['train']))

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)

    # Load fine-tuned model from checkpoint
    checkpoint = torch.load(os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, f'cls_training_checkpoint_seed_{args.random_seed}.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    del checkpoint

    model.to(device)
    model.eval()
    write_log(logger, 'Loaded fine-tuned model')

    # Wrap model with textattack
    wrapped_model = HuggingFaceModelWrapper(model, tokenizer)

    # Define attack recipe
    attack_list = [
        TextFoolerJin2019.build(wrapped_model),
        BAEGarg2019.build(wrapped_model),
        PWWSRen2019.build(wrapped_model),
    ]
    # Define attack dataset
    if args.data_name == 'IMDB':
        attack_dataset = HuggingFaceDataset('imdb', split='test')
    elif args.data_name == 'sst2':
        attack_dataset = HuggingFaceDataset('glue', 'sst2', split='test')
    elif args.data_name == 'cola':
        attack_dataset = HuggingFaceDataset('glue', 'cola', split='test')
    elif args.data_name == 'mnli':
        attack_dataset = HuggingFaceDataset('glue', 'mnli', split='test')
    elif args.data_name == 'mrpc':
        attack_dataset = HuggingFaceDataset('glue', 'mrpc', split='test')

    # Define attacker
    log_path = os.path.join(args.result_path, args.data_name)
    attacker_list = [
        Attacker(attack_list[0], attack_dataset, AttackArgs(num_examples=100,
                                                            log_summary_to_json=os.path.join(log_path, 'result_textfooler.json'),
                                                            disable_stdout=True)),
        Attacker(attack_list[1], attack_dataset, AttackArgs(num_examples=100,
                                                            log_summary_to_json=os.path.join(log_path, 'result_bae.json'),
                                                            disable_stdout=True)),
        Attacker(attack_list[2], attack_dataset, AttackArgs(num_examples=100,
                                                            log_summary_to_json=os.path.join(log_path, 'result_pwws.json'),
                                                            disable_stdout=True)),
    ]

    write_log(logger, 'Start textattack!')
    # Attack!
    for each_attacker in attacker_list:
        attack_result = each_attacker.attack_dataset()
