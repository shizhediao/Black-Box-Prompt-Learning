import argparse
import logging
import torch
import math
import os
import random
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from torch.optim import Adam, AdamW
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM 
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

LABEL2ID_CONFIG = {
    "mnli": {" no": 0, " maybe": 1, " yes": 2},
    "qqp": {" no": 0, " yes": 1},
    "sst2": {" terrible": 0, " great": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": {" no": 0, " yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
    "CI": {' background': 0, ' comparison': 1, ' extension': 2, ' future': 3, ' motivation': 4, ' use': 5},
    "SE": {' comparison': 0, ' conjunction': 1, ' evaluation': 2, ' feature': 3, ' hyponym': 4, ' part': 5, ' function': 6},
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4} ,
    "HP": {' unhelpful': 0, ' helpful': 1}, # review helpfulness
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
    "mr": {" terrible": 0, " great": 1},
    "mpqa": {" terrible": 0, " great": 1}
}

LABEL_CONVERT = {
    "mnli": {0: ' no', 1: ' maybe', 2: ' yes'},
    "qqp": {0: ' no', 1: ' yes'},
    "sst2": {0: ' terrible', 1: ' great'},
    'mrpc': {0: ' no', 1: ' yes'},
    'cola': {0: ' no', 1: ' yes'},
    'wnli': {0:  ' no', 1: ' yes'},
    'qnli': {0: ' yes', 1: ' no'},
    'rte': {0: ' yes', 1: ' no'},
    'CI': {'Background': ' background', 'CompareOrContrast': ' comparison', 'Extends': ' extension', 'Future': ' future', 'Motivation': ' motivation', 'Uses': ' use'},
    'SE': {'COMPARE': ' comparison', 'CONJUNCTION': ' conjunction', 'EVALUATE-FOR': ' evaluation', 'FEATURE-OF': ' feature', 'HYPONYM-OF': ' hyponym', 'PART-OF': ' part', 'USED-FOR': ' function'},
    'RCT': {'BACKGROUND': ' background', 'CONCLUSIONS': ' conclusion', 'METHODS': ' method', 'OBJECTIVE': ' objective', 'RESULTS': ' result'},
    'HP': {False: ' unhelpful', True: ' helpful'},
}

TEMPLATE_CONFIG = {
    "mnli": " entailment? [MASK].",
    "qqp": "? [MASK],",
    "sst2": " It was [MASK].",
    "mrpc": "? [MASK],",
    "cola": " correct? [MASK].",
    "wnli": " entailment? [MASK].",
    "qnli": " entailment? [MASK].",
    "rte": " entailment? [MASK].",
    "CI": " What is the intent? [MASK].", 
    "SE": " What is the relation? [MASK].",
    "RCT": " It is [MASK]. ",
    "HP": " It is [MASK].",
    "imdb": "It was [MASK].",
    "cr": "It was [MASK].",
}

def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = 0, 0

    b = prompt_emb.max()
    def f(v):
        s = (prompt_emb - v).clamp(0, 1).sum()
        return s - k
    itr = 0

    v = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr

def constrainScoreByWholeExact(prompt_embeds):
    for i in range(len(prompt_embeds)):
        v, itr = solve_v_total_exact(prompt_embeds[i])
        prompt_embeds[i].sub_(v).clamp_(0, 1)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--ce_loss", type=bool, default=True)
    parser.add_argument("--sample_size", type=int, default=20, help="IMPORTANT, sample size per batch")
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--prompt_search_space", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Whether to run wandb.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=450, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--model_name_or_path", type=str, default='roberta-large', help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--k_shot", default=-1, type=int, help="-1 denotes full-shot")
    parser.add_argument("--use_ngram", default=True, type=bool, help="If True, will extract ngrams and use them.")
    parser.add_argument("--api_limit", type=int, default=8000 , help="The limit of the API request")
    args = parser.parse_args()

    args.train_file = './dataset/' + args.file_name + '/train.csv' if args.file_name else None
    args.validation_file = './dataset/' + args.file_name + '/dev.csv' if args.file_name else None
    args.test_file = './dataset/' + args.file_name + '/test.csv' if args.file_name else None

    sanity = not (args.task_name and args.file_name)
    assert sanity

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    return args

def pmi():
    args = parse_args()
    result=[]
    if args.file_name:
        with open("./pmi/" + args.file_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))
    elif args.task_name:
        with open("./pmi/" + args.task_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))

    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(map(int, unique))
    return ngram_index_list

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        if wrapper.count % 100 == 0:
            print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
        return res
    wrapper.count = 0
    return wrapper

class ApiCallLimitError(Exception):
    pass

ngram_list = pmi()

def main():
    args = parse_args()
    assert args.task_name != 'stsb'
    ce_loss_string = 'True' if args.ce_loss else 'False'

    # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
    task_name = args.task_name if args.task_name else args.train_file
    args.unique_task_name = task_name.replace("/", ".")
    args.experiment_id = task_name + str(args.prompt_length) + str(args.prompt_learning_rate) \
                         + str(args.num_train_epochs) + str(args.seed) + str(args.prompt_search_space) + ce_loss_string #'dataset/CI/train.csv1020.0013042160.01falseFALSE'

    if args.use_wandb:
        args.group_name = "RoBERTa_BDPL_" + task_name
        wandb.init(config=args, project="blackbox_prompt", group=args.group_name)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # download the dataset.
    if args.task_name is not None:
        if args.task_name in task_to_keys.keys():
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raise(NotImplementedError)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if args.task_name:
        label_to_id = LABEL2ID_CONFIG[args.task_name]
    elif args.file_name:
        label_to_id = LABEL2ID_CONFIG[args.file_name]

    num_labels = len(label_to_id)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    # init model
    model = RobertaForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    
    args.device = torch.device("cuda", args.cuda)
    model.to(args.device)

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    @counter
    def train_api_request(input_ids=None, attention_mask=None):
        sequence_output = model(input_ids=input_ids, attention_mask=attention_mask)
        return sequence_output

    prompt_length = args.prompt_length
    hingeloss = MarginLoss(margin=args.margin, target=False)
    ce_loss = CrossEntropyLoss()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        if args.low_resource:
            train_random_samples = random.sample(range(0, len(examples["label"])), len(examples["label"])//10)
            for key in examples.keys():
                examples[key] = [examples[key][k] for k in train_random_samples]

        if args.file_name == 'HP':
            for k in range(len(examples["text_a"])):
                if examples["text_a"][k] == None:
                    examples["text_a"].remove(examples["text_a"][k])
                    examples["label"].remove(examples["label"][k])
                    break

        if args.task_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.task_name]
        elif args.file_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.file_name]
        template_base = template_cfg.replace('[MASK]', tokenizer.mask_token)

        if sentence2_key:
            sent1_list = []
            for sent1 in examples[sentence1_key]:
                sent1_list.append(sent1 + template_base)
            texts = (sent1_list, examples[sentence2_key])
        else:
            template = [template_base] * len(examples[sentence1_key])
            texts = (examples[sentence1_key], template)
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, add_special_tokens=False)

        texts = []
        template = [template_base] * len(examples[sentence1_key])
        if sentence2_key:
            for tuple_ in list(zip(examples[sentence1_key], template, examples[sentence2_key])):
                sent_1 = tokenizer.tokenize(tuple_[0])[:200]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                sent_2 = tokenizer.tokenize(tuple_[2])[:200]
                new_sent_2 = tokenizer.convert_tokens_to_string(sent_2)
                texts.append(new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
        else:
            for tuple_ in list(zip(examples[sentence1_key], template)):
                sent_1 = tokenizer.tokenize(tuple_[0])[:400]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                texts.append(new_sent_1 + tuple_[1])
        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

        if args.task_name:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.task_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)

        elif args.file_name in DOMAIN_DATASET:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.file_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)
        else:
            target_encodings = tokenizer.batch_encode_plus(examples["label"], add_special_tokens=False)
            result["labels"]= torch.tensor(target_encodings['input_ids']).squeeze(dim=1).to(args.device)
            
        return result

    def preprocess_function_k_shot(examples):
        random_indices = list(range(0, len(examples["label"])))
        random.shuffle(random_indices)

        new_examples = {}
        for key in examples.keys():
            new_examples[key] = []
        label_count = {}

        for index in random_indices:
            label = examples['label'][index]
            if label not in label_count:
                label_count[label] = 0

            if label_count[label] < args.k_shot:
                for key in examples.keys():
                    new_examples[key].append(examples[key][index])
                label_count[label] += 1
        
        print("Finish few-shot sampling!")

        result = preprocess_function(new_examples)
        return result

    with accelerator.main_process_first():
        if args.k_shot >= 0:
            # k-shot learning
            raw_train_dataset_split = raw_datasets["train"].train_test_split(test_size=0.5)
            raw_train_dataset = raw_train_dataset_split['train']
            raw_eval_dataset = raw_train_dataset_split['test']
            train_dataset = raw_train_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=100000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = raw_eval_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=100000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            if args.task_name == 'mnli':
                test_dataset = raw_datasets["validation_matched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                test_dataset_mm = raw_datasets["validation_mismatched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
            else:
                test_dataset = raw_datasets["validation"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
        else:
            train_dataset = raw_datasets["train"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = raw_datasets["validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            test_dataset = raw_datasets["test" if args.file_name != None else "validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        print("length of train data",len(train_dataset))
        print("length of eval data",len(eval_dataset))
        print("length of test data",len(test_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.task_name == 'mnli':
        test_dataloader_mm = DataLoader(test_dataset_mm, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader_mm = accelerator.prepare(test_dataloader_mm)
    else:
        test_dataloader_mm = None
    model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader, test_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be shorter in multiprocess)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * (num_update_steps_per_epoch)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name, experiment_id=args.experiment_id)
    elif args.file_name in DOMAIN_DATASET:
        metric = load_metric('f1', args.experiment_id)
    else:
        metric = load_metric('accuracy', args.experiment_id)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval_result = 0
    best_prompts_probs = None
    best_epoch = 0
    eval_results = []
    test_results = []    

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Black-box Training
    prompt_search_space = args.prompt_search_space
    prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
    prompts_probs.requires_grad = True
    
    prompt_optimizer = AdamW([{
        "params": [prompts_probs],
        "weight_decay": args.weight_decay,
    }], lr=args.prompt_learning_rate)

    print("----Optimizing black-box prompts----")
    for epoch in range(args.num_train_epochs):
        try:
            for step, batch in enumerate(train_dataloader):
                prompts_dist = torch.distributions.Categorical(prompts_probs)
                with torch.no_grad():
                    if args.trial and step >= 100:
                        break
                    bsz = len(batch['input_ids'])
                    label = batch["labels"].to(args.device)
                    loss_list = []
                    prompts_discrete_indices_list = []
                    for k in range(args.sample_size):
                        prompts_discrete_indices = prompts_dist.sample()
                        prompts_discrete_indices_list.append(prompts_discrete_indices)
                        if args.use_ngram:
                            prompts_discrete_indices_ngram_list = []
                            indices_list = prompts_discrete_indices.int().tolist()
                            for idx in indices_list:
                                prompts_discrete_indices_ngram_list.append(ngram_list[idx])
                            prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
                            cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                        else: 
                            cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)

                        cur_attention_mask = torch.cat([torch.ones(bsz, 1).to(args.device), torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"][:, 1:]],dim=1)
                        mask_pos = np.where(np.array(cur_input_ids.cpu()) == tokenizer.mask_token_id) 
                        mask_pos = torch.tensor(mask_pos[-1])
                        sequence_output = train_api_request(input_ids=cur_input_ids, attention_mask=cur_attention_mask)
                        last_hidden_state = sequence_output[0].squeeze()
                        logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                        label_keys = list(label_to_id.keys())
                        label_map = {}
                        for target in label_keys:
                            label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
                        
                        converted_target = label.clone()
                        for key, val in label_map.items():
                            converted_target[label == key] = val
                        interest_index = list(label_map.keys())
                        logits = logits[:, interest_index]
                        pred = logits.argmax(dim=-1)

                        if args.ce_loss:
                            loss = ce_loss(logits.view(-1, config.num_labels), converted_target)
                        else:
                            loss = hingeloss(logits, converted_target)
                        loss_list.append(loss.item())

                        if train_api_request.count >= args.api_limit:
                            raise ApiCallLimitError()

                    loss_avg = sum(loss_list) / args.sample_size
                    
                    prompt_optimizer.zero_grad()

                    derivative = (-1 / prompts_probs).repeat(args.sample_size, 1, 1)
                    for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                        for i in range(prompt_length):
                            derivative[k][i][prompts_discrete_indices[i]] *= -1

                    prompts_probs.grad = torch.zeros_like(prompts_probs)
                    for k in range(args.sample_size):
                        prompts_probs.grad += 1 / (args.sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]
                    
                    torch.nn.utils.clip_grad_norm_(prompts_probs, 3)
                    prompt_optimizer.step()
                    constrainScoreByWholeExact(prompts_probs)
                    
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        progress_bar.update(1)
                        completed_steps += 1
                    if completed_steps >= args.max_train_steps:
                        break
        except ApiCallLimitError:
            pass

        eval_result = evaluate(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, prompts_probs=prompts_probs, prompt_length=prompt_length, tokenizer=tokenizer)

        if eval_result >= best_eval_result:
            best_eval_result = eval_result
            best_prompts_probs = prompts_probs

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
            
        if train_api_request.count >= args.api_limit:
            break
    
    test(args, model, test_dataloader, metric, accelerator, epoch, test_results, prompts_probs=best_prompts_probs, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)

def evaluate(args,  model, eval_dataloader, metric, ce_loss,config, accelerator, epoch, results, prompts_probs=None, prompt_length=None,tokenizer=None):
    prompts_discrete_indices = prompts_probs.argmax(1)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

    for step, batch in enumerate(eval_dataloader):
        if args.trial and step >= 100:
            break
        bsz = len(batch['input_ids'])

        if args.use_ngram:
            batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
        else:
            batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
        batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

        mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
        mask_pos = torch.tensor(mask_pos[-1])
        label_to_id = model.config.label2id 

        sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
        last_hidden_state = sequence_output[0].squeeze()
        logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

        label = batch["labels"].to(args.device)
        label_keys = list(label_to_id.keys())
        label_map = {}
        for target in label_keys:
            label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
        converted_target = label.clone()
        for key, val in label_map.items():
            converted_target[label == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        eval_loss_c = ce_loss(logits.view(-1, config.num_labels), converted_target)
        predictions = logits.argmax(dim=-1)

        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(converted_target),
        )

    if args.file_name in DOMAIN_DATASET:
        eval_metric = metric.compute(average='macro')
    else:
        eval_metric = metric.compute()

    logger.info("** eval **")
    logger.info(f"epoch {epoch}: {eval_metric}")

    if args.task_name == 'cola':
        key = 'matthews_correlation'
    elif args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] or args.file_name in ['MR', 'CR']:
        key = 'accuracy'
    else:
        key = 'f1'

    eval_result = eval_metric[key]
    results.append(eval_result)

    return eval_result

def test(args, model, test_dataloader, metric, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):
    if args.task_name == None or args.k_shot >= 0:
        prompts_discrete_indices = prompts_probs.argmax(1)

        if args.use_ngram:
            prompts_discrete_indices_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_indices_ngram_list.append(ngram_list[idx])
            prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            
            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                prompt_sample = tokenizer.decode(prompts_discrete_indices_ngram_list)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 
            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
            last_hidden_state = sequence_output[0].squeeze()
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]

            predictions = logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(converted_target),
            )
                
        if args.file_name in DOMAIN_DATASET:
            test_metric = metric.compute(average='macro')
        else:
            test_metric = metric.compute()

        if args.task_name == 'mnli':
            for step, batch in enumerate(test_dataloader_mm):
                bsz = len(batch['input_ids'])
                
                if args.use_ngram:
                    batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                    prompt_sample = tokenizer.decode(prompts_discrete_indices_ngram_list)
                else:
                    batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

                mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
                mask_pos = torch.tensor(mask_pos[-1])
                label_to_id = model.config.label2id 
                sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
                last_hidden_state = sequence_output[0].squeeze()
                logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                label = batch["labels"].to(args.device)
                label_keys = list(label_to_id.keys())
                label_map = {}
                for target in label_keys:
                    label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
                converted_target = label.clone()
                for key, val in label_map.items():
                    converted_target[label == key] = val
                interest_index = list(label_map.keys())
                logits = logits[:, interest_index]

                predictions = logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(converted_target),
                )
            test_metric_mm = metric.compute()

        if args.task_name == 'cola':
            key = 'matthews_correlation'
        elif args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] or args.file_name in ['MR', 'CR']:
            key = 'accuracy'
        else:
            key = 'f1'
        test_result = test_metric[key]
        results.append(test_result)

        logger.info("** test **")
        logger.info(f"epoch {epoch}: {test_metric}")
        if args.use_wandb:
            for key in test_metric.keys():
                eval_key = 'Black_test_' + key
                wandb.log({eval_key: test_metric[key]})
            if args.task_name == 'mnli':
                for key in test_metric_mm.keys():
                    eval_key = 'Black_test_' + key + '_mm'
                    wandb.log({eval_key: test_metric_mm[key]})

if __name__ == "__main__":
    main()