import argparse
import logging
import torch
import math
import os
import random
import datasets
import pandas as pd
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from torch.optim import Adam
from transformers import (
    AutoTokenizer,
    SchedulerType,
    set_seed,
)
from transformers.utils.versions import require_version
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
import openai, time
import sys
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
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4},
    "HP": {' unhelpful': 0, ' helpful': 1},
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
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
    'imdb': {0: ' terrible', 1: ' great'},
    'cr': {0: ' terrible', 1: ' great'},
}

TEMPLATE_CONFIG = {
    "mnli": " entailment?",
    "qqp": " equivalent?",
    "sst2": " What is the sentiment?",
    "mrpc": " equivalent?",
    "cola": " correct?",
    "wnli": " What is the relation?",
    'qnli': " entailment?",
    "rte": " entailment?",
    "CI": " What is the intent?",
    "SE": " What is the relation?",
    "RCT": " What is the role?",
    "HP": " Helpful?",
    "sst2": " It was ",
    "imdb": " It was .",
    "cr": " It was ",
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
    parser.add_argument("--model_name_or_path", type=str, default='text-babbage-001', help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--ce_loss", default=True, type=bool, help="If True, will use crossentropy loss.")
    parser.add_argument("--sample_size", type=int, default=10, help="IMPORTANT, sample size per batch")
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--prompt_search_space", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts")
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to run wandb.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=450, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size (per device) for the evaluation dataloader.")
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
    parser.add_argument("--api_key", type=str, default="" , help="GPT-3 API KEY")
    parser.add_argument("--api_limit", type=int, default=8000 , help="The limit of the GPT-3 API request")

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
        with open("./pmi/" + "pmi_" + args.file_name.lower() + "_gpt" + ".txt",'r') as f:
            for line in f:
                result.append(line.strip('\n'))
    elif args.task_name:
        with open("./pmi/" + "pmi_" + args.task_name.lower() + "_gpt" + ".txt",'r') as f:
            for line in f:
                result.append(line.strip('\n'))
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(unique)
    return ngram_index_list

ngram_list = pmi()

def create_batches(dataset, batch_size=2, shuffle=False):
        if isinstance(dataset, dict):
            dataset_dict = dataset
        else:
            dataset_dict = {'input': [], 'labels': []}
            dataset_dict['input'] = dataset['input']
            dataset_dict['labels'] = dataset['labels']
        
        if shuffle:
            dataset_dict = pd.DataFrame(dataset_dict)
            dataset_dict = dataset_dict.sample(frac=1)
            dataset_dict = dataset_dict.to_dict(orient='list')

        batches = {'sentence': [], 'labels':[]}
        for i in range(0,len(dataset_dict['input']),batch_size):
            batches['sentence'].append(dataset_dict['input'][i: i + batch_size])
            batches['labels'].append(dataset_dict['labels'][i: i + batch_size])
        return batches


# @counter
def complete_gpt3(prompt, l, model_name, temp=0.0, num_log_probs=None, echo=False, n=None):
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=100, echo=echo, stop='\n')
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
            
    return response

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        if wrapper.count % 100 == 0:
            print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
        return res
    wrapper.count = 0
    return wrapper

@counter
def train_api_request(prompt, l, model_name, temp=0.0, num_log_probs=None, echo=False, n=None):
    response=complete_gpt3(prompt, l, model_name, temp=temp, num_log_probs=num_log_probs, echo=echo, n=n)
    return response

class ApiCallLimitError(Exception):
    pass

def get_regular_label_probs(response, batch, labels, args, if_null=True, split="train"):
    assert len(response['choices']) == len(batch)
    label_probs = torch.zeros([len(response['choices']), 1, len(labels)])
    all_missing_positions = []
    for a, ans in enumerate(response['choices']):
        for l, label in enumerate(labels):
            if label in ans['logprobs']['tokens']:
                label_probs[a,:,l] = np.exp(ans['logprobs']['token_logprobs'][0])
            else:
                position = (a, l)
                all_missing_positions.append(position)
                
    if len(all_missing_positions) > 0:
        all_additional_prompts = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            missing_prompt = batch[which_sentence] + labels[which_label]
            all_additional_prompts.append(missing_prompt)
        additional_dataset = {'input': all_additional_prompts, 'labels': all_missing_positions}

        batches = create_batches(additional_dataset, batch_size=len(batch[0]))

        for m, missing_batch in enumerate(batches['sentence']):
            if split == "train":
                missing_response = train_api_request(missing_batch, l=0, model_name=args.model_name_or_path, num_log_probs=1, echo=True)
            else: 
                missing_response = complete_gpt3(missing_batch, l=0, model_name=args.model_name_or_path, num_log_probs=1, echo=True)

            for idx, missing_ans in enumerate(missing_response['choices']):
                which_sentence, which_label = batches['labels'][m][idx]
                label_probs[which_sentence,:,which_label] = np.exp(missing_ans['logprobs']['token_logprobs'][-1])

    assert (label_probs > 0).all(), "all should be populated with non-zero value"
            
    if if_null: 
        return label_probs
    label_probs = label_probs / torch.sum(label_probs, dim=2, keepdim=True)
    return label_probs

def main():
    args = parse_args()
    openai.api_key = args.api_key

    ce_loss_string = 'True' if args.ce_loss else 'False'
    # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
    task_name = args.task_name if args.task_name else args.train_file
    args.experiment_id = task_name + str(args.prompt_length) + str(args.prompt_learning_rate) +\
                         str(args.learning_rate) + str(args.num_train_epochs) \
                         + str(args.seed) + str(args.prompt_search_space) + str(args.std) + ce_loss_string

    if args.use_wandb:
        args.group_name = "GPT3_BDPL_" + task_name
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
            # Downloading and loading a dataset from the hub.
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
        id_to_label = LABEL_CONVERT[args.task_name]
    elif args.file_name:
        label_to_id = LABEL2ID_CONFIG[args.file_name]
        id_to_label = LABEL_CONVERT[args.file_name]
    args.num_labels = len(label_to_id)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=not args.use_slow_tokenizer)
    args.device = torch.device("cuda", args.cuda)

    prompt_length = args.prompt_length
    hingeloss = MarginLoss(margin=args.margin, target=False)
    ce_loss = CrossEntropyLoss()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    def preprocess_function(examples):
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

        result= {'input':[]}

        for i in range(len(examples[sentence1_key])):
            if sentence2_key is None:
                ori_sent_id = tokenizer.tokenize(examples[sentence1_key][i])[:400]
                new_sent = tokenizer.convert_tokens_to_string(ori_sent_id)
                result["input"].append('input: '+ new_sent + template_cfg + "\n" + "output:")
            else:
                result["input"].append('input: sentence one: '+ examples[sentence1_key][i] + ' sentence two: ' + examples[sentence2_key][i] + template_cfg + "\n" + "output:")

        if args.task_name or args.file_name in DOMAIN_DATASET:
            result['labels'] = [id_to_label[x] for x in examples["label"]]
        else:
            result['labels'] = examples["label"]

        return result

    def preprocess_function_k_shot(examples):
        # Tokenize texts
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
            elif args.task_name == 'qqp':
                raw_valid_dataset_split = raw_datasets["validation"].train_test_split(test_size=0.025)
                raw_test_dataset = raw_valid_dataset_split['test']
                test_dataset = raw_test_dataset.map(
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
    
    train_batches = create_batches(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    eval_batches = create_batches(eval_dataset, batch_size=args.per_device_eval_batch_size)
    test_batches = create_batches(test_dataset, batch_size=args.per_device_eval_batch_size)
    if args.task_name == 'mnli':
        test_batches_mm = create_batches(test_dataset_mm, batch_size=args.per_device_eval_batch_size)
        test_batches_mm = accelerator.prepare(test_batches_mm)
    else:
        test_batches_mm = None
    eval_batches, test_batches = accelerator.prepare(eval_batches, test_batches)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be shorter in multiprocess)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_batches['sentence']) / args.gradient_accumulation_steps) # 106
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
    best_epoch = 0
    eval_results = []
    test_results = []
    args.loss_back_epoch = args.num_train_epochs
    
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num batches = {len(train_batches['sentence'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    prompt_search_space = args.prompt_search_space
    prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
    prompts_probs.requires_grad = True

    prompt_optimizer = Adam([ {
            "params": [prompts_probs],
            "weight_decay": 0,
            "lr": args.prompt_learning_rate
        },])
    
    best_eval_result = 0.0
    best_prompts_probs = None

    print("----Optimizing black-box prompts----")
    for epoch in range(args.num_train_epochs):
        train_batches = create_batches(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        train_batches = accelerator.prepare(train_batches)

        try:
            for step in range(len(train_batches['sentence'])):
                prompts_dist = torch.distributions.Categorical(prompts_probs)

                with torch.no_grad():
                    if args.trial and step >= 100:
                        break
                    bsz = len(train_batches['sentence'][step])
                    labels = train_batches["labels"][step]
                    loss_list = []
                    prompts_discrete_indices_list = []
                    for k in range(args.sample_size):
                        prompts_discrete_indices = prompts_dist.sample()
                        prompts_discrete_indices_list.append(prompts_discrete_indices)
                        if args.use_ngram:
                            prompts_discrete_ngram_list = []
                            indices_list = prompts_discrete_indices.int().tolist()
                            for idx in indices_list:
                                prompts_discrete_ngram_list.append(ngram_list[idx])
                            
                            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                        else: 
                            indices_list = prompts_discrete_indices.int().tolist()
                            prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

                        batch = []
                        for i in range(len(train_batches['sentence'][step])):
                            batch.append('Definition: ' + prompts_discrete + '\t' + train_batches['sentence'][step][i])

                        responses = train_api_request(batch, l=1, model_name=args.model_name_or_path, num_log_probs=100, echo=False, n=None)
                        label_keys = list(label_to_id.keys())
                        converted_target = torch.tensor([label_to_id[label] for label in labels])

                        label_probs = get_regular_label_probs(responses, batch, label_keys, args, if_null = True)
                        logits = label_probs.squeeze()
                        pred = logits.argmax(dim=-1)

                        if args.ce_loss:
                            loss = ce_loss(logits.view(-1, args.num_labels), converted_target)
                        else:
                            loss = hingeloss(logits, converted_target)
                        loss_list.append(loss.item())

                        if train_api_request.count >= args.api_limit:
                            raise ApiCallLimitError()

                    loss_avg = sum(loss_list) / args.sample_size
                    
                    prompt_optimizer.zero_grad()

                    derivative = [-1 / prompts_probs] * args.sample_size
                    for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                        for i in range(prompt_length):
                            derivative[k][i][prompts_discrete_indices[i]] *= -1

                    prompts_probs.grad = torch.zeros_like(prompts_probs)
                    for k in range(args.sample_size):
                        prompts_probs.grad += 1 / (args.sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]
                
                    torch.nn.utils.clip_grad_norm_(prompts_probs, 3)
                    prompt_optimizer.step()
                    constrainScoreByWholeExact(prompts_probs)

                    if step % args.gradient_accumulation_steps == 0 or step == len(train_batches['sentence']) - 1:
                        progress_bar.update(1)
                        completed_steps += 1
                    if completed_steps >= args.max_train_steps:
                        break

                    
        except ApiCallLimitError:
            pass

        eval_result = evaluate(args, eval_batches, metric, ce_loss, accelerator, epoch, best_epoch, best_eval_result, eval_results, prompts_probs=prompts_probs, prompt_length=prompt_length, tokenizer=tokenizer, linear_layer=None, prompts=None, prompt_embedding_fc=None, label_to_id = label_to_id)

        if eval_result > best_eval_result:
            best_eval_result = eval_result
            best_prompts_probs = prompts_probs

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

        if train_api_request.count >= args.api_limit:
            break

    test(args, test_batches, metric, accelerator, epoch, test_results, prompts_probs=best_prompts_probs, prompt_length=prompt_length, tokenizer=tokenizer, linear_layer=None, prompts=None, label_to_id=label_to_id, test_batches_mm=test_batches_mm)


def evaluate(args, eval_batches, metric, ce_loss, accelerator, epoch, best_epoch, best_result, results, prompts_probs=None, prompt_length=None,tokenizer=None, linear_layer=None, prompts=None, prompt_embedding_fc=None, label_to_id=None):

    if prompts_probs is not None:
        prompts_discrete_indices = prompts_probs.argmax(1)

        if args.use_ngram:
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)

        else: 
            indices_list = prompts_discrete_indices.int().tolist()
            prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)


    for step in range(len(eval_batches['sentence'])):
        if args.trial and step >= 100:
            break

        label = eval_batches["labels"][step]
        
        if prompts_probs is not None:
            batch = []
            for i in range(len(eval_batches['sentence'][step])):
                batch.append('Definition: ' + prompts_discrete + '\t' + eval_batches['sentence'][step][i])

        else: 
            batch = eval_batches['sentence']

        responses = complete_gpt3(batch, l=1, model_name=args.model_name_or_path, num_log_probs=100, echo=False, n=None)

        label_keys = list(label_to_id.keys())

        converted_target = torch.tensor([label_to_id[i] for i in label])

        label_probs = get_regular_label_probs(responses, batch, label_keys, args, if_null = True, split="eval") # logits_only : True 
        logits = label_probs.squeeze()

        eval_loss_c = ce_loss(logits.view(-1, args.num_labels), converted_target)
        predictions = logits.argmax(dim=-1)

        if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
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

def test(args, test_batches, metric, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None, linear_layer=None, prompts=None, label_to_id=None, test_batches_mm=None):
    if args.task_name == None or args.k_shot >= 0:
        if prompts_probs is not None:
            prompts_discrete_indices = prompts_probs.argmax(1)

            if args.use_ngram:
                prompts_discrete_ngram_list = []
                indices_list = prompts_discrete_indices.int().tolist()
                for idx in indices_list:
                    prompts_discrete_ngram_list.append(ngram_list[idx])
                prompts_discrete = ' '.join(prompts_discrete_ngram_list)

            else: 
                indices_list = prompts_discrete_indices.int().tolist()
                prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

        for step in range(len(test_batches['sentence'])):
            if args.trial and step >= 100:
                break
            label = test_batches['labels'][step]

            if prompts_probs is not None:
                batch = []
                for i in range(len(test_batches['sentence'][step])):
                    batch.append('Definition: ' + prompts_discrete + '\t' + test_batches['sentence'][step][i])
            else: 
                batch = test_batches['sentence']
            responses = complete_gpt3(batch, l=1, model_name=args.model_name_or_path, num_log_probs=100, echo=False, n=None)

            label_keys = list(label_to_id.keys())
            converted_target = torch.tensor([label_to_id[i] for i in label])
            label_probs = get_regular_label_probs(responses, batch, label_keys, args, if_null = True, split="test") # logits_only : True 
            logits = label_probs.squeeze()
            predictions = logits.argmax(dim=-1)
            if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
            
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(converted_target),
            )

        if args.file_name in DOMAIN_DATASET:
            test_metric = metric.compute(average='macro')
        else:
            test_metric = metric.compute()

        if args.task_name == 'mnli':
            for step in range(len(test_batches_mm['sentence'])):
                label = test_batches_mm['labels'][step]

                if prompts_probs is not None:
                    batch = []
                    for i in range(len(test_batches_mm['sentence'][step])):
                        batch.append('Definition: ' + prompts_discrete + '\t' + test_batches_mm['sentence'][step][i])
                else: 
                    batch = test_batches_mm['sentence']
                responses = complete_gpt3(batch, l=1, model_name=args.model_name_or_path, num_log_probs=100, echo=False, n=None)

                label_keys = list(label_to_id.keys())
                converted_target = torch.tensor([label_to_id[i] for i in label])
                label_probs = get_regular_label_probs(responses, batch, label_keys, args, if_null = True, split="test") # logits_only : True 
                logits = label_probs.squeeze()
                predictions = logits.argmax(dim=-1)
                if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
                
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