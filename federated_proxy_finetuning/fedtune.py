import argparse
import json
import os
import random
from itertools import chain
import numpy as np
import torch
from flgo.benchmark.toolkits.partition import IIDPartitioner, DiversityPartitioner, DirichletPartitioner
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, \
    BertForSequenceClassification
from fedparty import Server, Client
import sys
sys.path.append("..//")
from sub_model_construct.sub_model import SubRobertaForSequenceClassification, SplitSubRobertaIntermediate, \
    SplitSubRobertaOutput, SubBertForSequenceClassification, SplitSubBertOutput, SplitSubBertIntermediate


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


def split_dataset(dataset, p=0.0):
    s1 = int(len(dataset) * p)
    s2 = len(dataset) - s1
    if s1 == 0:
        return dataset, None
    elif s2 == 0:
        return None, dataset
    else:
        return torch.utils.data.random_split(dataset, [s2, s1])


def parse_args():
    parser = argparse.ArgumentParser('federated fine-tuning script', add_help=False)
    # random seed
    parser.add_argument('--seed', default=1206, type=int)

    # federate environment hyperparameters
    parser.add_argument('--num_clients', default=100, type=int)
    parser.add_argument('--imbalance', default=0.0, type=float)
    parser.add_argument('--dir_alpha', default=1.0, type=float)
    parser.add_argument('--diversity', default=1.0, type=float)
    parser.add_argument('--partitioner', type=str, default='iid', choices=['iid', 'dir', 'div'])
    parser.add_argument('--train_holdout', default=0.0, type=float)
    parser.add_argument('--test_holdout', default=0.0, type=float)

    # server hyperparameters
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--num_rounds', default=10, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--lr_scheduler_type', default=0, type=int)
    parser.add_argument('--proportion', default=0.2, type=float)
    parser.add_argument('--sample', type=str, default='uniform')
    parser.add_argument('--aggregate', type=str, default='uniform')
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_client', type=bool, default=False)
    parser.add_argument('--metric_for_best_model', default='accuracy', type=str)
    parser.add_argument('--pre_sample', default=True, type=bool)
    parser.add_argument('--align_interval', type=int, default=10)
    parser.add_argument('--align_epochs', type=float, default=0.01)
    parser.add_argument('--align_retained_ratio', type=float, default=0.5)

    # local trainer hyperparameters
    parser.add_argument('--per_device_train_batch_size', default=32, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int)
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--push_to_hub', default=False, type=bool)
    parser.add_argument('--label_names', default=['labels'], type=str, nargs='+')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--dataloader_num_workers', default=16, type=int)
    parser.add_argument('--remove_unused_columns', default=False, type=bool)

    # model hyperparameters
    parser.add_argument('--model_checkpoint', default=None, type=str)
    parser.add_argument('--sub_model_checkpoint', default=None, type=str)

    # lora hyperparameters
    parser.add_argument('--lora_rank', default=4, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--bias', default='none', type=str)
    parser.add_argument('--modules_to_save', default=['classifier'], type=str, nargs='+')
    parser.add_argument('--retained_layers_idx', default=[], type=int, nargs='+')

    # prefix hyperparameters
    parser.add_argument('--num_virtual_tokens', default=32, type=int)

    # dataset hyperparameters
    parser.add_argument('--dataset_path', default='../datasets/glue', type=str)
    parser.add_argument('--task_name', default=None, type=str)
    parser.add_argument('--dataset_label_name', default='label', type=str)
    parser.add_argument('--distill_dataset1_path', default='../datasets/bookcorpus', type=str)
    parser.add_argument('--distill_dataset2_path', default='../datasets/wikipedia', type=str)

    # metric hyperparameters
    parser.add_argument('--evaluate_path', default="../evaluate/metrics/glue", type=str)

    # method hyperparameters 
    parser.add_argument('--method', default="peft", type=str, choices=['peft', 'pft'])
    parser.add_argument('--tuning_method', default="lora", type=str, choices=['lora'])

    return parser.parse_args()


def main(args):
    model_checkpoint = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, truncation=True, max_length=512)

    dataset = load_dataset(args.dataset_path, args.task_name)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['idx', sentence1_key, sentence2_key] if sentence2_key is not None else ['idx', sentence1_key])

    num_classes = dataset['train'].features['label'].num_classes if args.task_name != 'stsb' else 1
    if num_classes != 1:
        id2label = {i: dataset['train'].features['label'].names[i] for i in range(num_classes)}
        label2id = {dataset['train'].features['label'].names[i]: i for i in range(num_classes)}
    para = None if args.partitioner == 'iid' else args.diversity if args.partitioner == 'div' else args.dir_alpha
    fedtask_name = 'dataset{}-c{}-partition{}-im{}-para{}'.format(
        args.task_name if args.task_name else args.dataset_path.split('/')[-1], args.num_clients, args.partitioner, args.imbalance, para)
    args.fedtask_name = fedtask_name
    fedtask_path = os.path.join('./task', fedtask_name)
    if not os.path.exists(fedtask_path):
        os.makedirs(fedtask_path)
        if args.partitioner == 'iid':
            data_partitioner = IIDPartitioner(num_clients=args.num_clients, imbalance=args.imbalance)
        elif args.partitioner == 'div':
            data_partitioner = DiversityPartitioner(num_clients=args.num_clients, diversity=args.diversity,
                                                    index_func=lambda X: [xi[args.dataset_label_name] for xi in X])
        elif args.partitioner == 'dir':
            data_partitioner = DirichletPartitioner(
                num_clients=args.num_clients, alpha=args.dir_alpha,
                index_func=lambda X: [xi[args.dataset_label_name] for xi in X], imbalance=args.imbalance
            )
        else:
            raise TypeError('unknown partitioner type')

        local_datas_idx = data_partitioner(tokenized_dataset['train'])
        feddata = {}
        for cid in range(args.num_clients):
            feddata[str(cid)] = {'data': local_datas_idx[cid]}
        with open(os.path.join(fedtask_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
    else:
        with open(os.path.join(fedtask_path, 'data.json'), 'r') as outf:
            feddata = json.load(outf)

    if args.task_name != 'mnli':
        server_test_data, server_val_data = split_dataset(tokenized_dataset['validation'], args.test_holdout)
        server_test_data = tokenized_dataset['validation'].select(server_test_data.indices) if args.test_holdout > 0 else \
                    tokenized_dataset['validation']
        server_val_data = tokenized_dataset['validation'].select(server_val_data.indices) if args.test_holdout > 0 else None
    else:
        server_matched_test_data, server_matched_val_data = split_dataset(
            tokenized_dataset['validation_matched'], args.test_holdout)
        server_matched_test_data = tokenized_dataset['validation_matched'].select(
            server_matched_test_data.indices) if args.test_holdout > 0 else tokenized_dataset['validation_matched']
        server_matched_val_data = tokenized_dataset['validation_matched'].select(
            server_matched_val_data.indices) if args.test_holdout > 0 else None

        server_mismatched_test_data, server_mismatched_val_data = split_dataset(
            tokenized_dataset['validation_mismatched'], args.test_holdout)
        server_mismatched_test_data = tokenized_dataset['validation_mismatched'].select(
            server_mismatched_test_data.indices) if args.test_holdout > 0 else tokenized_dataset['validation_mismatched']
        server_mismatched_val_data = tokenized_dataset['validation_mismatched'].select(
            server_mismatched_val_data.indices) if args.test_holdout > 0 else None

        server_test_data = {
            'matched': server_matched_test_data,
            'mismatched': server_mismatched_test_data
        }
        server_val_data = {
            'matched': server_matched_val_data,
            'mismatched': server_mismatched_val_data
        } if args.test_holdout > 0 else None
    fm = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    ) if num_classes > 1 else AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_classes
    )
    if args.tuning_method == 'lora':
        target_modules = []
        for i in range(fm.config.num_hidden_layers):
            if isinstance(fm, RobertaForSequenceClassification) or isinstance(fm, BertForSequenceClassification):
                target_modules.append('layer.{}.attention.self.value'.format(i))
                target_modules.append('layer.{}.attention.output.dense'.format(i))
                target_modules.append('layer.{}.attention.self.key'.format(i))
                target_modules.append('layer.{}.attention.self.query'.format(i))
            else:
                raise TypeError('unknown model type')
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            modules_to_save=args.modules_to_save,
        )
    else:
        raise ValueError('unknown tuning methods')
    fm = get_peft_model(fm, peft_config)
    if args.method == 'peft':
        sub_fm = fm
    elif args.method == 'pft':
        if 'roberta' in model_checkpoint:
            sub_fm = SubRobertaForSequenceClassification.from_pretrained(
                args.sub_model_checkpoint,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id
            )
            if args.align_interval < args.num_rounds and args.align_retained_ratio > 0:
                num_neuron = sub_fm.base_model.config.intermediate_rank  # 200
                hidden_size = sub_fm.base_model.config.hidden_size
                hidden_act = sub_fm.base_model.config.hidden_act
                hidden_dropout_prob = sub_fm.base_model.config.hidden_dropout_prob
                layer_norm_eps = sub_fm.base_model.config.layer_norm_eps

                num_retained_neuron = int(num_neuron * args.align_retained_ratio)  # 150 不用训练
                num_other_neuron = num_neuron - num_retained_neuron  # 50
                for i, layer in enumerate(sub_fm.base_model.encoder.layer):
                    if i not in args.retained_layers_idx:
                        setattr(layer, 'split_intermediate',
                                SplitSubRobertaIntermediate(
                                    input_dim=hidden_size,
                                    output_dim1=num_retained_neuron,
                                    output_dim2=num_other_neuron,
                                    hidden_act=hidden_act
                                ))
                        setattr(layer, 'split_output',
                                SplitSubRobertaOutput(
                                    input_dim1=num_retained_neuron,
                                    input_dim2=num_other_neuron,
                                    output_dim=hidden_size,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    layer_norm_eps=layer_norm_eps
                                ))
        elif 'bert' in model_checkpoint:
            sub_fm = SubBertForSequenceClassification.from_pretrained(
                args.sub_model_checkpoint,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id
            )
            if args.align_interval < args.num_rounds and args.align_retained_ratio > 0:
                num_neuron = sub_fm.base_model.config.intermediate_rank
                hidden_size = sub_fm.base_model.config.hidden_size
                hidden_act = sub_fm.base_model.config.hidden_act
                hidden_dropout_prob = sub_fm.base_model.config.hidden_dropout_prob
                layer_norm_eps = sub_fm.base_model.config.layer_norm_eps

                num_retained_neuron = int(num_neuron * args.align_retained_ratio)
                num_other_neuron = num_neuron - num_retained_neuron
                for i, layer in enumerate(sub_fm.base_model.encoder.layer):
                    if i not in args.retained_layers_idx:
                        setattr(layer, 'split_intermediate',
                                SplitSubBertIntermediate(
                                    input_dim=hidden_size,
                                    output_dim1=num_retained_neuron,
                                    output_dim2=num_other_neuron,
                                    hidden_act=hidden_act
                                ))
                        setattr(layer, 'split_output',
                                SplitSubBertOutput(
                                    input_dim1=num_retained_neuron,
                                    input_dim2=num_other_neuron,
                                    output_dim=hidden_size,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    layer_norm_eps=layer_norm_eps
                                ))
        else:
            raise TypeError('unknown model')
        sub_fm = get_peft_model(sub_fm, peft_config)
    else:
        raise TypeError('unknown method')
    server = Server(
        config=vars(args),
        model=fm,
        sub_model=sub_fm,
        tokenizer=tokenizer,
        test_data=server_test_data,
        val_data=server_val_data
    )
    if args.align_interval < args.num_rounds and args.method == 'pft':
        def preprocess_function2(examples):
            token_ids = tokenizer(examples["text"], truncation=True, max_length=512)
            concatenated_examples = {k: list(chain(*token_ids[k])) for k in token_ids.keys()}
            total_length = len(concatenated_examples[list(token_ids.keys())[0]])
            if total_length >= 512:
                total_length = (total_length // 512) * 512
            result = {
                k: [t[i: i + 512] for i in range(0, total_length, 512)]
                for k, t in concatenated_examples.items()
            }
            return result

        distill_dataset1 = load_dataset(args.distill_dataset1_path)
        tokenized_distill_dataset1 = distill_dataset1.map(preprocess_function2, batched=True,
                                          remove_columns=['text'], batch_size=5000, num_proc=32)
        distill_dataset2 = load_dataset(args.distill_dataset2_path, '20220301.en')
        tokenized_distill_dataset2 = distill_dataset2.map(preprocess_function2, batched=True,
                                          remove_columns=['text', 'id', 'url', 'title'], batch_size=5000, num_proc=32)
        tokenized_distill_dataset = concatenate_datasets([tokenized_distill_dataset1['train'], tokenized_distill_dataset2['train']])
        server.align_dataset = tokenized_distill_dataset
    clients = []
    for i in range(args.num_clients):
        client_data = tokenized_dataset['train'].select(feddata[str(i)]['data'])
        client_train_data, client_val_data = split_dataset(client_data, args.train_holdout)
        client = Client(
            config=vars(args),
            model=sub_fm,
            tokenizer=tokenizer,
            train_data=client_data.select(client_train_data.indices) if args.train_holdout > 0 else client_data,
            val_data=client_data.select(client_val_data.indices) if args.train_holdout > 0 else None,
            id=i
        )
        clients.append(client)
    server.register_clients(clients)
    server.run()


if __name__ == '__main__':
    all_args = parse_args()
    print(all_args)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    os.environ['PYTHONHASHSEED'] = str(all_args.seed)
    main(all_args)

