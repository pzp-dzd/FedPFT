import argparse
import os
import random
from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding, \
    BertModel
from distill_model import DistillRobertaModel, DistillBertModel
from sub_model import SubRobertaConfig, SubRobertaModel, SubBertConfig, SubBertModel


def parse_args():
    parser = argparse.ArgumentParser('distill script', add_help=False)
    # random seed
    parser.add_argument('--seed', default=123, type=int)

    # trainer hyperparameters
    parser.add_argument('--per_device_train_batch_size', default=32, type=int)
    parser.add_argument('--save_strategy', default='steps', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--lr_scheduler_type', default='cosine', type=str)
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--logging_steps', default=0.05, type=float)
    parser.add_argument('--save_steps', default=0.1, type=float)
    parser.add_argument('--push_to_hub', default=False, type=bool)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--distill_epoch', default=1, type=int)
    parser.add_argument('--dataloader_num_workers', default=16, type=int)
    parser.add_argument('--remove_unused_columns', default=False, type=bool)
    parser.add_argument('--warm_up_ratio', default=0.06, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--resume_from_checkpoint', default=None, type=str)

    # model hyperparameters
    parser.add_argument('--model_checkpoint', default=None, type=str)

    # dataset hyperparameters
    parser.add_argument('--dataset1_path', default='../datasets/bookcorpus', type=str)
    parser.add_argument('--dataset2_path', default='../datasets/wikipedia', type=str)

    # algorithm hyperparameters
    parser.add_argument('--qk_rank', default=256, type=int)
    parser.add_argument('--intermediate_rank', default=1024, type=int)
    parser.add_argument('--distill', default=False, type=bool)
    parser.add_argument('--layers_retained', default=[], type=int, nargs='+')
    parser.add_argument('--distill_alpha', default=1, type=float)
    parser.add_argument('--split_distill', default=False, type=bool)

    return parser.parse_args()


def construct_sub_model(model, qk_rank, intermediate_rank, sub_model_path, layers_retained):
    if isinstance(model, RobertaModel):
        sub_config = SubRobertaConfig(
            qk_rank=qk_rank,
            intermediate_rank=intermediate_rank,
            layers_retained=layers_retained,
            **model.config.__dict__,
        )
        sub_model = SubRobertaModel(sub_config)
        sub_model.embeddings.load_state_dict(model.embeddings.state_dict())
        sub_model.pooler.load_state_dict(model.pooler.state_dict())
        for i, layer in enumerate(sub_model.encoder.layer):
            layer.attention.load_state_dict(model.encoder.layer[i].attention.state_dict())
            layer.output.LayerNorm.load_state_dict(model.encoder.layer[i].output.LayerNorm.state_dict())
            if i not in layers_retained:
                intermediate_weight = model.encoder.layer[i].intermediate.dense.weight.data  # intermediate_size * dim_i
                intermediate_bias = model.encoder.layer[i].intermediate.dense.bias.data  # intermediate_size
                layer_output_weight = model.encoder.layer[i].output.dense.weight.data  # dim_i * intermediate_size
                layer_output_bias = model.encoder.layer[i].output.dense.bias.data  # dim_i
                neuron_saliency = torch.cat([intermediate_weight, layer_output_weight.transpose(0, 1)], dim=1).norm(dim=1)
                neuron_retained = sorted(neuron_saliency.sort(descending=True)[1][:intermediate_rank])
                new_intermediate_weight = intermediate_weight[neuron_retained]
                new_intermediate_bias = intermediate_bias[neuron_retained]
                new_layer_output_weight = layer_output_weight[:, neuron_retained]
                layer.intermediate.dense.load_state_dict(OrderedDict([
                    ('weight', new_intermediate_weight),
                    ('bias', new_intermediate_bias)
                ]), strict=False)
                layer.output.dense.load_state_dict(OrderedDict([
                    ('weight', new_layer_output_weight),
                    ('bias', layer_output_bias)
                ]), strict=False)
            else:
                layer.intermediate.dense.load_state_dict(model.encoder.layer[i].intermediate.dense.state_dict())
                layer.output.dense.load_state_dict(model.encoder.layer[i].output.dense.state_dict())
        sub_model.save_pretrained(sub_model_path)
    elif isinstance(model, BertModel):
        sub_config = SubBertConfig(
            qk_rank=qk_rank,
            intermediate_rank=intermediate_rank,
            layers_retained=layers_retained,
            **model.config.__dict__,
        )
        sub_model = SubBertModel(sub_config)
        sub_model.embeddings.load_state_dict(model.embeddings.state_dict())
        sub_model.pooler.load_state_dict(model.pooler.state_dict())
        for i, layer in enumerate(sub_model.encoder.layer):
            layer.attention.load_state_dict(model.encoder.layer[i].attention.state_dict())
            layer.output.LayerNorm.load_state_dict(model.encoder.layer[i].output.LayerNorm.state_dict())
            if i not in layers_retained:
                intermediate_weight = model.encoder.layer[
                    i].intermediate.dense.weight.data  # intermediate_size * dim_i
                intermediate_bias = model.encoder.layer[i].intermediate.dense.bias.data  # intermediate_size
                layer_output_weight = model.encoder.layer[i].output.dense.weight.data  # dim_i * intermediate_size
                layer_output_bias = model.encoder.layer[i].output.dense.bias.data  # dim_i
                neuron_saliency = torch.cat([intermediate_weight, layer_output_weight.transpose(0, 1)], dim=1).norm(
                    dim=1)
                neuron_retained = sorted(neuron_saliency.sort(descending=True)[1][:intermediate_rank])
                new_intermediate_weight = intermediate_weight[neuron_retained]
                new_intermediate_bias = intermediate_bias[neuron_retained]
                new_layer_output_weight = layer_output_weight[:, neuron_retained]
                layer.intermediate.dense.load_state_dict(OrderedDict([
                    ('weight', new_intermediate_weight),
                    ('bias', new_intermediate_bias)
                ]), strict=False)
                layer.output.dense.load_state_dict(OrderedDict([
                    ('weight', new_layer_output_weight),
                    ('bias', layer_output_bias)
                ]), strict=False)
            else:
                layer.intermediate.dense.load_state_dict(model.encoder.layer[i].intermediate.dense.state_dict())
                layer.output.dense.load_state_dict(model.encoder.layer[i].output.dense.state_dict())
        sub_model.save_pretrained(sub_model_path)


def main(args):
    model_checkpoint = args.model_checkpoint
    fm = AutoModel.from_pretrained(
        args.model_checkpoint
    )
    sub_model_path = './sub_model/sub-{}-retained{}-r1{}-r2{}'.format(model_checkpoint.split("/")[-1], ''.join(str(i) for i in args.layers_retained), args.qk_rank, args.intermediate_rank)
    if not os.path.exists(sub_model_path):
        construct_sub_model(fm, args.qk_rank, args.intermediate_rank, sub_model_path, args.layers_retained)
    if isinstance(fm, RobertaModel):
        sub_model = SubRobertaModel.from_pretrained(sub_model_path)
    elif isinstance(fm, BertModel):
        sub_model = SubBertModel.from_pretrained(sub_model_path)
    else:
        raise TypeError("unknown pretrain model type")
    if args.distill:
        for params in fm.parameters():
            params.requires_grad = False
        for params in sub_model.parameters():
            params.requires_grad = False
        for i, layer in enumerate(sub_model.encoder.layer):
            if i not in args.layers_retained:
                for name, params in layer.named_parameters():
                    params.requires_grad = False if 'attention' in name else True

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        def preprocess_function(examples):
            token_ids = tokenizer(examples["text"], truncation=True, max_length=512)
            concatenated_examples = {k: list(chain(*token_ids[k])) for k in token_ids.keys()}
            total_length = len(concatenated_examples[list(token_ids.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= 512:
                total_length = (total_length // 512) * 512
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + 512] for i in range(0, total_length, 512)]
                for k, t in concatenated_examples.items()
            }
            return result

        dataset1 = load_dataset(args.dataset1_path)
        tokenized_dataset1 = dataset1.map(preprocess_function, batched=True,
                                          remove_columns=['text'], batch_size=5000, num_proc=32)
        dataset2 = load_dataset(args.dataset2_path, '20220301.en')
        tokenized_dataset2 = dataset2.map(preprocess_function, batched=True,
                                          remove_columns=['text', 'id', 'url', 'title'], batch_size=5000, num_proc=32)
        tokenized_dataset = concatenate_datasets([tokenized_dataset1['train'], tokenized_dataset2['train']])
        model_name = sub_model_path.split("/")[-1]
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        train_args = TrainingArguments(
            "./sub_model/{}-distilled-bookcorpus&wikipedia-lr{}-mgn{}-epoch{}-da{}-bs{}-with-wl".format(model_name, args.learning_rate, args.max_grad_norm, args.distill_epoch, args.distill_alpha, batch_size),
            remove_unused_columns=args.remove_unused_columns,
            save_strategy=args.save_strategy,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            push_to_hub=args.push_to_hub,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warm_up_ratio,
            num_train_epochs=args.distill_epoch,
            dataloader_num_workers=args.dataloader_num_workers,
            save_steps=args.save_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
            logging_first_step=True,
            ddp_find_unused_parameters=False
        )
        if isinstance(fm, RobertaModel):
            model = DistillRobertaModel(sub_model.config, fm, sub_model, args.layers_retained, args.distill_alpha)
        elif isinstance(fm, BertModel):
            model = DistillBertModel(sub_model.config, fm, sub_model, args.layers_retained, args.distill_alpha)
        else:
            raise ValueError("unknown model")
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        train_results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(output_dir="./sub_model/{}-distilled-bookcorpus&wikipedia-lr{}-mgn{}-epoch{}-da{}-bs{}-with-wl".format(model_name, args.learning_rate, args.max_grad_norm, args.distill_epoch, args.distill_alpha, batch_size))


if __name__ == '__main__':
    all_args = parse_args()
    print(all_args)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    os.environ['PYTHONHASHSEED'] = str(all_args.seed)
    main(all_args)
