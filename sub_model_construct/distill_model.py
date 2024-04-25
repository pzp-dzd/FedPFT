import os
from dataclasses import dataclass
import torch
from transformers import RobertaPreTrainedModel, PretrainedConfig, BertPreTrainedModel, ViTPreTrainedModel
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union, Callable, Mapping, Any


@dataclass
class DistillOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DistillRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, teacher, student, layers_retained, alpha=0.0, post_init=True):
        super(DistillRobertaModel, self).__init__(config)
        self.teacher = teacher
        self.student = student
        self.config = config
        self.layers_need_to_distill = \
            sorted(list(set([i for i in range(config.num_hidden_layers)]) - set(layers_retained)))
        self.layers_retained = layers_retained
        self.alpha = alpha

        if post_init:
            self.post_init()

    def forward(self, **kwargs):
        kwargs['return_dict'] = kwargs['return_dict'] if 'return_dict' in kwargs.keys() else self.config.use_return_dict
        kwargs['output_hidden_states'] = True
        kwargs['input_ids'] = kwargs['input_ids'] if 'input_ids' in kwargs.keys() else None
        kwargs['attention_mask'] = kwargs['attention_mask'] if 'attention_mask' in kwargs.keys() else None
        kwargs['token_type_ids'] = kwargs['token_type_ids'] if 'token_type_ids' in kwargs.keys() else None
        kwargs['position_ids'] = kwargs['position_ids'] if 'position_ids' in kwargs.keys() else None
        kwargs['head_mask'] = kwargs['head_mask'] if 'head_mask' in kwargs.keys() else None
        kwargs['inputs_embeds'] = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs.keys() else None
        kwargs['encoder_hidden_states'] = kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs.keys() else None
        kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs.keys() else None
        kwargs['past_key_values'] = kwargs['past_key_values'] if 'past_key_values' in kwargs.keys() else None
        kwargs['use_cache'] = kwargs['use_cache'] if 'use_cache' in kwargs.keys() else None
        kwargs['output_attentions'] = False
        self.teacher.eval()
        self.student.embeddings.eval()
        for i in self.layers_retained:
            self.student.encoder.layer[i].eval()
        for i in self.layers_need_to_distill:
            self.student.encoder.layer[i].attention.eval()
        teacher_outputs = self.teacher(**kwargs)
        student_outputs = self.student(**kwargs)
        loss = 0.0
        for i in self.layers_need_to_distill:
            teacher_output = teacher_outputs[2][i + 1] if self.teacher.pooler is not None else teacher_outputs[1][i + 1]
            student_output = student_outputs[2][i + 1] if self.teacher.pooler is not None else student_outputs[1][i + 1]
            std1 = teacher_output.pow(2).mean().sqrt()
            loss1 = (teacher_output - student_output).div(std1).pow(2).mean()
            teacher_intermediate_weight = self.teacher.encoder.layer[i].intermediate.dense.weight.data
            teacher_output_weight = self.teacher.encoder.layer[i].output.dense.weight.data
            student_intermediate_weight = self.student.encoder.layer[i].intermediate.dense.weight.data
            student_output_weight = self.student.encoder.layer[i].output.dense.weight.data

            # loss2
            w1 = torch.matmul(teacher_output_weight, teacher_intermediate_weight)
            w2 = torch.matmul(student_output_weight, student_intermediate_weight)
            std2 = w1.pow(2).mean().sqrt()
            loss2 = (w1 - w2).div(std2).pow(2).mean()
            loss += loss1 + loss2 * self.alpha
        loss /= len(self.layers_need_to_distill)

        if not kwargs['return_dict']:
            output = student_outputs
            return ((loss,) + output) if loss is not None else output

        return DistillOutput(
            loss=loss,
            last_hidden_state=student_outputs.last_hidden_state,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )

    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.student.load_state_dict(state_dict=state_dict, strict=strict)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        self.student.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )

    def from_pretrained(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        return self.student.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs
        )


class DistillBertModel(BertPreTrainedModel):
    def __init__(self, config, teacher, student, layers_retained, alpha=0.0, post_init=True):
        super(DistillBertModel, self).__init__(config)
        self.teacher = teacher
        self.student = student
        self.config = config
        self.layers_need_to_distill = \
            sorted(list(set([i for i in range(config.num_hidden_layers)]) - set(layers_retained)))
        self.layers_retained = layers_retained
        self.alpha = alpha

        if post_init:
            self.post_init()

    def forward(self, **kwargs):
        kwargs['return_dict'] = kwargs['return_dict'] if 'return_dict' in kwargs.keys() else self.config.use_return_dict
        kwargs['output_hidden_states'] = True
        kwargs['input_ids'] = kwargs['input_ids'] if 'input_ids' in kwargs.keys() else None
        kwargs['attention_mask'] = kwargs['attention_mask'] if 'attention_mask' in kwargs.keys() else None
        kwargs['token_type_ids'] = kwargs['token_type_ids'] if 'token_type_ids' in kwargs.keys() else None
        kwargs['position_ids'] = kwargs['position_ids'] if 'position_ids' in kwargs.keys() else None
        kwargs['head_mask'] = kwargs['head_mask'] if 'head_mask' in kwargs.keys() else None
        kwargs['inputs_embeds'] = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs.keys() else None
        kwargs['encoder_hidden_states'] = kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs.keys() else None
        kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs.keys() else None
        kwargs['past_key_values'] = kwargs['past_key_values'] if 'past_key_values' in kwargs.keys() else None
        kwargs['use_cache'] = kwargs['use_cache'] if 'use_cache' in kwargs.keys() else None
        kwargs['output_attentions'] = False
        self.teacher.eval()
        self.student.embeddings.eval()
        for i in self.layers_retained:
            self.student.encoder.layer[i].eval()
        for i in self.layers_need_to_distill:
            self.student.encoder.layer[i].attention.eval()
        teacher_outputs = self.teacher(**kwargs)
        student_outputs = self.student(**kwargs)
        loss = 0.0
        for i in self.layers_need_to_distill:
            teacher_output = teacher_outputs[2][i + 1] if self.teacher.pooler is not None else teacher_outputs[1][i + 1]
            student_output = student_outputs[2][i + 1] if self.teacher.pooler is not None else student_outputs[1][i + 1]
            std1 = teacher_output.pow(2).mean().sqrt()
            loss1 = (teacher_output - student_output).div(std1).pow(2).mean()

            teacher_intermediate_weight = self.teacher.encoder.layer[i].intermediate.dense.weight.data
            teacher_output_weight = self.teacher.encoder.layer[i].output.dense.weight.data
            student_intermediate_weight = self.student.encoder.layer[i].intermediate.dense.weight.data
            student_output_weight = self.student.encoder.layer[i].output.dense.weight.data

            # loss2
            w1 = torch.matmul(teacher_output_weight, teacher_intermediate_weight)
            w2 = torch.matmul(student_output_weight, student_intermediate_weight)
            std2 = w1.pow(2).mean().sqrt()
            loss2 = (w1 - w2).div(std2).pow(2).mean()
            loss += loss1 + loss2 * self.alpha
        loss /= len(self.layers_need_to_distill)

        if not kwargs['return_dict']:
            output = student_outputs
            return ((loss,) + output) if loss is not None else output

        return DistillOutput(
            loss=loss,
            last_hidden_state=student_outputs.last_hidden_state,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )

    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.student.load_state_dict(state_dict=state_dict, strict=strict)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        self.student.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )

    def from_pretrained(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        return self.student.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs
        )


class DistillViTModel(ViTPreTrainedModel):
    def __init__(self, config, teacher, student, layers_retained, alpha=0.0, post_init=True):
        super(DistillViTModel, self).__init__(config)
        self.teacher = teacher
        self.student = student
        self.config = config
        self.layers_need_to_distill = \
            sorted(list(set([i for i in range(config.num_hidden_layers)]) - set(layers_retained)))
        self.layers_retained = layers_retained
        self.alpha = alpha

        if post_init:
            self.post_init()

    def forward(self, **kwargs):
        kwargs['return_dict'] = kwargs['return_dict'] if 'return_dict' in kwargs.keys() else self.config.use_return_dict
        kwargs['output_hidden_states'] = True
        kwargs['pixel_values'] = kwargs['pixel_values'] if 'pixel_values' in kwargs.keys() else None
        kwargs['head_mask'] = kwargs['head_mask'] if 'head_mask' in kwargs.keys() else None
        kwargs['bool_masked_pos'] = kwargs['bool_masked_pos'] if 'bool_masked_pos' in kwargs.keys() else None
        kwargs['interpolate_pos_encoding'] = kwargs['interpolate_pos_encoding'] if 'interpolate_pos_encoding' in kwargs.keys() else None
        kwargs['output_attentions'] = False
        self.teacher.eval()
        self.student.embeddings.eval()
        for i in self.layers_retained:
            self.student.encoder.layer[i].eval()
        for i in self.layers_need_to_distill:
            self.student.encoder.layer[i].attention.eval()
            self.student.encoder.layer[i].layernorm_before.eval()
            self.student.encoder.layer[i].layernorm_after.eval()
        teacher_outputs = self.teacher(**kwargs)
        student_outputs = self.student(**kwargs)
        loss = 0.0
        for i in self.layers_need_to_distill:
            teacher_output = teacher_outputs[2][i + 1] if self.teacher.pooler is not None else teacher_outputs[1][i + 1]
            student_output = student_outputs[2][i + 1] if self.teacher.pooler is not None else student_outputs[1][i + 1]
            std1 = teacher_output.pow(2).mean().sqrt()
            loss1 = (teacher_output - student_output).div(std1).pow(2).mean()

            if self.alpha > 0:
                teacher_intermediate_weight = self.teacher.encoder.layer[i].intermediate.dense.weight.data
                teacher_output_weight = self.teacher.encoder.layer[i].output.dense.weight.data
                student_intermediate_weight = self.student.encoder.layer[i].intermediate.dense.weight.data
                student_output_weight = self.student.encoder.layer[i].output.dense.weight.data

                # loss2
                w1 = torch.matmul(teacher_output_weight, teacher_intermediate_weight)
                w2 = torch.matmul(student_output_weight, student_intermediate_weight)
                std2 = w1.pow(2).mean().sqrt()
                loss2 = (w1 - w2).div(std2).pow(2).mean()
            else:
                loss2 = 0
            loss += loss1 + loss2 * self.alpha
        loss /= len(self.layers_need_to_distill)

        if not kwargs['return_dict']:
            output = student_outputs
            return ((loss,) + output) if loss is not None else output

        return DistillOutput(
            loss=loss,
            last_hidden_state=student_outputs.last_hidden_state,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )

    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.student.load_state_dict(state_dict=state_dict, strict=strict)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        self.student.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )

    def from_pretrained(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        return self.student.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs
        )
