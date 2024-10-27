"""PyTorch Custom GPT model."""

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F

# from transformers.file_utils import cached_path
from huggingface_hub import cached_download as cached_path
from huggingface_hub import cached_download, hf_hub_download, snapshot_download

from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'GPT-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-base-uncased.tar.gz",
    'GPT-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-large-uncased.tar.gz",
    'GPT-base-cased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-base-cased.tar.gz",
    'GPT-large-cased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-large-cased.tar.gz",
    'GPT-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-base-multilingual-uncased.tar.gz",
    'GPT-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-base-multilingual-cased.tar.gz",
    'GPT-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/GPT/GPT-base-chinese.tar.gz",
}
# CONFIG_NAME = 'GPT_config.json'
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class GPTConfig(object):
    # Used later in PreTrainedGPTModel

    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                            "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPTConfig` from a Python dictionary of parameters."""
        config = GPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class GPTLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(GPTLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GPTEmbeddings(nn.Module):

    def __init__(self, config):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = GPTLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTSelfAttention(nn.Module):
    def __init__(self, config):
        super(GPTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Added causal mask for GPT. 
        causal_mask = torch.tril(torch.ones(attention_scores.size(-2), attention_scores.size(-1), device=hidden_states.device))
        attention_scores = attention_scores * causal_mask - 1e10 * (1 - causal_mask)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


class GPTSelfOutput(nn.Module):
    def __init__(self, config):
        super(GPTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = GPTLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GPTAttention(nn.Module):
    def __init__(self, config):
        super(GPTAttention, self).__init__()
        self.self = GPTSelfAttention(config)
        self.output = GPTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, att_score = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_score


class GPTIntermediate(nn.Module):
    def __init__(self, config):
        super(GPTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)  # [batch, max_len, nslot]
        if tmp_score is not None:
            hidden_states[:, tgt_pos, :] = tmp_score  # hidden state is of the size of [16, 128, 3072], but it only changes [16, 3, 3072]. 
        if imp_op == 'return':
            imp_weights = []
        if imp_pos is not None:
            for layer, pos in imp_pos:
                if imp_op == 'remove':
                    hidden_states[:, tgt_pos, pos] = 0.0
                if imp_op == 'enhance':
                    hidden_states[:, tgt_pos, pos] *= 2.0
                if imp_op == 'return':
                    imp_weights.append(hidden_states[0, tgt_pos, pos].item())

        if imp_op == 'return':
            return hidden_states, imp_weights
        else:
            return hidden_states


class GPTOutput(nn.Module):
    def __init__(self, config):
        super(GPTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = GPTLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GPTLayer(nn.Module):
    def __init__(self, config):
        super(GPTLayer, self).__init__()
        self.attention = GPTAttention(config)
        self.intermediate = GPTIntermediate(config)
        self.output = GPTOutput(config)

    def forward(self, hidden_states, attention_mask, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        attention_output, att_score = self.attention(hidden_states, attention_mask)
        if imp_op == 'return':
            intermediate_output, imp_weights = self.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        else:
            intermediate_output = self.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        layer_output = self.output(intermediate_output, attention_output)
        if imp_op == 'return':
            return layer_output, intermediate_output, imp_weights
        else:
            return layer_output, intermediate_output


class GPTEncoder(nn.Module):
    def __init__(self, config):
        super(GPTEncoder, self).__init__()
        self.layer = nn.ModuleList([GPTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, tgt_layer=None, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        # [1, 128, 768] for hidden states
        all_encoder_layers = []
        ffn_weights = None
        if imp_op == 'return':
            imp_weights = []
        for layer_index, layer_module in enumerate(self.layer):
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
            else:
                imp_pos_at_this_layer = None
            # TODO, we should understand more about here. 
            if imp_op == 'return':
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                imp_weights.extend(imp_weights_l)
            else:
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
        all_encoder_layers.append(hidden_states)
        if imp_op == 'return':
            return all_encoder_layers, ffn_weights, imp_weights
        else:
            return all_encoder_layers, ffn_weights


class GPTPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(GPTPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = GPTLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GPTLMPredictionHead(nn.Module):
    def __init__(self, config, GPT_model_embedding_weights):
        super(GPTLMPredictionHead, self).__init__()
        self.transform = GPTPredictionHeadTransform(config)

        self.decoder = nn.Linear(GPT_model_embedding_weights.size(1),
                                GPT_model_embedding_weights.size(0),
                                bias=False)
        self.decoder.weight = GPT_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(GPT_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class GPTOnlyLMHead(nn.Module):
    def __init__(self, config, GPT_model_embedding_weights):
        super(GPTOnlyLMHead, self).__init__()
        self.predictions = GPTLMPredictionHead(config, GPT_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PreTrainedGPTModel(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedGPTModel, self).__init__()
        if not isinstance(config, GPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPTConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_GPT_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, GPTLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        resolved_archive_file = None
        # Attempt to load from Hugging Face's cache or local files
        try:
            resolved_archive_file = snapshot_download(repo_id=pretrained_model_name)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return None

        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info(f"Extracting archive file {resolved_archive_file} to temp dir {tempdir}")
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir

        # Load configuration
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = GPTConfig.from_json_file(config_file)
        logger.info(f"Model config loaded: {config}")

        # Instantiate model
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        # Map keys in state_dict to align with custom model's architecture
        state_dict = cls._map_state_dict_keys(state_dict)

        missing_keys, unexpected_keys, error_msgs = [], [], []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # load(model, prefix='' if hasattr(model, 'gpt') else 'gpt.')
        load(model, prefix='' if hasattr(model, 'gpt') else 'gpt.')
        
        # Logging
        if missing_keys:
            logger.info(f"Weights not initialized from pretrained model: {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unused weights in pretrained model: {unexpected_keys}")

        if tempdir:
            shutil.rmtree(tempdir)

        return model

    @staticmethod
    def _map_state_dict_keys(state_dict):
        """Map or rename keys in the state_dict to align with custom GPT architecture, and handle attention/embedding discrepancies."""
        mapped_state_dict = {}
        for key, value in state_dict.items():
            # Skip GPT-style prediction head weights if unused
            if key.startswith("cls.predictions"):
                continue

            # Map embedding weights and position embeddings, but skip embedding LayerNorm if unused
            if key == "wte.weight":
                new_key = "gpt.embeddings.word_embeddings.weight"
            elif key == "wpe.weight":
                new_key = "gpt.embeddings.position_embeddings.weight"
            elif key == "ln_f.weight":
                new_key = "gpt.final_layer_norm.weight"
            elif key == "ln_f.bias":
                new_key = "gpt.final_layer_norm.bias"

            # Handle attention components for custom split or combined QKV in GPT-2
            elif key.startswith("h.") and "attn.c_attn" in key:
                layer_index = key.split(".")[1]
                # Split c_attn weights into query, key, value if needed
                if "weight" in key:
                    qkv_weight = value.chunk(3, dim=1)
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.query.weight"] = qkv_weight[0]
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.key.weight"] = qkv_weight[1]
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.value.weight"] = qkv_weight[2]
                    continue  # Skip adding this key to mapped_state_dict
                elif "bias" in key:
                    qkv_bias = value.chunk(3, dim=0)
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.query.bias"] = qkv_bias[0]
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.key.bias"] = qkv_bias[1]
                    mapped_state_dict[f"gpt.encoder.layer.{layer_index}.attention.self.value.bias"] = qkv_bias[2]
                    continue  # Skip adding this key to mapped_state_dict
                
            # Map other sub-layer components in the transformer
            elif key.startswith("h."):
                layer_index = key.split(".")[1]
                if "attn.c_proj" in key:
                    new_key = key.replace(f"h.{layer_index}.attn.c_proj", f"gpt.encoder.layer.{layer_index}.attention.output.dense")
                elif "ln_1" in key:
                    new_key = key.replace(f"h.{layer_index}.ln_1", f"gpt.encoder.layer.{layer_index}.attention.output.LayerNorm")
                elif "ln_2" in key:
                    new_key = key.replace(f"h.{layer_index}.ln_2", f"gpt.encoder.layer.{layer_index}.output.LayerNorm")
                elif "mlp.c_fc" in key:
                    new_key = key.replace(f"h.{layer_index}.mlp.c_fc", f"gpt.encoder.layer.{layer_index}.intermediate.dense")
                elif "mlp.c_proj" in key:
                    new_key = key.replace(f"h.{layer_index}.mlp.c_proj", f"gpt.encoder.layer.{layer_index}.output.dense")
                else:
                    new_key = key  # Keep as-is if no specific mapping is identified

            # Final layer normalization
            elif key.startswith("ln_f"):
                new_key = key.replace("ln_f", "gpt.final_layer_norm")

            # Keep other keys as-is if not identified for specific mapping
            else:
                new_key = key

            mapped_state_dict[new_key] = value

        return mapped_state_dict


class GPTModel(PreTrainedGPTModel):

    def __init__(self, config):
        super(GPTModel, self).__init__(config)
        self.embeddings = GPTEmbeddings(config)
        self.encoder = GPTEncoder(config)
        self.apply(self.init_GPT_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, tgt_pos=None, tgt_layer=None, tmp_score=None, imp_pos=None, imp_op=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if imp_op == 'return':
            encoded_layers, ffn_weights, imp_weights = self.encoder(embedding_output,
                                        extended_attention_mask,
                                        tgt_layer=tgt_layer,
                                        tgt_pos=tgt_pos,
                                        tmp_score=tmp_score,
                                        imp_pos=imp_pos,
                                        imp_op=imp_op
                                        )
        else:
            encoded_layers, ffn_weights = self.encoder(embedding_output,
                                        extended_attention_mask,
                                        tgt_layer=tgt_layer,
                                        tgt_pos=tgt_pos,
                                        tmp_score=tmp_score,
                                        imp_pos=imp_pos,
                                        imp_op=imp_op
                                        )
        sequence_output = encoded_layers[-1]
        if imp_op == 'return':
            return sequence_output, ffn_weights, imp_weights
        else:
            return sequence_output, ffn_weights


class GPTForCausalLM(PreTrainedGPTModel):

    def __init__(self, config):
        super(GPTForCausalLM, self).__init__(config)
        self.gpt = GPTModel(config)
        self.cls = GPTOnlyLMHead(config, self.gpt.embeddings.word_embeddings.weight)
        self.apply(self.init_GPT_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None, imp_pos=None, imp_op=None):
        if tmp_score is not None:
            batch_size = tmp_score.shape[0]
            input_ids = input_ids.repeat(batch_size, 1)
            token_type_ids = token_type_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
        if imp_op == 'return':
            last_hidden, ffn_weights, imp_weights = self.gpt(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size), (n_imp_pos)
        else:
            last_hidden, ffn_weights = self.gpt(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)  # (batch, max_len, hidden_size), (batch, max_len, ffn_size)
        last_hidden = last_hidden[:, tgt_pos, :]  # (batch, hidden_size)
        ffn_weights = ffn_weights[:, tgt_pos, :]  # (batch, ffn_size)
        tgt_logits = self.cls(last_hidden)  # (batch, n_vocab)
        tgt_prob = F.softmax(tgt_logits, dim=1)  # (batch, n_vocab)

        # get the max probability
        max_index = torch.argmax(tgt_prob, dim=1)
        # get the value of the max probability
        max_prob = torch.max(tgt_prob, dim=1)

        if imp_op == 'return':
            return imp_weights
        else:
            if tmp_score is None:
                # return ffn_weights at a layer and the final logits at the [MASK] position
                return ffn_weights, tgt_logits
            else:
                # return final probabilities and grad at a layer at the [MASK] position
                gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, tgt_label]), tmp_score) 
                print(tgt_prob[:, tgt_label])
                # Let's print tgt_prob[:, tgt_label]
                # # tensor([6.2380e-06, 6.2274e-06, 6.2170e-06, 6.2066e-06, 6.1963e-06, 6.1862e-06,
                # 6.1761e-06, 6.1661e-06, 6.1561e-06, 6.1463e-06, 6.1366e-06, 6.1270e-06,
                # 6.1174e-06, 6.1080e-06, 6.0987e-06, 6.0895e-06], device='cuda:0',
                # grad_fn=<SelectBackward0>)
                # In the next batch 
                # [6.0804e-06, 6.0715e-06, 6.0626e-06, 6.0539e-06, 6.0453e-06, 6.0368e-06,
                # 6.0285e-06, 6.0202e-06, 6.0121e-06, 6.0042e-06, 5.9963e-06, 5.9886e-06,
                # 5.9811e-06, 5.9737e-06, 5.9663e-06, 5.9592e-06]
                return tgt_prob, gradient[0] # which is [16, 3072]
