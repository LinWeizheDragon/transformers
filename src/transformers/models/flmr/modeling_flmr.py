# coding=utf-8
# Copyright 2024 FLMR Authors, The Hugging Face Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch FLMR model for Open Domain Question Answering."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import copy
import importlib
import os
import string
import pathlib
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import load
import torch.distributed as dist

from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length

from ... import AutoImageProcessor
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..bert.modeling_bert import BertModel
from ..clip import CLIPVisionModel
from .configuration_flmr import FLMRConfig, FLMRVisionConfig, FLMRTextConfig

from .flmr_utils import (
    get_rank,
    get_world_size,
    get_default_group,
    colbert_score_reduce,
    colbert_score,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FLMRConfig"
_CHECKPOINT_FOR_DOC = "weizhelin/flmr"

FLMR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "weizhelin/flmr",
    # See all FLMR models at https://huggingface.co/models?filter=flmr
]

FLMR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "weizhelin/flmr",
    # See all FLMR models at https://huggingface.co/models?filter=flmr
]

FLMR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "weizhelin/flmr",
    # See all FLMR models at https://huggingface.co/models?filter=flmr
]



##########
# Outputs
##########


@dataclass
class FLMRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of the `doc()` function of [`FLMRModelForRetrieval`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The FLMR encoder outputs the *pooler_output* that corresponds to the embedding of the first token of the context representation. 
            This output can be used to embed questions for nearest neighbors queries with query embeddings.
        late_interaction_output (`torch.FloatTensor` of shape `(batch_size, context_embedding_length, embeddings_size)`):
            The FLMR encoder outputs the *late_interaction_output* that corresponds to the question representation. The embeddings of all tokens are included for late interaction retrieval.
            This output is to be used to embed contexts for late-interaction retrieval with query embeddings.
        context_mask (`torch.FloatTensor` of shape `(batch_size, context_embedding_length)`):
            The FLMR encoder outputs the *context_mask* that corresponds to the mask of the context representation.
    """

    pooler_output: torch.FloatTensor
    late_interaction_output: torch.FloatTensor = None
    context_mask: torch.FloatTensor = None


@dataclass
class FLMRQueryEncoderOutput(ModelOutput):
    """
    Class for outputs of the `query()` function of [`FLMRModelForRetrieval.query()`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The FLMR encoder outputs the *pooler_output* that corresponds to the embedding of the first token of the query representation. 
            This output can be used to embed questions for nearest neighbors queries with context embeddings.
        late_interaction_output (`torch.FloatTensor` of shape `(batch_size, query_embedding_length, embeddings_size)`):
            The FLMR encoder outputs the *late_interaction_output* that corresponds to the question representation. The embeddings of all tokens are included for late interaction retrieval.
            This output is to be used to embed questions for late-interaction retrieval with context embeddings.
    """

    pooler_output: torch.FloatTensor
    late_interaction_output: torch.FloatTensor = None


@dataclass
class FLMRModelForRetrievalOutput(ModelOutput):
    """
    Class for outputs of [`FLMRModelForRetrieval.query()`].

    Args:
        scores (`torch.FloatTensor` of shape `(batch_size, num_positive_examples + num_negative_examples)`):
            The FLMR encoder outputs the *scores* that corresponds to the late-interaction scores of the input query and context. Each query is associated with `num_positive_examples` positive examples and `num_negative_examples` negative examples, and the scores are the late-interaction scores of the query and these examples.
        in_batch_negative_loss (`torch.FloatTensor` of shape `(batch_size, query_embedding_length, embeddings_size)`):
            The FLMR encoder outputs the *late_interaction_output* that corresponds to the question representation. The embeddings of all tokens are included for late interaction retrieval.
            This output is to be used to embed questions for late-interaction retrieval with context embeddings.
    """

    pooler_output: torch.FloatTensor
    late_interaction_output: torch.FloatTensor = None




class FLMRPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FLMRTextModel(FLMRPreTrainedModel):
    base_model_prefix = "flmr_text_model"

    def __init__(self, config: FLMRTextConfig):
        super().__init__(config)
        self.bert_model = BertModel(config, add_pooling_layer=True)
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size



class FLMRVisionModel(FLMRPreTrainedModel):
    base_model_prefix = "flmr_vision_model"

    def __init__(self, config: FLMRVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config)

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

##################
# PreTrainedModel
##################


class FLMRPretrainedModelForRetrieval(FLMRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FLMRConfig
    load_tf_weights = None
    base_model_prefix = "flmr"


###############
# Actual Models
###############


FLMR_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FLMRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        query_tokenizer ([`FLMRQueryEncoderTokenizer]): The tokenizer used for tokenizing the query.
            The query tokenizer can be initialized with `FLMRQueryEncoderTokenizer.from_pretrained(pretrained_model_name_or_path)`.
        context_tokenizer ([`FLMRContextEncoderTokenizer]): The tokenizer used for tokenizing the context.
            The context tokenizer can be initialized with `FLMRContextEncoderTokenizer.from_pretrained(pretrained_model_name_or_path)`.
"""

FLMR_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs (for a pair title+text for example):

            ```
            tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            ```

            (b) For single sequences (for a question for example):

            ```
            tokens:         [CLS] the dog is hairy . [SEP]
            token_type_ids:   0   0   0   0  0     0   0
            ```

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""



class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


@add_start_docstrings(
    "The bare FLMR model that can be used to generate late-interaction embeddings for both multi-modal queries and documents. ",
    FLMR_START_DOCSTRING,
)
class FLMRModelForRetrieval(FLMRPretrainedModelForRetrieval):
    _keys_to_ignore_on_load_unexpected = [r"cls"]
    
    def __init__(
            self, 
            config: FLMRConfig, 
            query_tokenizer=None, 
            context_tokenizer=None
        ):
        super().__init__(config)
        self.config = config

        self.context_text_encoder = FLMRTextModel(config.text_config)
        self.context_text_encoder_linear = nn.Linear(config.text_config.hidden_size, config.dim, bias=False)

        self.query_tokenizer = query_tokenizer
        self.context_tokenizer = context_tokenizer


        self.mapping_network_prefix_length = self.config.mapping_network_prefix_length
        self.vision_encoder_embedding_size = self.config.vision_config.hidden_size
        self.text_encoder_embedding_size = self.config.text_config.hidden_size
        self.late_interaction_embedding_size = self.config.dim

        self.context_vision_projection = MLP(
            (
                self.vision_encoder_embedding_size,
                (self.late_interaction_embedding_size * self.mapping_network_prefix_length) // 2,
                self.late_interaction_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if self.config.use_vision_encoder:
            # self.vision_model_config_class = self.config.vision_model_config_class
            # self.vision_model_class = self.config.vision_model_class
            # self.vision_model_version = self.config.vision_model_version

            self.context_vision_encoder = FLMRVisionModel(config.vision_config)
        
            if self.config.use_transformer_mapping_network:
                # This is a PreFLMR style model
                transformer_mapping_config_base = self.config.transformer_mapping_config_base
                try:
                    from transformers import BertConfig
                    from transformers.models.bert.modeling_bert import BertEncoder
                except Exception as e:
                    raise ImportError(f"Failed to import BertConfig and BertEncoder from transformers. {e}")
            
                transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
                
                assert self.config.text_config.hidden_size == transformer_mapping_config.hidden_size, f"hidden_size {self.config.text_config.hidden_size} != transformer_mapping_config.hidden_size {transformer_mapping_config.hidden_size}. To use cross attention, the dimensions must match."
                # shallow transformer
                transformer_mapping_config.num_hidden_layers = self.config.transformer_mapping_num_hidden_layers
                # add cross attention
                transformer_mapping_config.is_decoder = True
                transformer_mapping_config.add_cross_attention = True
                
                # The linear layer from vision encoder to transformer input
                self.transformer_mapping_input_linear = nn.Linear(self.vision_encoder_embedding_size, transformer_mapping_config.hidden_size)
                
                # The transformer encoder
                self.transformer_mapping_network = BertEncoder(transformer_mapping_config)

                # The linear layer from transformer output to FLMR dim
                self.transformer_mapping_output_linear = nn.Linear(
                    transformer_mapping_config.hidden_size, 
                    self.late_interaction_embedding_size
                )

        if self.config.separate_query_and_context_text_encoder:
            self.query_text_encoder = copy.deepcopy(self.context_text_encoder)
            self.query_text_encoder_linear = copy.deepcopy(self.context_text_encoder_linear)
        else:
            self.query_text_encoder = self.context_text_encoder
            self.query_text_encoder_linear = self.context_text_encoder_linear

        if self.config.separate_query_and_context_vision_encoder:
            self.query_vision_encoder = copy.deepcopy(self.context_vision_encoder)
            self.query_vision_projection = copy.deepcopy(self.context_vision_projection)
        else:
            self.query_vision_encoder = self.context_vision_encoder
            self.query_vision_projection = self.context_vision_projection

        if self.config.load_cpu_extension:
            FLMRModelForRetrieval.try_load_torch_extensions()

        if self.config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.context_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        
        if self.config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(self.config.mask_instruction_token, add_special_tokens=False)[0]
        else:
            self.mask_instruction = False

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def use_gpu(self):
        return self.device.type == "cuda"
    
    @classmethod
    def from_pretrained(self, name_or_path, **kwargs):
        obj = super().from_pretrained(name_or_path, **kwargs)
        return obj

    @classmethod
    def try_load_torch_extensions(cls):
        if hasattr(cls, "loaded_extensions"):
            return

        logger.info(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)
        
        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
                logger.error(f"can not find the separator in the input_ids: {input_ids[i].tolist()}")
        mask = [
            [(x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2) for index, x in enumerate(d)] for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
    
    def forward(
            self, 
            query_input_ids: Optional[torch.Tensor]=None,
            query_attention_mask: Optional[torch.Tensor]=None,
            query_pixel_values: Optional[torch.Tensor]=None,
            query_image_features: Optional[torch.Tensor]=None,
            context_input_ids: Optional[torch.Tensor]=None,
            context_attention_mask: Optional[torch.Tensor]=None,
            context_pixel_values: Optional[torch.Tensor] = None,
            context_image_features: Optional[torch.Tensor] = None,
            use_in_batch_negatives: bool = True,
            in_batch_negatives_from_all_gpus: bool = False, 
            num_negative_examples: int = 1,
            query_concat_output_from_vision_encoder: Optional[bool] = None,
            query_concat_output_from_text_encoder: Optional[bool] = None,
            context_concat_output_from_vision_encoder: Optional[bool] = None,
            context_concat_output_from_text_encoder: Optional[bool] = None,
        ) -> Union[FLMRModelForRetrievalOutput, Tuple[Tensor, ...]]:

        if query_concat_output_from_vision_encoder is None:
            query_concat_output_from_vision_encoder = self.config.query_concat_output_from_vision_encoder
        
        if query_concat_output_from_text_encoder is None:
            query_concat_output_from_text_encoder = self.config.query_concat_output_from_text_encoder
        
        if context_concat_output_from_vision_encoder is None:
            context_concat_output_from_vision_encoder = self.config.context_concat_output_from_vision_encoder

        if context_concat_output_from_text_encoder is None:
            context_concat_output_from_text_encoder = self.config.context_concat_output_from_text_encoder
        
        query_outputs = self.query(
            input_ids=query_input_ids, 
            attention_mask=query_attention_mask,
            pixel_values=query_pixel_values,
            image_features=query_image_features,
            concat_output_from_vision_encoder=query_concat_output_from_vision_encoder,
            concat_output_from_text_encoder=query_concat_output_from_text_encoder,
        )
        Q = query_outputs.late_interaction_output

        context_outputs = self.doc(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            pixel_values=context_pixel_values,
            image_features=context_image_features,
            concat_output_from_vision_encoder=context_concat_output_from_vision_encoder,
            concat_output_from_text_encoder=context_concat_output_from_text_encoder,
            keep_dims='return_mask'
        )
        D, D_mask = context_outputs.late_interaction_output, context_outputs.context_mask

        print(Q.shape, D.shape, D_mask.shape)

        # Gather tensors from other GPUs
        if in_batch_negatives_from_all_gpus:
            Q, D, D_mask = self.gather_tensors_from_other_gpus(Q, D, D_mask)
        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(num_negative_examples+1, dim=0).contiguous()

        scores = self.score(Q_duplicated, D, D_mask)

        if use_in_batch_negatives:
            ib_loss = self.compute_ib_loss_new(Q, D, D_mask)
            return scores, ib_loss

        return scores
    
    def compute_ib_loss_new(self, Q, D, D_mask):
        # Q: batch_size x q_len x dim
        # D: batch_size*n_docs x i_len x dim
        # D_mask: batch_size*n_docs x i_len x dim
        # 1 x batch_size*n_docs x i_len x dim matmul batch_size x 1 x q_len x dim
        # = batch_size x batch_size*n_docs x i_len x q_len

        scores = (D.float().unsqueeze(0) @ Q.float().permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze
        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1))
        
        in_batch_scores = scores.reshape(Q.size(0), -1)

        batch_size = Q.shape[0]
        batch_size_with_pos_and_neg = D.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos
        
        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size  
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(scores.device)
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step*i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)
        
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return loss

    def gather_tensors_from_other_gpus(self, query_embeddings, item_embeddings, item_mask):
        # print("get rank", get_rank())
        # print("get world size", get_world_size())
        # Gather embeddings from other GPUs
        n_nodes = get_world_size()
        if n_nodes == 1:
            return query_embeddings, item_embeddings, item_mask
        # Create placeholder to hold embeddings passed from other ranks
        global_query_embeddings_placeholder = [torch.zeros(*query_embeddings.shape, dtype=query_embeddings.dtype).to(query_embeddings.device) for _ in range(n_nodes)]
        global_item_embeddings_placeholder = [torch.zeros(*item_embeddings.shape, dtype=item_embeddings.dtype).to(item_embeddings.device) for _ in range(n_nodes)]
        global_item_mask_placeholder = [torch.zeros(*item_mask.shape, dtype=item_mask.dtype).to(item_mask.device) for _ in range(n_nodes)]
        dist.all_gather(global_query_embeddings_placeholder, query_embeddings.detach())
        dist.all_gather(global_item_embeddings_placeholder, item_embeddings.detach())
        dist.all_gather(global_item_mask_placeholder, item_mask.detach())

        global_query_embeddings = []
        global_item_embeddings = []
        global_item_mask = []
        # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
        # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
        # input()
        current_rank = get_rank()
        for rank_index, remote_q_embeddings in enumerate(global_query_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_query_embeddings.append(remote_q_embeddings)
            else:
                global_query_embeddings.append(query_embeddings)

        for rank_index, remote_item_embeddings in enumerate(global_item_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_embeddings.append(remote_item_embeddings)
            else:
                global_item_embeddings.append(item_embeddings)
        
        for rank_index, remote_item_mask in enumerate(global_item_mask_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_mask.append(remote_item_mask)
            else:
                global_item_mask.append(item_mask)

        # Replace the previous variables with gathered tensors
        query_embeddings = torch.cat(global_query_embeddings)
        item_embeddings = torch.cat(global_item_embeddings)
        item_mask = torch.cat(global_item_mask)

        return query_embeddings, item_embeddings, item_mask

    
    def query(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor, 
            pixel_values: Optional[torch.Tensor] = None,
            image_features: Optional[torch.Tensor] = None,
            concat_output_from_vision_encoder: Optional[bool] = None,
            concat_output_from_text_encoder: Optional[bool] = None,
        ):

        if concat_output_from_vision_encoder is None:
            concat_output_from_vision_encoder = self.config.query_concat_output_from_vision_encoder
        
        if concat_output_from_text_encoder is None:
            concat_output_from_text_encoder = self.config.query_concat_output_from_text_encoder

        input_modality = []
        if pixel_values is not None or image_features is not None:
            input_modality.append('image')
        if input_ids is not None and attention_mask is not None:
            input_modality.append('text')
        
        if 'image' in input_modality:
            assert pixel_values is not None or image_features is not None, "pixel_values or image_features must be provided if image modality is used"
            assert pixel_values is None or image_features is None, "pixel_values and image_features cannot be provided at the same time"
        
        if 'text' in input_modality:
            assert input_ids is not None and attention_mask is not None, "input_ids and attention_mask must be provided if text modality is used"
            # Forward the text encoder
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            text_encoder_hidden_states = self.query_text_encoder(input_ids, attention_mask=attention_mask)[0]
            text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
            mask = torch.tensor(self.query_mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
            
            text_embeddings = text_embeddings * mask

        if 'image' in input_modality:
            batch_size = pixel_values.shape[0]
            if pixel_values is not None:
                # Forward the vision encoder
                pixel_values = pixel_values.to(self.device)
                if len(pixel_values.shape) == 5:
                    # Multiple ROIs are provided
                    # merge the first two dimensions
                    pixel_values = pixel_values.reshape(-1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4])
                outputs = self.query_vision_encoder(pixel_values, output_hidden_states=True)
                vision_embeddings = outputs.last_hidden_state[:, 0]
            
            if image_features is not None:
                vision_embeddings = image_features.to(self.device)
            
            # Forward the vision projection / mapping network
            vision_embeddings = self.query_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(
                batch_size, -1, self.late_interaction_embedding_size
            )

            if self.config.use_transformer_mapping_network:
                # select the second last layer
                vision_second_last_layer_hidden_states = outputs.hidden_states[-2][:, 1:]
                # transformer_mapping
                transformer_mapping_input_features = self.transformer_mapping_input_linear(vision_second_last_layer_hidden_states)
                
                # Cross attention only attends to the first 32 tokens
                encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
                cross_attention_length = self.config.transformer_mapping_cross_attention_length
                if text_encoder_hidden_states.shape[1] > cross_attention_length:
                    text_encoder_hidden_states = text_encoder_hidden_states[:, :cross_attention_length]
                    encoder_mask = encoder_mask[:, :cross_attention_length]
                
                # Obtain cross attention mask
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
                # Pass through the transformer mapping
                transformer_mapping_output_features = self.transformer_mapping_network(
                    transformer_mapping_input_features,
                    encoder_hidden_states=text_encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                ).last_hidden_state
                # Convert the dimension to FLMR dim
                transformer_mapping_output_features = self.transformer_mapping_output_linear(transformer_mapping_output_features) 
                # Merge with the vision embeddings
                vision_embeddings = torch.cat([vision_embeddings, transformer_mapping_output_features], dim=1)

        if concat_output_from_vision_encoder and concat_output_from_text_encoder:
            Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        elif concat_output_from_vision_encoder:
            Q = vision_embeddings
        elif concat_output_from_text_encoder:
            Q = text_embeddings

        return FLMRQueryEncoderOutput(
            pooler_output=Q[:, 0, :],
            late_interaction_output=torch.nn.functional.normalize(Q, p=2, dim=2),
        )

    def doc(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor, 
            pixel_values: Optional[torch.Tensor] = None,
            image_features: Optional[torch.Tensor] = None,
            concat_output_from_vision_encoder: Optional[bool] = None,
            concat_output_from_text_encoder: Optional[bool] = None,
            keep_dims: Optional[bool] = True,
            return_mask: Optional[bool] = True,
        ):

        assert keep_dims in [True, False, 'return_mask']

        if concat_output_from_vision_encoder is None:
            concat_output_from_vision_encoder = self.config.context_concat_output_from_vision_encoder
        
        if concat_output_from_text_encoder is None:
            concat_output_from_text_encoder = self.config.context_concat_output_from_text_encoder
        
        input_modality = []
        if pixel_values is not None or image_features is not None:
            input_modality.append('image')
        if input_ids is not None and attention_mask is not None:
            input_modality.append('text')

        if 'image' in input_modality:
            assert pixel_values is not None or image_features is not None, "pixel_values or image_features must be provided if image modality is used"
            assert pixel_values is None or image_features is None, "pixel_values and image_features cannot be provided at the same time"
        
        if 'text' in input_modality:
            assert input_ids is not None and attention_mask is not None, "input_ids and attention_mask must be provided if text modality is used"
            # Forward the text encoder
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            text_embeddings = self.context_text_encoder(input_ids, attention_mask=attention_mask)[0]
            text_embeddings = self.context_text_encoder_linear(text_embeddings)

            mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
            text_embeddings = text_embeddings * mask

        if 'image' in input_modality:
            if pixel_values is not None:
                # Forward the vision encoder
                pixel_values = pixel_values.to(self.device)
                outputs = self.context_vision_encoder(pixel_values)
                vision_embeddings = outputs.last_hidden_state[:, 0]
            
            if image_features is not None:
                vision_embeddings = image_features.to(self.device)
            
            batch_size = vision_embeddings.shape[0]
            
            # Forward the vision projection / mapping network
            vision_embeddings = self.context_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(
                -1, self.mapping_network_prefix_length, self.late_interaction_embedding_size
            )

            image_mask = torch.ones(batch_size, vision_embeddings.shape[1], 1).to(self.device)

        if concat_output_from_vision_encoder and concat_output_from_text_encoder:
            # Note: vision embeddings must be in the front since the ColBERT engine only indexes embeddings up to number of 1's in the mask
            # TODO: fix the engine to support masks with discontinuous 0 and 1.
            D = torch.cat([vision_embeddings, text_embeddings], dim=1)
            # concatenate the mask
            mask = torch.cat([mask, image_mask], dim=1)
        elif concat_output_from_vision_encoder:
            D = vision_embeddings
            mask = image_mask
        elif concat_output_from_text_encoder:
            D = text_embeddings
            mask = mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return FLMRContextEncoderOutput(
            pooler_output=D[:, 0, :],
            late_interaction_output=D,
            context_mask=mask.bool() if return_mask else None,
        )

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        # if self.colbert_config.similarity == 'l2':
        #     assert self.colbert_config.interaction == 'colbert'
        #     return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, use_gpu=self.use_gpu)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
    

    # @add_start_docstrings_to_model_forward(FLMR_ENCODERS_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=FLMRQueryEncoderOutput, config_class=_CONFIG_FOR_DOC)
    # def forward(
    #     self,
    #     input_ids: Optional[Tensor] = None,
    #     attention_mask: Optional[Tensor] = None,
    #     token_type_ids: Optional[Tensor] = None,
    #     inputs_embeds: Optional[Tensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[FLMRQueryEncoderOutput, Tuple[Tensor, ...]]:
    #     r"""
    #     Return:

    #     Examples:

    #     ```python
    #     >>> from transformers import FLMRQueryEncoder, FLMRTokenizer

    #     >>> tokenizer = FLMRTokenizer.from_pretrained("facebook/flmr-question_encoder-single-nq-base")
    #     >>> model = FLMRQueryEncoder.from_pretrained("facebook/flmr-question_encoder-single-nq-base")
    #     >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
    #     >>> embeddings = model(input_ids).pooler_output
    #     ```
    #     """
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    #         input_shape = input_ids.size()
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if attention_mask is None:
    #         attention_mask = (
    #             torch.ones(input_shape, device=device)
    #             if input_ids is None
    #             else (input_ids != self.config.pad_token_id)
    #         )
    #     if token_type_ids is None:
    #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    #     outputs = self.text_encoder(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     if not return_dict:
    #         return outputs[1:]
    #     return FLMRQueryEncoderOutput(
    #         pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
    #     )
