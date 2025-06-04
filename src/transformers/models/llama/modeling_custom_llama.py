import torch.nn as nn
import torch
import torch.special
from .modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaModel, 
    LlamaDecoderLayer, 
    LlamaRMSNorm, 
    LlamaForCausalLM,
    FlashAttentionKwargs,
)
from .configuration_llama import LlamaConfig
from transformers.utils import auto_docstring, can_return_tuple
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from ...modeling_outputs import BaseModelOutputWithPast
from ...masking_utils import create_causal_mask

from typing import Optional
import logging


logger = logging.get_logger(__name__)


class CustomLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__(config, device)

    # The difference here is we didn't do torch.no_grad
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        if ORIG_IMPL:
            #print("self.inv_freq.shape", self.inv_freq.shape)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()
            #print("inv_freq_expanded.shape", inv_freq_expanded.shape)
            #print("position_ids_expanded.shape", position_ids_expanded.shape)

            #print("position_ids_expanded", position_ids_expanded)
            #print("inv_freq_expanded", inv_freq_expanded)

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                #print("freqs.shape", freqs.shape)
                emb = torch.cat((freqs, freqs), dim=-1)
                #print("emb.shape", emb.shape)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
        else:
            pass

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        #if cache_position is None:
        #    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #    cache_position = torch.arange(
        #        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #    )

        #if position_ids is None:
        #    position_ids = cache_position.unsqueeze(0)

        if position_ids is None:
            # Create position_ids as a differentiable tensor
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device, 
                dtype=torch.float32  # Use float32 instead of default int64
            ).unsqueeze(0)
            # Make it require gradients
            position_ids = position_ids.clone().detach().requires_grad_(True)
        

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )        


@auto_docstring
class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        print("CustomLlamaForCausalLM")
        super(LlamaForCausalLM, self).__init__(config)
        self.model = CustomLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
