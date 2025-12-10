# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

# Import Cache classes for Qwen3 compatibility
try:
    from transformers.cache_utils import DynamicCache, StaticCache
    CACHE_AVAILABLE = True
except ImportError:
    try:
        from transformers.modeling_utils import DynamicCache, StaticCache
        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # Detect if this is a Qwen3 model (requires DynamicCache)
        self.is_qwen3 = False
        if CACHE_AVAILABLE:
            try:
                from transformers.models.qwen3 import Qwen3ForCausalLM
                self.is_qwen3 = isinstance(self.base_causallm, Qwen3ForCausalLM)
            except ImportError:
                # Try alternative import path
                try:
                    model_type = type(self.base_causallm).__name__
                    self.is_qwen3 = 'Qwen3' in model_type or 'qwen3' in str(type(self.base_causallm)).lower()
                except:
                    pass

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                # Handle different past_key_values formats (list vs Cache object)
                # For Qwen3, we MUST use DynamicCache, not list
                if self.is_qwen3 and CACHE_AVAILABLE:
                    # Qwen3 requires DynamicCache format
                    from transformers.cache_utils import DynamicCache
                    
                    # Convert kv_cache to legacy format if it's a Cache object
                    if hasattr(kv_cache, 'to_legacy_cache'):
                        legacy_cache = kv_cache.to_legacy_cache()
                    elif isinstance(kv_cache, (list, tuple)):
                        legacy_cache = kv_cache
                    else:
                        raise TypeError(f"Unknown kv_cache type for Qwen3: {type(kv_cache)}")
                    
                    # Slice the cache
                    sliced_cache = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in legacy_cache
                    ]
                    
                    # Convert back to DynamicCache (required for Qwen3)
                    past_key_values = DynamicCache.from_legacy_cache(sliced_cache)
                    
                elif CACHE_AVAILABLE and hasattr(kv_cache, 'to_legacy_cache'):
                    # Other models with Cache object (but not Qwen3)
                    # Convert to legacy format, slice, then convert back
                    legacy_cache = kv_cache.to_legacy_cache()
                    past_key_values = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in legacy_cache
                    ]
                    # Convert back to Cache object
                    from transformers.cache_utils import DynamicCache
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                elif isinstance(kv_cache, (list, tuple)):
                    # Old format: list of tuples (for GPT2, LLaMA, etc.)
                    past_key_values = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]
                else:
                    # Fallback: try to convert to list
                    past_key_values = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values
            
            # For Qwen3, ensure kv_cache is always DynamicCache (not list)
            # This handles cases where FSDP or other wrappers might convert it to list
            if self.is_qwen3 and CACHE_AVAILABLE and isinstance(kv_cache, (list, tuple)):
                from transformers.cache_utils import DynamicCache
                kv_cache = DynamicCache.from_legacy_cache(kv_cache)

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        # Handle past_key_values format for final pass (same logic as in loop)
        final_past_key_values = None
        if kv_cache:
            if self.is_qwen3 and CACHE_AVAILABLE:
                # Qwen3 requires DynamicCache format
                from transformers.cache_utils import DynamicCache
                
                # Convert kv_cache to legacy format if it's a Cache object
                if hasattr(kv_cache, 'to_legacy_cache'):
                    legacy_cache = kv_cache.to_legacy_cache()
                elif isinstance(kv_cache, (list, tuple)):
                    legacy_cache = kv_cache
                else:
                    raise TypeError(f"Unknown kv_cache type for Qwen3 final pass: {type(kv_cache)}")
                
                # Slice the cache
                sliced_cache = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in legacy_cache
                ]
                
                # Convert back to DynamicCache (required for Qwen3)
                final_past_key_values = DynamicCache.from_legacy_cache(sliced_cache)
            elif CACHE_AVAILABLE and hasattr(kv_cache, 'to_legacy_cache'):
                # Other models with Cache object
                legacy_cache = kv_cache.to_legacy_cache()
                sliced_cache = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in legacy_cache
                ]
                from transformers.cache_utils import DynamicCache
                final_past_key_values = DynamicCache.from_legacy_cache(sliced_cache)
            elif isinstance(kv_cache, (list, tuple)):
                # Old format: list of tuples (for GPT2, LLaMA, etc.)
                final_past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
            else:
                # Fallback
                final_past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
        
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=final_past_key_values,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0
        batch_size = input_ids.shape[0]

        # Initialize tokens list for each sample in batch
        tokens_list = [input_ids[i].detach().tolist() for i in range(batch_size)]
        
        # Track which samples are still generating (not finished)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).repeat(batch_size, 1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_tokens = torch.argmax(outputs.logits[:, -1], dim=-1)  # (batch_size,)
        
        # Update tokens for all active samples
        for i in range(batch_size):
            if active_mask[i]:
                tokens_list[i].append(next_tokens[i].item())
                if next_tokens[i].item() == self.eos_token_id:
                    active_mask[i] = False

        # Prepare embeddings for next step
        new_token_embeds = self.embedding(next_tokens).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embeds), dim=1)

        # get other tokens
        for step in range(max_new_tokens - 1):
            if not active_mask.any():
                break
                
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_tokens = torch.argmax(outputs.logits[:, -1], dim=-1)  # (batch_size,)
            
            # Update tokens for active samples
            for i in range(batch_size):
                if active_mask[i]:
                    tokens_list[i].append(next_tokens[i].item())
                    if next_tokens[i].item() == self.eos_token_id:
                        active_mask[i] = False
            
            # Prepare embeddings for next step (only for active samples)
            if active_mask.any():
                # For finished samples, use padding token embedding (or repeat last)
                next_tokens_for_embed = next_tokens.clone()
                # Use eos_token_id for finished samples to maintain shape
                new_token_embeds = self.embedding(next_tokens_for_embed).unsqueeze(1)
                new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embeds), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                # Continue with current embeddings for syncing (even if all samples finished)
                # This ensures same number of forward passes across all devices
                if active_mask.any() or new_inputs_embeds is not None:
                    # Continue with existing embeddings
                    _ = self.base_causallm(inputs_embeds=new_inputs_embeds)
                    # Append dummy token for finished samples to maintain shape
                    if not active_mask.any():
                        dummy_embeds = self.embedding(torch.tensor([self.eos_token_id] * batch_size, device=input_ids.device)).unsqueeze(1)
                        new_inputs_embeds = torch.cat((new_inputs_embeds, dummy_embeds), dim=1)
                else:
                    # Fallback: create dummy input if needed
                    dummy_embeds = self.embedding(torch.tensor([self.eos_token_id] * batch_size, device=input_ids.device)).unsqueeze(1)
                    _ = self.base_causallm(inputs_embeds=dummy_embeds)
                    new_inputs_embeds = dummy_embeds

        # Convert tokens_list to tensor, padding to same length
        max_len = max(len(tokens) for tokens in tokens_list)
        padded_tokens = []
        for tokens in tokens_list:
            padded = tokens + [self.eos_token_id] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        result_tensor = torch.tensor(padded_tokens, device=input_ids.device)

        if output_embedding:
            # for analysis purpose
            return result_tensor, new_inputs_embeds
        else:
            return result_tensor
