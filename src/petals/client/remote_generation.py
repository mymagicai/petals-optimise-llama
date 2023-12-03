import contextlib
import dataclasses
from contextvars import ContextVar
from typing import ContextManager, List, Optional

import torch
import transformers
from hivemind.utils.logging import get_logger
from transformers.generation.utils import ModelOutput

from petals.client.inference_session import InferenceSession
from petals.client.remote_sequential import RemoteSequential
from petals.utils.misc import DUMMY, docstring_from

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RemotePastKeyValues:
    """A mock class representing the fact that `past_key_values` do exist but are stored on remote servers."""

    hypo_ids: Optional[torch.LongTensor] = None

    def __getitem__(self, _index: int) -> List[torch.Tensor]:
        return [DUMMY]  # For compatibility with BloomForCausalLM.prepare_inputs_for_generation()


_skipped_tokens = ContextVar("skipped_tokens", default=0)


class _SkipTokensMixin:
    # This override is used in RemoteGenerationMixin by has to be defined in a class not named as "GenerationMixin"
    # due to how transformers.PreTrainedModel.can_generate() works
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> dict:
        input_ids = input_ids[:, _skipped_tokens.get() :]
        _skipped_tokens.set(0)
        return super().prepare_inputs_for_generation(input_ids, **kwargs)

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)
    
def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(1, -1)
    return new_tokens, new_probs

def model_forward(model, x, input_pos):
    return model(x, input_pos)
def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])
            
class RemoteGenerationMixin(_SkipTokensMixin):
    """
    This class is an upgrade to `transformers.GenerationMixin` that:

    - Designed to be compatible with most `transformers.GenerationMixin` strategies and options
    - Supports generation inside a remote InferenceSession, so that remote servers store your attention caches and
      you don't have to rerun the prefix through all the servers to generate each new token
    - Supports multiple `.generate()` calls inside one InferenceSession, so you can easily run interactive generation
      by showing tokens on the fly (multiple calls like `.generate(None, max_new_tokens=1, ...)`) or
      accept prompts from a user in a chat bot (multiple calls like `.generate(new_prompts, ...)`).
    - If there is no active session, `.generate()` will create a new InferenceSession with proper `max_length`.
      Otherwise, `.generate()` will use the active session. You can use the `session=...` argument to override that.
    """

    @docstring_from(RemoteSequential.active_session)
    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self.transformer.h.active_session

    @docstring_from(RemoteSequential.use_session)
    def use_session(self, session: Optional[InferenceSession]) -> ContextManager[InferenceSession]:
        return self.transformer.h.use_session(session)

    @docstring_from(RemoteSequential.inference_session)
    def inference_session(self, **kwargs) -> ContextManager[InferenceSession]:
        return self.transformer.h.inference_session(**kwargs)

    @docstring_from(transformers.GenerationMixin.generate.__doc__)
    def generate(
        self, inputs: Optional[torch.Tensor] = None, *args, session: Optional[InferenceSession] = None, 
        draft_model = None, 
        speculate_k: Optional[int] = None,**kwargs
    ):
        self._fix_generate_kwargs(kwargs)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            session_max_length = self.transformer.config.pre_seq_len
            if max_length is not None:
                session_max_length += max_length
            else:
                session_max_length += (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the tokens from the previous .generate() call
            n_prev_tokens = session.output_ids.shape[1] if session.output_ids is not None else 0
            if n_prev_tokens > 0:
                if kwargs.get("num_beams", 1) > 1:
                    logger.warning(
                        "Beam search will not work properly in the resumed petals.InferenceSession "
                        "since intermediate beam entries are lost"
                    )

                if inputs is not None:
                    inputs = torch.cat([session.output_ids, inputs], dim=1)
                else:
                    inputs = session.output_ids

                # Don't actually run all previous tokens through the transformer,
                # but keep them for transformers.GenerationMixin (e.g., to compute repetition_penalty)
                _skipped_tokens.set(max(0, n_prev_tokens - 1))
            
            # Now integrate the speculative decoding logic
            is_speculative = draft_model is not None and speculate_k is not None
            generated_sequence = []

            if is_speculative:
                cur_token = inputs
                input_pos = 0
                desired_length = kwargs.get('max_length', 50)

                while input_pos < desired_length:
                    next_tokens = speculative_decode(
                        self, draft_model, cur_token, input_pos, speculate_k, **kwargs
                    )

                    generated_sequence.extend(next_tokens.tolist())
                    cur_token = next_tokens[-1].unsqueeze(0)
                    input_pos += len(next_tokens)

                    if cur_token.item() == self.config.eos_token_id:
                        break

                generated_sequence_tensor = torch.tensor(generated_sequence, dtype=inputs.dtype, device=inputs.device)
                result = generated_sequence_tensor.unsqueeze(0)
                return result
            else:
                result = super().generate(inputs, *args, **kwargs)

                sequences = result.sequences if isinstance(result, ModelOutput) else result
                # Save tokens from this .generate() call
                session.output_ids = sequences
                # Crop the last tokens from the previous call
                sequences = sequences[:, n_prev_tokens:].clone()
                if isinstance(result, ModelOutput):
                    result.sequences = sequences
                else:
                    result = sequences

            return result

    @staticmethod
    def _fix_generate_kwargs(kwargs: dict):
        # Suppress inappropriate "Both max_new_tokens and max_length" HF warning
        if "max_length" in kwargs and kwargs["max_length"] is None:
            del kwargs["max_length"]

        # Support do_sample = {0, 1} for backward compatibility with Petals < 2.1.0
        do_sample = kwargs.get("do_sample")
        if isinstance(do_sample, int):
            kwargs["do_sample"] = bool(do_sample)

    @staticmethod
    def _reorder_cache(past_key_values: RemotePastKeyValues, beam_idx: torch.LongTensor) -> RemotePastKeyValues:
        return dataclasses.replace(past_key_values, hypo_ids=beam_idx)
