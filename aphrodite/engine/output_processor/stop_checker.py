from typing import Callable, Optional, List, Dict, Tuple

from transformers import PreTrainedTokenizer

from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import Sequence, SequenceStatus
from aphrodite.lora.request import LoRARequest


class StopChecker:
    """AphroditeEngine helper class which separates out the logic involving
    stop checking. This checks things such as: whether the eos token was
    emitted, whether the max_tokens has been consumed, whether a stop string
    has been emitted, or if we have exceeded the max model len.
    """

    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence],
                                                 PreTrainedTokenizer]):
        # Do not use it directly, but use `self._get_max_model_len`.
        self._max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self._sequence_buffers: Dict[int, List[int]] = {}  # seq_id -> buffered_tokens
        self._active_banned_patterns: Dict[int, List[List[int]]] = {}  # seq_id -> potential banned patterns
        self._rollback_length: Dict[int, int] = {}  # seq_id -> num tokens to rollback

    def _get_max_model_len(self, lora_req: Optional[LoRARequest]):
        if lora_req and lora_req.long_lora_max_len:
            return lora_req.long_lora_max_len
        else:
            return self._max_model_len

    def _check_banned_sequences(
        self,
        seq: Sequence,
        token_id: int,
        sampling_params: SamplingParams,
    ) -> Tuple[bool, Optional[List[int]], bool, int]:
        """Check if the token continues a banned sequence.
        
        Returns:
            Tuple[should_buffer: bool, tokens_to_release: Optional[List[int]], 
                  should_ban: bool, rollback_length: int]
        """
        seq_id = seq.seq_id
        buffer = self._sequence_buffers.get(seq_id, [])
        active_patterns = self._active_banned_patterns.get(seq_id, [])
        
        # Check if this token starts any banned sequences
        if not buffer:
            matching_patterns = [
                pattern for pattern in sampling_params.banned_strings
                if pattern[0] == token_id
            ]
            if matching_patterns:
                # Only ban if any pattern is single token
                should_ban = any(len(pattern) == 1 for pattern in matching_patterns)
                # Buffer if any pattern is longer
                should_buffer = any(len(pattern) > 1 for pattern in matching_patterns)
                
                if should_buffer:
                    self._sequence_buffers[seq_id] = [token_id]
                    self._active_banned_patterns[seq_id] = [p for p in matching_patterns if len(p) > 1]
                    seq.status = SequenceStatus.BUFFERING
                
                return should_buffer, None, should_ban, 1 if should_ban else 0

        # Check existing buffer
        if buffer:
            buffer.append(token_id)
            next_idx = len(buffer) - 1
            
            # Update active patterns
            still_active = []
            for pattern in active_patterns:
                if len(pattern) > next_idx and pattern[next_idx] == token_id:
                    if len(pattern) == len(buffer):
                        # Found complete banned sequence - clear buffers and rollback
                        del self._sequence_buffers[seq_id]
                        del self._active_banned_patterns[seq_id]
                        seq.status = SequenceStatus.RUNNING
                        return True, None, True, len(buffer)
                    still_active.append(pattern)
            
            if still_active:
                self._active_banned_patterns[seq_id] = still_active
                return True, None, False, 0
            else:
                # No patterns match anymore - release buffer
                tokens = self._sequence_buffers.pop(seq_id)
                self._active_banned_patterns.pop(seq_id)
                seq.status = SequenceStatus.RUNNING
                return False, tokens, False, 0

        return False, None, False, 0

    def maybe_stop_sequence(
        self,
        seq: Sequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
    ) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.get_last_token_id()
        if last_token_id in sampling_params.stop_token_ids:
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if any stop strings are matched.
        stop_str = self._check_stop_strings(seq, new_char_count,
                                            sampling_params)
        if stop_str is not None:
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self._get_max_model_len(lora_req):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check banned strings if any exist
        if sampling_params.banned_strings:
            should_buffer, tokens_to_release, should_ban, rollback_len = self._check_banned_sequences(
                seq, seq.get_last_token_id(), sampling_params)
            
            if should_ban:
                # Remove the banned sequence and roll back
                if new_char_count:
                    seq.output_text = seq.output_text[:-new_char_count]
                # Set status to WAITING to trigger rescheduling
                seq.status = SequenceStatus.WAITING
                # Reset sequence state for recomputation
                seq.data.reset_state_for_recompute()
                return None, rollback_len

            if should_buffer:
                return [], 0  # Signal to skip this token
                
            if tokens_to_release:
                return tokens_to_release, 0  # Release buffered tokens

        return None, 0

    @staticmethod
    def _check_stop_strings(seq: Sequence, new_char_count: int,
                            sampling_params: SamplingParams) -> Optional[str]:
        """Check if any stop strings are matched and truncate sequence
        output text accordingly.

        Returns the stop string if matched or else None.
        """
        if not new_char_count:
            return None

        for stop_str in sampling_params.stop:
            stop_string_len = len(stop_str)
            # Avoid searching already-searched text.
            stop_index = seq.output_text.find(
                stop_str, -new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if sampling_params.include_stop_str_in_output:
                # Truncate to end of stop string.
                stop_index += stop_string_len
                if stop_index >= len(seq.output_text):
                    # No truncation required.
                    return stop_str

            # Truncate the output text to either the beginning
            # or end of the stop string.
            seq.output_text = seq.output_text[:stop_index]
            return stop_str
        return None
