"""Unwrapped ProtGPT2 contexts, enforced before sampling, never by rewriting IDs."""
import json
from pathlib import Path

import torch
from transformers import LogitsProcessor
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

PROTOCOL_FILE = 'SEQUENCE_PROTOCOL.json'


def sequence_protocol(sequence_format):
    if sequence_format not in ('legacy', 'unwrapped'):
        raise ValueError('sequence_format must be legacy or unwrapped')
    return {'version': 1, 'sequence_format': sequence_format,
            'artificial_terminal_loss': 'masked_independently_of_aa_cap',
            'teacher_student_context': 'identical_sampled_token_ids',
            'loss_distribution': 'full_vocabulary_unconstrained'}


def resolve_sequence_format(adapter_path, requested='auto'):
    if requested not in ('auto', 'legacy', 'unwrapped'):
        raise ValueError('sequence_format must be auto, legacy or unwrapped')
    path = Path(adapter_path) / PROTOCOL_FILE if adapter_path else None
    recorded = json.loads(path.read_text())['sequence_format'] if path and path.is_file() else None
    if requested != 'auto' and recorded is not None and requested != recorded:
        raise ValueError('Generation sequence format disagrees with checkpoint protocol')
    resolved = recorded if requested == 'auto' and recorded else ('legacy' if requested == 'auto' else requested)
    sequence_protocol(resolved)
    return resolved


def whitespace_token_mask(tokenizer):
    """Include mixed residue/whitespace BPE tokens; exclude special-token spellings."""
    decoder = {v: k for k, v in bytes_to_unicode().items()}
    specials = set(tokenizer.all_special_ids)
    blocked = []
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        blocked.append(token_id not in specials and any(decoder.get(c) in b' \t\r\n\v\f'
                       for c in token if c in decoder))
    return torch.tensor(blocked, dtype=torch.bool)


class UnwrappedProteinLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.blocked = whitespace_token_mask(tokenizer)

    def __call__(self, input_ids, scores):
        self.blocked = self.blocked.to(scores.device)
        return scores.masked_fill(self.blocked[None, :], -float('inf'))
