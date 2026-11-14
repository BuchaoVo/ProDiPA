"""Opt-in decoded-length cap for ProtGPT2 byte-level BPE generation.

Count non-whitespace bytes conservatively (exactly residues for canonical ASCII
proteins). Mask a token BEFORE it can cross the cap; never slice a protein.
An EOS forced at the boundary is recorded separately from sampled natural EOS.
"""
import torch
from transformers import LogitsProcessor
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class ProteinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_length_aa: int, prompt_length: int = 1):
        if max_length_aa <= 0 or tokenizer.eos_token_id is None:
            raise ValueError('A positive residue budget and an EOS token are required')
        self.limit = int(max_length_aa)
        self.prompt_length = int(prompt_length)
        self.eos = tokenizer.eos_token_id
        decoder = {v: k for k, v in bytes_to_unicode().items()}
        specials = set(tokenizer.all_special_ids)
        lengths = []
        for i in range(len(tokenizer)):
            token = tokenizer.convert_ids_to_tokens(i)
            if i in specials:
                lengths.append(0)
            else:
                try:
                    lengths.append(sum(decoder[c] not in b' \t\r\n\v\f' for c in token))
                except KeyError as error:
                    raise ValueError('Length control requires a GPT2 byte-level BPE tokenizer') from error
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.forced_eos = None

    def __call__(self, input_ids, scores):
        self.lengths = self.lengths.to(scores.device)
        completion = input_ids[:, self.prompt_length:]
        used = self.lengths[completion].sum(dim=1)
        finished = (completion == self.eos).any(dim=1)
        boundary = (used >= self.limit) & ~finished
        if self.forced_eos is None:
            self.forced_eos = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=scores.device)
        self.forced_eos |= boundary
        scores = scores.masked_fill(self.lengths[None, :] > (self.limit-used)[:, None], -float('inf'))
        stop = boundary | finished
        scores[stop] = -float('inf')
        scores[stop, self.eos] = 0
        return scores

    def forced_flags(self):
        return [] if self.forced_eos is None else self.forced_eos.detach().cpu().tolist()
