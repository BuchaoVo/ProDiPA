"""Complete-sequence student supervision with weighted true terminal EOS.

The auxiliary term is not applied to sampled/censored rollout endpoints and
does not modify PoE targets. Residue targets constrain premature termination.
"""
import csv
import math

import torch
import torch.nn.functional as F

from .data import UnconditionalProDiPADataCollator
from .trainer import UnconditionalProDiPATrainer


class EOSAnchorCollator(UnconditionalProDiPADataCollator):
    def __init__(self, *args, anchor_path, **kwargs):
        super().__init__(*args, **kwargs)
        with open(anchor_path) as handle:
            rows = list(csv.DictReader(handle))
        sequences = [r['sequence'] for r in rows]
        if not sequences or len(set(sequences)) != len(sequences):
            raise ValueError('EOS anchors must be a nonempty deduplicated training set')
        self.anchors = []
        for seq in sequences:
            if not 50 <= len(seq) <= 512 or set(seq) - set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError('EOS anchors must be complete canonical 50-512aa sequences; no truncation')
            tokens = self.tokenizer.encode(seq, add_special_tokens=False)
            if not tokens or self.tokenizer.eos_token_id in tokens:
                raise ValueError('Unexpected EOS inside a protein')
            self.anchors.append([self.tokenizer.eos_token_id, *tokens])

    def __call__(self, features):
        batch = super().__call__(features)
        anchors = [self.anchors[int(r['sample_id']) % len(self.anchors)] for r in features]
        width = max(map(len, anchors))
        batch['anchor_input_ids'] = torch.tensor([r + [self.pad_token_id]*(width-len(r)) for r in anchors])
        batch['anchor_attention_mask'] = torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in anchors])
        return batch


def sequence_anchor_loss(logits, input_ids, attention_mask, eos_token_id, weight):
    """BOS predicts the first residue; each residue predicts the next or EOS.

    Inputs contain BOS + complete protein, with no terminal EOS. This is exactly
    shifted full-token CE on BOS + protein + EOS, with padding ignored by length
    (not token ID, since ProtGPT2 BOS/EOS/PAD can share the same ID).
    """
    if not math.isfinite(weight) or weight < 1:
        raise ValueError('EOS terminal weight must be finite and at least 1')
    if (logits.ndim != 3 or input_ids.ndim != 2 or input_ids.shape != attention_mask.shape
            or logits.shape[:2] != input_ids.shape or input_ids.shape[0] == 0):
        raise ValueError('Anchor logits, tokens and masks must have matching nonempty batch shapes')
    lengths = attention_mask.sum(-1).long()
    if (lengths < 2).any():
        raise ValueError('Each anchor must contain BOS and protein tokens')
    expected = torch.arange(attention_mask.shape[1], device=lengths.device)[None,:] < lengths[:,None]
    if not torch.equal(expected.to(attention_mask.dtype), attention_mask):
        raise ValueError('EOS anchors must be right-padded contiguous contexts')
    if (input_ids[:, 0] != eos_token_id).any() or ((input_ids[:, 1:] == eos_token_id) & expected[:, 1:]).any():
        raise ValueError('Anchors require BOS followed by protein tokens without embedded EOS')
    targets = torch.full_like(input_ids, -100)
    targets[:, :-1] = input_ids[:, 1:]
    targets.masked_fill_(~expected, -100)
    targets.scatter_(1, (lengths-1)[:, None], eos_token_id)
    terminal = torch.zeros_like(expected).scatter_(1, (lengths-1)[:, None], True)
    residue = expected & ~terminal
    token_ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-100, reduction='none').view_as(input_ids)
    weights = expected.to(token_ce.dtype) + (weight-1)*terminal.to(token_ce.dtype)
    loss = (token_ce*weights).sum()/weights.sum()
    with torch.no_grad():
        eos_probability = (logits[..., eos_token_id].float()-logits.float().logsumexp(-1)).exp()
        metrics = {'terminal_ce': token_ce[terminal].detach().mean(),
                   'residue_ce': token_ce[residue].detach().mean(),
                   'terminal_eos_probability': eos_probability[terminal].mean(),
                   'nonterminal_eos_probability': eos_probability[residue].mean(),
                   'start_eos_probability': eos_probability[:, 0].mean()}
    return loss, metrics


class EOSAnchoredUnconditionalTrainer(UnconditionalProDiPATrainer):
    def __init__(self, *args, eos_anchor_weight, eos_anchor_loss_weight, **kwargs):
        if not math.isfinite(eos_anchor_loss_weight) or eos_anchor_loss_weight <= 0:
            raise ValueError('Full-sequence auxiliary coefficient must be explicitly positive and finite')
        super().__init__(*args, **kwargs)
        self.eos_anchor_weight = eos_anchor_weight
        self.eos_anchor_loss_weight = eos_anchor_loss_weight
        # Both losses are already means; let Trainer divide by accumulation.
        self.model_accepts_loss_kwargs = False

    def _canonicalize_rollouts(self, generated_ids, prompt_ids):
        result = super()._canonicalize_rollouts(generated_ids, prompt_ids)
        completion = generated_ids[:, prompt_ids.shape[1]:]
        sequences = self._tokenizer.batch_decode(completion, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        lengths = [len(''.join(s.split())) for s in sequences]
        ended = (completion == self._tokenizer.eos_token_id).any(dim=1).tolist()
        canonical = [bool(s) and not (set(''.join(s.split()).upper())-set('ACDEFGHIKLMNPQRSTVWY')) for s in sequences]
        processor = getattr(self, '_length_processor', None)
        forced = processor.forced_eos.tolist() if processor is not None else [False]*len(lengths)
        valid = [ok and end and not capped and 50<=n<=512
                 for ok,end,capped,n in zip(canonical,ended,forced,lengths)]
        values = torch.tensor([sum(lengths),sum(n>512 for n in lengths),sum(ended),len(lengths),
                               sum(n==0 for n in lengths),sum(n<50 for n in lengths),sum(valid)],
                              dtype=torch.float32,device=generated_ids.device)
        values = self.accelerator.reduce(values,reduction='sum')
        self._buffer_train_metrics({'train/rollout_mean_aa':values[0]/values[3],
                                    'train/rollout_over_512_rate':values[1]/values[3],
                                    'train/rollout_eos_rate':values[2]/values[3],
                                    'train/rollout_empty_rate':values[4]/values[3],
                                    'train/rollout_under_50_rate':values[5]/values[3],
                                    'train/rollout_valid_complete_rate':values[6]/values[3]})
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        js = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        outputs = model(input_ids=inputs['anchor_input_ids'], attention_mask=inputs['anchor_attention_mask'], use_cache=False)
        anchor_ce, metrics = sequence_anchor_loss(outputs.logits, inputs['anchor_input_ids'],
            inputs['anchor_attention_mask'], self._tokenizer.eos_token_id, self.eos_anchor_weight)
        auxiliary = self.eos_anchor_loss_weight * anchor_ce
        loss = js + auxiliary
        self._buffer_train_metrics({'train/eos_anchor_loss': auxiliary.detach(),
                                    'train/anchor_weighted_ce': anchor_ce.detach(),
                                    'train/eos_anchor_ce': metrics['terminal_ce'],
                                    'train/anchor_residue_ce': metrics['residue_ce'],
                                    'train/eos_anchor_probability': metrics['terminal_eos_probability'],
                                    'train/anchor_nonterminal_eos_probability': metrics['nonterminal_eos_probability'],
                                    'train/anchor_start_eos_probability': metrics['start_eos_probability'],
                                    'train/total_loss': loss.detach()})
        return (loss, {'loss': loss.detach()}) if return_outputs else loss
