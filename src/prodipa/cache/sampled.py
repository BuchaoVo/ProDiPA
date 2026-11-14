"""Full-support categorical sampling and finite-cache mixture cross entropy."""
from __future__ import annotations
import torch


def encode_sequence(tokenizer, prompt, sequence, max_length=768,
                    training_format="instruction_response"):
    if training_format == "native_protein":
        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Native protein tokenization requires BOS and EOS tokens")
        encoded = tokenizer(
            sequence,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        target_ids = list(encoded["input_ids"])
        spans = list(encoded["offset_mapping"])
        if not target_ids or len(target_ids) != len(spans):
            raise ValueError("Native protein tokenization produced no aligned BPE tokens")
        if tokenizer.decode(
            target_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) != sequence:
            raise ValueError("Native BPE tokens do not exactly reconstruct the protein")
        if spans[0][0] != 0 or spans[-1][1] != len(sequence):
            raise ValueError("Native BPE offsets do not cover the complete protein")
        if any(left >= right or (i and left != spans[i - 1][1])
               for i, (left, right) in enumerate(spans)):
            raise ValueError("Native BPE offsets are not contiguous")
        source = [tokenizer.bos_token_id]
        target = target_ids + [tokenizer.eos_token_id]
        ids = source + target
        if len(ids) > max_length:
            raise ValueError(f"Would truncate {len(ids)} tokens at {max_length}")
        # Position i predicts the corresponding BPE token at i+1.  EOS remains
        # in causal CE but not in the property-teacher soft target positions.
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor([-100] * len(source) + target),
            "positions": torch.arange(len(target_ids), dtype=torch.long),
            "residue_spans": torch.tensor(spans, dtype=torch.long),
            "target_start": len(source),
            "target_ids": torch.tensor(target),
        }
    if training_format != "instruction_response":
        raise ValueError(f"Unknown training format: {training_format}")
    source = tokenizer(prompt, add_special_tokens=True)['input_ids']
    text = f'Seq=<{sequence}>'
    proto = tokenizer.sp_model.encode(text, out_type='immutable_proto')
    pieces = list(proto.pieces)
    target = tokenizer(text + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    if target != [p.id for p in pieces] + [tokenizer.eos_token_id]:
        raise ValueError('SentencePiece offsets and tokenizer IDs disagree')
    # Pure residue tokens only; tokens crossing a wrapper boundary stay in grammar CE.
    body = [i for i, p in enumerate(pieces) if 5 <= p.begin < p.end <= 5+len(sequence)]
    spans = [(max(0, p.begin-5), min(len(sequence), p.end-5)) for p in pieces]
    if ''.join(text[pieces[i].begin:pieces[i].end] for i in body) != sequence:
        raise ValueError('Residue spans do not exactly reconstruct the protein')
    ids = source + target
    if len(ids) > max_length:
        raise ValueError(f'Would truncate {len(ids)} tokens at {max_length}')
    return {'input_ids': torch.tensor(ids), 'labels': torch.tensor([-100]*len(source)+target),
            'positions': torch.tensor([len(source)+i-1 for i in body]),
            'residue_spans': torch.tensor([spans[i] for i in body]),
            'target_start': len(source), 'target_ids': torch.tensor(target)}


def draw_tokens(log_probs, count, generator):
    if count < 1 or not torch.isfinite(log_probs).all():
        raise ValueError('Nonfinite teacher distribution or invalid sample count')
    return torch.multinomial(log_probs.exp(), count, replacement=True, generator=generator)


def mixture_weights(preference, rho, device=None):
    w = torch.as_tensor(preference, dtype=torch.float32, device=device)
    if not 0 <= rho <= 1 or w.shape != (3,) or (w < 0).any() or abs(float(w.sum())-1) > 1e-5:
        raise ValueError('Invalid mixture')
    return torch.cat((w.new_tensor([1-rho]), rho*w))


def cached_cross_entropy(student_logp, targets, weights, mode='sampled'):
    """Average all cached draws (Rao-Blackwellize the teacher-index draw).

    This is exactly CE of the finite empirical cache mixture, not exact CE of
    the underlying teacher mixture conditional on a realized cache.
    """
    terms = []
    for record in targets:
        if mode == 'sampled':
            term = -student_logp.gather(-1, record['sample_ids'].to(student_logp.device).long()).mean()
        elif mode == 'hard':
            term = -student_logp.gather(-1, record['hard_ids'].to(student_logp.device).long()[:, None]).mean()
        elif mode == 'topk':
            ids = record['top_ids'].to(student_logp.device).long()
            probability = record['top_logp'].to(student_logp.device).float().softmax(-1)
            term = -(student_logp.gather(-1, ids) * probability).sum(-1).mean()
        else:
            raise ValueError(mode)
        terms.append(term)
    return (torch.stack(terms) * weights.to(student_logp)).sum()
