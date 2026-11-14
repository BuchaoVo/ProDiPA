"""Explicit sequence/EOS protocols for ProtGPT2 Prefix teachers.

The pretrained ProtGPT2 corpus places a line break before the first residue,
wraps FASTA sequence text every 60 residues, and places another line break
before the closing ``<|endoftext|>`` token.  New teacher runs can reproduce
that token context with :data:`PROTGPT2_FASTA60_V1`.  The historical direct-EOS
protocol remains available only so existing checkpoints and configs retain an
unambiguous identity.
"""

from __future__ import annotations

import textwrap


PLAIN_DIRECT_EOS_V1 = "plain_direct_eos_v1"
PROTGPT2_FASTA60_V1 = "protgpt2_fasta60_v1"
SUPPORTED_PROTOCOLS = (PLAIN_DIRECT_EOS_V1, PROTGPT2_FASTA60_V1)


def format_teacher_sequence(sequence: str, protocol: str) -> str:
    """Return the exact non-special-token text seen by the teacher.

    The function never truncates and never fabricates an EOS token.  EOS/BOS
    are added separately by :func:`encode_teacher_sequence`, which makes the
    distinction between textual line breaks and model special tokens explicit.
    """

    if protocol == PLAIN_DIRECT_EOS_V1:
        return sequence
    if protocol == PROTGPT2_FASTA60_V1:
        return "\n" + "\n".join(textwrap.wrap(sequence, width=60)) + "\n"
    raise ValueError(
        f"Unknown teacher sequence protocol {protocol!r}; expected one of "
        f"{SUPPORTED_PROTOCOLS}"
    )


def encode_teacher_sequence(tokenizer, sequence: str, protocol: str) -> list[int]:
    """Encode one complete protein with exactly one true closing EOS target."""

    if tokenizer.eos_token_id is None:
        raise ValueError("ProtGPT2 tokenizer must define eos_token_id")
    text = format_teacher_sequence(sequence, protocol)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.eos_token_id, *token_ids, tokenizer.eos_token_id]


def protocol_metadata(protocol: str) -> dict[str, object]:
    """Machine-readable semantics stored with a new teacher checkpoint."""

    format_teacher_sequence("AC", protocol)  # fail closed on unknown values
    return {
        "version": 1,
        "name": protocol,
        "source_sequence_complete": True,
        "artificial_terminal_supervision": False,
        "true_eos_target_count_per_sequence": 1,
        "line_width_aa": 60 if protocol == PROTGPT2_FASTA60_V1 else None,
        "leading_newline": protocol == PROTGPT2_FASTA60_V1,
        "trailing_newline_before_eos": protocol == PROTGPT2_FASTA60_V1,
        "generation_compatibility": (
            "legacy_whitespace_sampling_with_pre_token_aa_safety_cap"
            if protocol == PROTGPT2_FASTA60_V1
            else "historical_direct_eos_only"
        ),
    }
