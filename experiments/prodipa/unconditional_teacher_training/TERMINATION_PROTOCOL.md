# ProtGPT2-compatible teacher termination protocol

## Scope

This protocol is for new Unconditional Prefix Teachers and their paired
multi-teacher student. It does not reinterpret or overwrite historical
checkpoints. The source CSV always stores a contiguous amino-acid sequence;
formatting is applied by the tokenizer pipeline, not by rewriting the data.

## Teacher input and target

`protgpt2_fasta60_v1` uses the format documented by the local ProtGPT2 model
card:

```text
<|endoftext|>\n
RESIDUES_1_TO_60\n
RESIDUES_61_TO_120\n
...\n
<|endoftext|>
```

In token form this is:

```text
[EOS] + tokenize("\n" + wrap(sequence, 60) + "\n") + [EOS]
```

The first EOS is the causal input/BOS. The leading newline, residues,
60-aa line breaks, trailing newline and one true terminal EOS are targets.
Right padding is masked. No sequence is truncated to fit the model context;
preflight fails instead.

The historical `plain_direct_eos_v1` path remains the default only for exact
reproduction of old teachers. A new run must explicitly select
`--sequence-protocol protgpt2_fasta60_v1` and use a new output directory.

## Student rollout and evaluation

- Use `sequence_format=legacy`; do not install the Unwrapped whitespace mask.
- Start generation from the native EOS/BOS token and preserve all sampled IDs.
- Student and all three teachers receive the identical sampled token context,
  including line breaks.
- `max_new_tokens=512` is a token-budget guard, not an amino-acid length limit.
- The pre-token 512-aa processor is a separate safety censor. It may force EOS
  before a token crosses 512 aa, but that terminal is not a natural completion.
- Natural EOS is supervised. AA-cap and token-budget terminals are masked from
  terminal loss while their preceding body tokens remain usable.
- Exported amino-acid text removes whitespace only after generation. The raw
  token IDs and raw completion remain the authoritative context.

Report direct EOS and newline-to-EOS separately. A generated sequence counts as
valid complete only if it has canonical residues, is 50–512 aa, and sampled a
natural EOS before either censoring mechanism.

## Required preflight evidence

The staged balanced package passes exact token round-trip checks. Across all
six 300/100 CSVs, the longest FASTA60 record is 185 tokens including BOS/EOS,
so none approaches the 512-token training limit. This establishes that teacher
EOS targets are not removed by tokenizer truncation.

Before any replacement training, also record by length stratum:

1. terminal EOS probability before and after the final newline;
2. whether EOS survives top-k=200 and top-p=0.9;
3. body, line-break and terminal-EOS loss separately;
4. natural EOS, AA-cap censor and token-budget censor counts.

An EOS-only auxiliary loss is not part of this protocol. Any later calibration
must be short, separately identified, and justified by these diagnostics.
