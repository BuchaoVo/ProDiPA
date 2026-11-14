"""CPU protocol regressions; no pretrained model loading or training."""
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from prodipa.unconditional.sequence_format import (
    PROTOCOL_FILE, UnwrappedProteinLogitsProcessor, resolve_sequence_format, sequence_protocol,
)
from prodipa.unconditional.trainer import UnconditionalProDiPATrainer


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained('__PRODIPA_MODELS__/ProtGPT2', local_files_only=True)


def test_sampler_blocks_all_whitespace_but_preserves_residues_and_eos(tokenizer):
    proc = UnwrappedProteinLogitsProcessor(tokenizer)
    scores = torch.zeros(2, len(tokenizer))
    result = proc(torch.tensor([[0], [0]]), scores)
    for text in ('\n', '\r', ' ', '\t', 'A\nC'):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert any(proc.blocked[ids])
    assert not proc.blocked[tokenizer.eos_token_id]
    assert not proc.blocked[tokenizer.encode('ACDEFGHIKLMNPQRSTVWY', add_special_tokens=False)].any()
    assert torch.isneginf(result[:, proc.blocked]).all()
    assert torch.equal(result[:, ~proc.blocked], scores[:, ~proc.blocked])
    # Byte-level mixed residue + whitespace tokens must be covered too, if present.
    for i in range(len(tokenizer)):
        if i not in tokenizer.all_special_ids:
            decoded = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            assert proc.blocked[i].item() == any(c in ' \t\r\n\v\f' for c in decoded)


def make_trainer(tokenizer, fmt='unwrapped'):
    tr = object.__new__(UnconditionalProDiPATrainer)
    tr._tokenizer = tokenizer
    tr.sequence_format = fmt
    tr._format_processor = UnwrappedProteinLogitsProcessor(tokenizer)
    tr._length_processor = None
    tr.max_length_aa = 0
    tr._train_metric_buffer = {}
    tr.accelerator = SimpleNamespace(reduce=lambda x, reduction: x)
    return tr


def test_censored_terminal_mask_is_independent_and_ids_are_not_retokenized(tokenizer):
    tr = make_trainer(tokenizer)
    eos = tokenizer.eos_token_id
    a = tokenizer.encode('A', add_special_tokens=False)[0]
    raw = torch.tensor([[eos,a,eos,eos], [eos,a,a,a], [eos,eos,eos,eos]])
    ids, mask, labels, _ = tr._canonicalize_rollouts(raw, raw[:, :1])
    assert ids[1].tolist() == [eos,a,a,a,eos]  # No BPE merge/re-encoding of AAA.
    assert labels[1].tolist() == [-100,a,a,a,-100]
    assert labels[0].tolist() == [-100,a,eos,-100,-100]
    assert mask[0].tolist() == [1,1,1,0,0]
    assert labels[2].tolist() == [-100,eos,-100,-100,-100]  # Natural empty response, not hidden.
    metrics = tr._flush_buffered_train_metrics()
    assert metrics['train/end_natural_count'] == 2
    assert metrics['train/end_token_budget_count'] == 1


def test_whitespace_is_rejected_not_silently_deleted(tokenizer):
    tr = make_trainer(tokenizer)
    raw = torch.tensor([[0, *tokenizer.encode('A\nC', add_special_tokens=False), 0]])
    with pytest.raises(ValueError, match='refusing to rewrite'):
        tr._canonicalize_rollouts(raw, raw[:, :1])
    tr.sequence_format = 'legacy'
    ids, _, _, _ = tr._canonicalize_rollouts(raw, raw[:, :1])
    assert torch.equal(ids, raw)


def test_position_metrics_exclude_padding_and_artificial_eos():
    tr = object.__new__(UnconditionalProDiPATrainer)
    tr._tokenizer = SimpleNamespace(eos_token_id=0)
    tr._format_processor = SimpleNamespace(blocked=torch.tensor([False,False,True,False]))
    tr.teacher_specs = [SimpleNamespace(name='fold'),SimpleNamespace(name='sol'),SimpleNamespace(name='thermo')]
    tr.accelerator = SimpleNamespace(reduce=lambda x, reduction: x)
    tr._train_metric_buffer = {}
    s = torch.tensor([[[.1,.7,.1,.1]]*5]).log()
    t = torch.tensor([[[.4,.4,.1,.1]]*5]).log()
    labels = torch.tensor([[1,2,0,-100,-100]])
    sampled = torch.tensor([[1,2,0,0,0]])
    losses = torch.tensor([[1.,2.,3.,99.,99.]])
    tr._log_position_metrics(losses,s,[t,t,t],t,labels,sampled)
    m = tr._flush_buffered_train_metrics()
    for region, loss in [('body',1),('whitespace',2),('natural_eos',3)]:
        prefix = 'train/position_'+region
        assert m[prefix+'_tokens'] == 1
        assert m[prefix+'_js_loss'] == loss
        assert m[prefix+'_student_eos_prob'] == pytest.approx(.1)
        assert m[prefix+'_poe_eos_prob'] == pytest.approx(.4)
    labels[0,1] = -100
    tr._log_position_metrics(losses,s,[t,t,t],t,labels,sampled)
    m = tr._flush_buffered_train_metrics()
    assert m['train/position_whitespace_tokens'] == 0
    assert 'train/position_whitespace_js_loss' not in m


def test_token_js_reduction_preserves_loss_and_gradient():
    torch.manual_seed(42)
    logits = torch.randn(2,4,6,requires_grad=True)
    s = logits.log_softmax(-1)
    t = torch.randn_like(logits).log_softmax(-1)
    labels = torch.tensor([[1,2,0,-100],[2,3,4,-100]])
    fn = UnconditionalProDiPATrainer._generalized_jsd_loss_from_log_probs
    for beta in (0., .5, 1.):
        full = fn(s,t,labels,beta)
        per = fn(s,t,beta=beta,reduction='none')
        torch.testing.assert_close(full,per[labels!=-100].mean())
    empty = fn(s,t,torch.full_like(labels,-100))
    empty.backward()
    assert logits.grad.eq(0).all()


def test_generation_metadata_is_backward_compatible_and_fail_closed(tmp_path):
    assert resolve_sequence_format(tmp_path) == 'legacy'
    assert resolve_sequence_format(tmp_path,'unwrapped') == 'unwrapped'
    (tmp_path/PROTOCOL_FILE).write_text(json.dumps(sequence_protocol('unwrapped')))
    assert resolve_sequence_format(tmp_path) == 'unwrapped'
    with pytest.raises(ValueError,match='disagrees'):
        resolve_sequence_format(tmp_path,'legacy')
    with pytest.raises(ValueError):
        resolve_sequence_format(tmp_path,'unknown')


@pytest.mark.parametrize('top_k',[None,3])
def test_js_monitoring_preserves_objective_gradient_and_uses_full_eos(top_k):
    tr = object.__new__(UnconditionalProDiPATrainer)
    tr._tokenizer = SimpleNamespace(eos_token_id=0)
    tr._format_processor = SimpleNamespace(blocked=torch.tensor([False,False,True,False,False,False]))
    tr.teacher_specs = [SimpleNamespace(name=n,temperature=1.) for n in ('fold','sol','thermo')]
    tr.accelerator = SimpleNamespace(reduce=lambda x,reduction: x)
    tr._train_metric_buffer = {}
    tr.top_k_loss = top_k
    tr.distill_beta = .5
    tr.use_entropy_rule = False
    tr._compute_dynamic_teacher_weights = lambda ts,s,ids,labels: (torch.full((*ids.shape,3),1/3),{})
    torch.manual_seed(3)
    student = torch.randn(1,4,6,requires_grad=True)
    teachers = [torch.randn_like(student) for _ in range(3)]
    labels = torch.tensor([[1,2,0,-100]])
    sampled = torch.tensor([[1,2,0,0]])
    loss, _, _, _ = tr._compute_jsd_loss(student,teachers,labels,sampled,1.)
    poe = torch.stack([t.log_softmax(-1) for t in teachers]).mean(0)
    slogits = student
    if top_k:
        indices = poe.topk(top_k,dim=-1).indices
        slogits = slogits.gather(-1,indices)
        poe = poe.gather(-1,indices)
    reference = tr._generalized_jsd_loss_from_log_probs(slogits.log_softmax(-1),poe.log_softmax(-1),labels)
    torch.testing.assert_close(loss,reference)
    a = torch.autograd.grad(loss,student,retain_graph=True)[0]
    b = torch.autograd.grad(reference,student)[0]
    torch.testing.assert_close(a,b)
    m = tr._flush_buffered_train_metrics()
    assert m['train/position_natural_eos_student_eos_prob'] == pytest.approx(student.detach().softmax(-1)[0,2,0].item())


def test_rollout_sampler_enforces_format_before_forward_and_restores_model_mode(tokenizer):
    tr = make_trainer(tokenizer)
    eos = tokenizer.eos_token_id
    a = tokenizer.encode('A',add_special_tokens=False)[0]
    newline = tokenizer.encode('\n',add_special_tokens=False)[0]

    class Model:
        training = True
        config = SimpleNamespace(use_cache=False)
        def eval(self): self.training = False
        def train(self): self.training = True
        def generate(self,input_ids,attention_mask,**kwargs):
            assert not self.training and self.config.use_cache
            scores = torch.full((1,len(tokenizer)),-100.)
            scores[0,newline] = 100.  # Must be blocked BEFORE choosing a token.
            scores[0,a] = 1.
            for processor in kwargs['logits_processor']:
                scores = processor(input_ids,scores)
            next_id = scores.argmax(-1,keepdim=True)
            assert next_id.item() == a
            return torch.cat([input_ids,next_id,torch.tensor([[eos]])],dim=1)

    model = Model()
    tr.accelerator.unwrap_model = lambda m: m
    tr.max_new_tokens = 4
    tr.gen_repetition_penalty = 1.2
    tr.gen_top_p = .95
    tr.gen_top_k = 500
    prompt = torch.tensor([[eos]])
    raw = tr._generate_on_policy(model,prompt,torch.ones_like(prompt),1.)
    ids,_,labels,_ = tr._canonicalize_rollouts(raw,prompt)
    assert ids.tolist() == [[eos,a,eos]]
    assert labels.tolist() == [[-100,a,eos]]
    assert model.training and not model.config.use_cache


def test_real_tiny_lora_prefix_rollout_backward_and_checkpoint_protocol(tokenizer, tmp_path):
    """Exercise actual Trainer/PEFT/Generate paths on CPU, without an optimizer update."""
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    from peft import LoraConfig, PrefixTuningConfig, TaskType, get_peft_model
    from prodipa.unconditional.eos_anchor import EOSAnchorCollator, EOSAnchoredUnconditionalTrainer
    from prodipa.unconditional.trainer import TeacherAdapterSpec
    torch.set_num_threads(2)
    torch.manual_seed(7)
    tokenizer.pad_token = tokenizer.eos_token
    cfg = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=16, n_layer=1,
                     n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    student = get_peft_model(GPT2LMHeadModel(cfg), LoraConfig(task_type=TaskType.CAUSAL_LM,
                            r=2,lora_alpha=4,target_modules=['c_attn']))
    pc = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,num_virtual_tokens=2)
    teacher = get_peft_model(GPT2LMHeadModel(cfg), pc, adapter_name='fold')
    for name in ('sol','thermo'):
        teacher.add_adapter(name, PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,num_virtual_tokens=2))
    anchor = tmp_path/'anchors.csv'
    anchor.write_text('sequence\n'+'A'*50+'\n'+'C'*60+'\n')
    collator = EOSAnchorCollator(tokenizer,anchor_path=anchor)
    trainer = EOSAnchoredUnconditionalTrainer(model=student,teacher_model=teacher,
        teacher_specs=[TeacherAdapterSpec(name=n,adapter_name=n,weight=1/3,temperature=1.) for n in ('fold','sol','thermo')],
        tokenizer=tokenizer, args=TrainingArguments(output_dir=str(tmp_path/'model'),use_cpu=True,
            report_to=[],disable_tqdm=True), eos_anchor_weight=4,eos_anchor_loss_weight=.01,
        max_new_tokens=4,sequence_format='unwrapped',save_generations=False)
    contexts = []
    hooks = [m.register_forward_pre_hook(lambda m,args,kwargs: contexts.append(kwargs['input_ids'].clone()),
              with_kwargs=True) for m in (student,teacher)]
    loss = trainer.compute_loss(student,collator([{'sample_id':0},{'sample_id':1}]))
    for hook in hooks: hook.remove()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters() if p.requires_grad)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    # Last five forwards: student rollout, three teacher adapters, student anchors.
    assert all(torch.equal(contexts[-5], c) for c in contexts[-4:-1])
    observed = contexts[-5]
    assert not trainer._format_processor.blocked[observed].any()
    metrics = trainer._flush_buffered_train_metrics()
    assert metrics['train/position_whitespace_tokens'] == 0
    assert metrics['train/end_natural_count']+metrics['train/end_token_budget_count'] == 2
    trainer.save_model()
    assert resolve_sequence_format(tmp_path/'model') == 'unwrapped'
    assert (tmp_path/'model/adapter_model.safetensors').is_file()
