import pytest
import torch

from types import SimpleNamespace
import json

from prodipa.unconditional.eos_anchor import EOSAnchorCollator, EOSAnchoredUnconditionalTrainer, sequence_anchor_loss
from prodipa.unconditional.trainer import UnconditionalProDiPATrainer


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, sequence, add_special_tokens=False):
        return [1] * len(sequence)


def test_full_sequence_loss_matches_weighted_causal_ce_and_ignores_padding():
    logits = torch.randn(2, 5, 4, requires_grad=True)
    ids = torch.tensor([[0,1,2,0,0], [0,2,3,1,2]])
    attention = torch.tensor([[1,1,1,0,0], [1,1,1,1,1]])
    loss, metrics = sequence_anchor_loss(logits, ids, attention, 0, 4)
    targets = torch.tensor([[1,2,0,-100,-100], [2,3,1,2,0]])
    weights = torch.tensor([[1.,1.,4.,0.,0.], [1.,1.,1.,1.,4.]])
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1,4),targets.flatten(),ignore_index=-100,reduction='none').reshape(2,5)
    torch.testing.assert_close(loss,(ce*weights).sum()/weights.sum())
    torch.testing.assert_close(metrics['terminal_ce'], (ce[0,2]+ce[1,4])/2)
    loss.backward()
    assert logits.grad[0,3:].eq(0).all()
    # Gradient descent lowers EOS at BOS and every internal prefix, and raises
    # it only at the true endpoint. BOS and PAD share EOS's ID deliberately.
    assert (logits.grad[0,:2,0] > 0).all()
    assert (logits.grad[1,:4,0] > 0).all()
    assert logits.grad[0,2,0] < 0 and logits.grad[1,4,0] < 0


def test_weight_one_is_plain_full_token_ce_and_batch_duplication_is_invariant():
    ids=torch.tensor([[0,1,2]])
    logits=torch.randn(1,3,4)
    loss,_=sequence_anchor_loss(logits,ids,torch.ones_like(ids),0,1)
    expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.tensor([1,2,0]))
    torch.testing.assert_close(loss,expected)
    duplicate,_=sequence_anchor_loss(logits.repeat(2,1,1),ids.repeat(2,1),torch.ones(2,3,dtype=torch.long),0,1)
    torch.testing.assert_close(loss,duplicate)


def test_real_causal_model_matches_standard_shifted_full_sequence_loss():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(7)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=8,n_positions=16,n_embd=16,n_layer=1,n_head=2,
                                    resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)).eval()
    ids=torch.tensor([[0,1,2,0],[0,3,4,5]])
    mask=torch.tensor([[1,1,1,0],[1,1,1,1]])
    logits=model(input_ids=ids,attention_mask=mask).logits
    loss,_=sequence_anchor_loss(logits,ids,mask,0,1)
    full_ids=torch.tensor([[0,1,2,0,0],[0,3,4,5,0]])
    full_mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]])
    labels=full_ids.clone();labels[:,0]=-100;labels[0,4]=-100
    reference=model(input_ids=full_ids,attention_mask=full_mask,labels=labels).loss
    torch.testing.assert_close(loss,reference)


def test_single_gradient_step_reduces_premature_eos_and_increases_terminal_eos():
    ids=torch.tensor([[0,1,2,3]])
    logits=torch.zeros(1,4,4,requires_grad=True)
    with torch.no_grad(): logits[:,:,0]=5
    before=logits.detach().softmax(-1)[0,:,0]
    loss,_=sequence_anchor_loss(logits,ids,torch.ones_like(ids),0,4)
    loss.backward()
    after=(logits.detach()-.1*logits.grad).softmax(-1)[0,:,0]
    assert (after[:3]<before[:3]).all()
    assert after[3]>before[3]


def test_anchor_collator_keeps_prompt_and_pads_by_length(tmp_path):
    path=tmp_path/'train.csv'
    path.write_text('sequence\n'+'A'*50+'\n'+'C'*60+'\n')
    collator=EOSAnchorCollator(Tokenizer(),anchor_path=path)
    batch=collator([{'sample_id':0},{'sample_id':1}])
    assert batch['prompt_ids'].tolist()==[[0],[0]]
    assert batch['anchor_attention_mask'].sum(-1).tolist()==[51,61]
    assert batch['anchor_input_ids'][0,50].item()==1
    assert batch['anchor_input_ids'][0,51:].eq(0).all()


@pytest.mark.parametrize('sequence',['A'*513,'A'*49,'AX'+'C'*50])
def test_anchor_collator_rejects_invalid_or_overlength_without_truncation(tmp_path,sequence):
    path=tmp_path/'train.csv';path.write_text('sequence\n'+sequence+'\n')
    with pytest.raises(ValueError): EOSAnchorCollator(Tokenizer(),anchor_path=path)


def test_anchor_mask_must_be_contiguous_right_padding():
    with pytest.raises(ValueError):
        sequence_anchor_loss(torch.randn(1,3,4),torch.tensor([[0,1,2]]),torch.tensor([[1,0,1]]),0,4)


@pytest.mark.parametrize('weight',[0,.5,float('nan'),float('inf')])
def test_invalid_terminal_weights_rejected(weight):
    with pytest.raises(ValueError):
        sequence_anchor_loss(torch.zeros(1,2,4),torch.tensor([[0,1]]),torch.ones(1,2),0,weight)


def test_embedded_eos_and_nonbinary_masks_rejected():
    for ids,mask in [(torch.tensor([[0,0,1]]),torch.tensor([[1,1,1]])),
                     (torch.tensor([[0,1,2]]),torch.tensor([[1,2,0]]))]:
        with pytest.raises(ValueError): sequence_anchor_loss(torch.zeros(1,3,4),ids,mask,0,4)


def test_trainer_uses_explicit_auxiliary_coefficient_without_changing_js(monkeypatch):
    trainer=object.__new__(EOSAnchoredUnconditionalTrainer)
    trainer._tokenizer=Tokenizer()
    trainer.eos_anchor_weight=4
    trainer.eos_anchor_loss_weight=.02
    logged={}
    trainer._buffer_train_metrics=logged.update
    monkeypatch.setattr(UnconditionalProDiPATrainer,'compute_loss',lambda *a,**kw:torch.tensor(.1))
    logits=torch.randn(1,3,4,requires_grad=True)
    inputs={'anchor_input_ids':torch.tensor([[0,1,2]]),'anchor_attention_mask':torch.ones(1,3,dtype=torch.long)}
    model=lambda **kw:SimpleNamespace(logits=logits)
    expected,_=sequence_anchor_loss(logits,inputs['anchor_input_ids'],inputs['anchor_attention_mask'],0,4)
    loss,outputs=trainer.compute_loss(model,inputs,return_outputs=True)
    torch.testing.assert_close(loss,.1+.02*expected)
    torch.testing.assert_close(logged['train/eos_anchor_loss'],.02*expected)
    assert not outputs['loss'].requires_grad
    assert 'train/anchor_start_eos_probability' in logged


def test_rollout_metrics_distinguish_empty_short_valid_and_overlong(monkeypatch):
    trainer=object.__new__(EOSAnchoredUnconditionalTrainer)
    trainer._tokenizer=SimpleNamespace(eos_token_id=0,batch_decode=lambda *a,**kw:['','A'*49,'A'*50,'A'*513])
    trainer.accelerator=SimpleNamespace(reduce=lambda x,**kw:x)
    logged={}
    trainer._buffer_train_metrics=logged.update
    monkeypatch.setattr(UnconditionalProDiPATrainer,'_canonicalize_rollouts',lambda *a:'unchanged')
    assert trainer._canonicalize_rollouts(torch.tensor([[0,0],[0,0],[0,0],[0,0]]),torch.zeros(4,1,dtype=torch.long))=='unchanged'
    assert logged['train/rollout_empty_rate']==.25
    assert logged['train/rollout_under_50_rate']==.5
    assert logged['train/rollout_valid_complete_rate']==.25
    assert logged['train/rollout_over_512_rate']==.25
    trainer._length_processor=SimpleNamespace(forced_eos=torch.tensor([False,False,True,False]))
    trainer._canonicalize_rollouts(torch.zeros(4,2,dtype=torch.long),torch.zeros(4,1,dtype=torch.long))
    assert logged['train/rollout_valid_complete_rate']==0


def test_old_eos_only_resume_and_warm_start_are_rejected(tmp_path):
    from prodipa.unconditional.train import _validate_eos_anchor_state, _eos_supervision_manifest
    anchor=tmp_path/'anchors.csv';anchor.write_text('sequence\n'+'A'*50+'\n')
    output=tmp_path/'new';output.mkdir()
    old=tmp_path/'old';old.mkdir()
    args=SimpleNamespace(eos_anchor_path=str(anchor),eos_anchor_weight=4,eos_anchor_loss_weight=.02,
                         student_lora_adapter_path='')
    training=SimpleNamespace(output_dir=str(output),resume_from_checkpoint=None)
    _validate_eos_anchor_state(args,training)
    (old/'checkpoint-200').mkdir()
    training.resume_from_checkpoint=str(old/'checkpoint-200')
    with pytest.raises(ValueError,match='resume'): _validate_eos_anchor_state(args,training)
    training.resume_from_checkpoint=None
    args.student_lora_adapter_path=str(old)
    (old/'RUN_MANIFEST.json').write_text(json.dumps({'script_arguments':{'eos_anchor_path':'legacy.csv'}}))
    with pytest.raises(ValueError,match='EOS-only'): _validate_eos_anchor_state(args,training)
    args.student_lora_adapter_path=''
    (output/'EOS_SUPERVISION.json').write_text(json.dumps(_eos_supervision_manifest(args)))
    (output/'checkpoint-100').mkdir()
    training.resume_from_checkpoint=str(output/'checkpoint-100')
    _validate_eos_anchor_state(args,training)
    args.eos_anchor_loss_weight=.03
    with pytest.raises(ValueError,match='configuration changed'): _validate_eos_anchor_state(args,training)
