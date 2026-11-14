import torch
from tgnn.config import get_config,CN
from tgnn.model import build_model
from tgnn.tokenizer import build_tokenizer


def estimate_therm(seqs, max_len = 1024):
    model_path = "estimation_checkpoints/ts_thermostablility_esm650m_regressor_new.pth"
    state = torch.load(model_path, map_location="cpu")
    cfg = get_config()
    cfg.update(CN(state["config"]))
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg).eval()
    model.load_state_dict(state["model"])
    pad_id = tokenizer.pad_id

    tems = []
    for seq in seqs:
        input_ids = tokenizer.encode(seq, bos=False, eos=False, device='cuda')
        # input_ids = tokenizer.encode(seq, bos=True, eos=True, device='cuda')
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype, device='cuda')))
        elif len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        input_ids = input_ids.to(dtype=torch.long, device="cuda")[None]
        output = model(input_ids)[0]
        output =  80 * torch.sigmoid(output) + 20
        tems.append(output.item())
    return(tems)


def estimate_sol(seqs, max_len = 512):
    model_path = "estimation_checkpoints/solubility_esm650m_classifier.pth"
    cfg = get_config()
    state = torch.load(model_path, map_location="cpu")
    cfg.update(CN(state["config"]))
    max_len = cfg.seq_len
    
    model = build_model(cfg).eval()
    model.load_state_dict(state["model"])
    model = model.cuda()
    
    tokenizer = build_tokenizer(cfg)    
    pad_id = tokenizer.pad_id
    output_list = list()
    for seq in seqs:
        try:
            input_ids = tokenizer.encode(seq, bos=True, eos=True, device="cuda")
        except:
            print(f'Invalid seq: {seq}')
            output_list.append(0.0)
            continue

        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype, device="cuda")))
        elif len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        input_ids = input_ids.to(dtype=torch.long, device="cuda")[None]
        
        output = model(input_ids)
        output = output.softmax(dim=-1)[0, 1].item()
        output_list.append(output)
    
    return output_list


