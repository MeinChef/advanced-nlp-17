import os
import pickle
import torch
from pathlib import Path
from contextlib import nullcontext
from nanoGPT.model import GPTConfig, GPT

class TextGenerator:
    def __init__(
            self, 
            out_dir,
            task: str = "task1",
            device: str = 'cuda', 
            dtype: str = 'bfloat16'
        ) -> None:
        self.device = device
        self.out_dir = out_dir
        
        # Load checkpoint
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        # Remove unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        
        # Load encoding
        meta_path = os.path.join(
            Path.cwd(),
            "nanoGPT",
            'data', 
            checkpoint['config']['dataset'],
            # f"shakespeare_{task}",
            'meta.pkl'
        ) if 'config' in checkpoint else None
        
        if meta_path and os.path.exists(meta_path):

            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.encode = lambda s: [meta['stoi'][c] for c in s]
            self.decode = lambda l: ''.join([meta['itos'][i] for i in l])
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)
        
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    def generate(
            self, 
            start_text: str, 
            max_new_tokens: int = 30,
            temperature: float = 0.8, 
            top_k: int = 200
        ) -> str:
        start_ids = self.encode(start_text)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        return self.decode(y[0].tolist())