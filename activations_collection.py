import datasets
from torch.utils.data import DataLoader
from llama import Llama
import torch
import fire
from tqdm import tqdm

import llamascope


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int = 256,
        max_batch_size: int = 16,
):
    # Load model
    print('--- Loading model ---')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Tokenisation
    def tokenise_and_pad(x, seq_len=max_seq_len):
        """Tokenise and pad/crop to seq_len tokens."""
        # Tokenise the whole input
        toks = generator.tokenizer.encode(x['text'], bos=True, eos=False)
        toks = torch.tensor(toks, dtype=torch.long, device='cuda')

        # Pad and crop to correct sequence length
        if len(toks) == seq_len:
            pass
        
        elif len(toks) > seq_len:
            toks = toks[:seq_len]
        
        else:
            pad_id = generator.tokenizer.pad_id
            pad_toks = torch.full((seq_len,), pad_id).to(toks)
            pad_toks[:len(toks)] = toks
            toks = pad_toks
        
        return {'tokens': toks, 'text': x['text']}

    # Initialise streaming dataset
    print('\n--- Initialising streaming dataset ---')
    dataset = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True
        ).map(tokenise_and_pad)
    
    # Build dataloader
    dataloader = DataLoader(dataset, batch_size=max_batch_size)

    # Attach instrumentation
    scope = llamascope.LlamaScope(generator.model)
    location = 'layers-16'
    scope.add_caching_hook(location)

    # Inference loop
    write_every = int(1e2)
    num_written = 0
    tokens_cache = []
    for i, data in tqdm(enumerate(dataloader)):
        toks = data['tokens']
        _ = generator.model(toks, start_pos=0)
        tokens_cache.append(toks.cpu())

        # Write activations and clear cache every write_every batches
        if i % write_every == 0 and i > 0:
            all_acts = torch.cat(scope.activations_cache[location])
            all_toks = torch.cat(toks)
            torch.save(
                {'activations': all_acts, 'tokens': all_toks},
                f'activations/{location}-{num_written}.pt'
                 )
            num_written += 1
            scope.clear_all_caches()
            tokens_cache = []


if __name__ == '__main__':
    fire.Fire(main)