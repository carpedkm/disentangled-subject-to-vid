#!/usr/bin/env python3
"""
stress_gpu.py  ―  saturate BOTH memory *and* compute on every visible GPU
Tested on 2× RTX A6000 (48 GB) with PyTorch ≥ 2.1

• Fills ≈ 98 % of the free VRAM with a dummy tensor
• Keeps the compute-utilisation pegged at ≈ 100 % via endless matmuls
  (press Ctrl-C to stop)

Run:  python stress_gpu.py            # defaults OK for A6000
       python stress_gpu.py --ratio 0.95 --size 6144  # tweak if needed
"""
import argparse, signal, sys, torch, torch.multiprocessing as mp

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--ratio', type=float, default=0.98,
                   help='fraction of *free* VRAM to reserve as dummy buffer')
    p.add_argument('--size',  type=int,   default=8192,
                   help='matrix dimension used for stress matmul (square)')
    p.add_argument('--dtype', default='float32',
                   choices=['float32', 'float16', 'bfloat16'],
                   help='dtype for both dummy and compute tensors')
    return p.parse_args()

def bytes_per_element(dtype): return torch.tensor([], dtype=dtype).element_size()

def worker(rank, ratio, size, dtype):
    torch.cuda.set_device(rank)
    dev = f'cuda:{rank}'
    dt  = getattr(torch, dtype)

    # 1) reserve VRAM ---------------------------------------------------------
    free, tot = torch.cuda.mem_get_info()
    reserve = int(free * ratio)
    dummy_elems = reserve // bytes_per_element(dt)
    dummy = torch.empty(dummy_elems, dtype=dt, device=dev)  # noqa: F841

    # 2) build compute tensors ------------------------------------------------
    A = torch.randn(size, size, dtype=dt, device=dev)
    B = torch.randn(size, size, dtype=dt, device=dev)
    C = torch.empty_like(A)

    print(f"[GPU{rank}] reserved {reserve/1e9:.1f} GB   "
          f"matmul size {size}×{size}   dtype {dtype}")

    # 3) burn the SMs ---------------------------------------------------------
    while True:
        torch.matmul(A, B, out=C)

def main():
    args = parse()
    mp.set_start_method('spawn', force=True)

    procs = []
    for g in range(torch.cuda.device_count()):
        p = mp.Process(target=worker,
                       args=(g, args.ratio, args.size, args.dtype))
        p.start(); procs.append(p)

    # graceful shutdown -------------------------------------------------------
    def _kill(*_):
        for p in procs: p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _kill)
    signal.signal(signal.SIGTERM, _kill)
    for p in procs: p.join()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        sys.exit("💥  No CUDA devices found.")
    main()