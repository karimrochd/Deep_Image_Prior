import argparse, json, itertools, os, subprocess, sys

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['denoise','superres','inpaint'])
    ap.add_argument('--image', required=True)
    ap.add_argument('--gt', default=None)
    ap.add_argument('--mask', default=None)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--grid', type=str, default='configs/grid_search.json')
    ap.add_argument('--save_dir', type=str, default='runs/sweep')
    return ap.parse_args()

def main():
    args = parse()
    with open(args.grid, 'r') as f:
        grid = json.load(f)
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    os.makedirs(args.save_dir, exist_ok=True)
    for combo in itertools.product(*values):
        cfg = {k:v for k,v in zip(keys, combo)}
        tag = "_".join(f"{k}-{v}" for k,v in cfg.items())
        out = os.path.join(args.save_dir, tag)
        cmd = [
            sys.executable, 'train.py',
            '--task', args.task, '--image', args.image, '--save_dir', out,
            '--net', cfg.get('net','unet'), '--depth', str(cfg.get('depth',5)), '--features', str(cfg.get('features',64)),
            '--optimizer', cfg.get('optimizer','adam'), '--lr', str(cfg.get('lr',0.01)),
            '--iterations', str(cfg.get('iterations',2000))
        ]
        if args.gt: cmd += ['--gt', args.gt]
        if args.task == 'superres': cmd += ['--scale', str(args.scale)]
        if args.task == 'inpaint':
            if not args.mask: raise ValueError('Provide --mask for inpaint sweep')
            cmd += ['--mask', args.mask]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
