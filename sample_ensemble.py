import argparse, os, json, torch
from dip.utils.seed import seed_everything
from dip.utils.image_io import load_image, save_image
from dip.models import make_model
from dip.tasks.common import make_task
from dip.optim.get_optimizer import get_optimizer
from dip.operators.masks import load_mask
from tqdm import tqdm

def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['denoise','superres','inpaint'])
    ap.add_argument('--image', required=True)
    ap.add_argument('--gt', default=None)
    ap.add_argument('--mask', default=None)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--net', default='unet')
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--features', type=int, default=64)
    ap.add_argument('--in_ch', type=int, default=32)
    ap.add_argument('--optimizer', default='adam')
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--iterations', type=int, default=1800)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--aggregate', default='mean', choices=['mean','median'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', default='runs/ensemble')
    return ap.parse_args()

def train_one(seed, args, obs, task, device):
    seed_everything(seed)
    model = make_model(args.net, in_ch=args.in_ch, out_ch=obs.shape[1], depth=args.depth, features=args.features).to(device)
    z = torch.randn(1, args.in_ch, obs.shape[-2], obs.shape[-1], device=device) * 0.1
    opt = get_optimizer(model.parameters(), name=args.optimizer, lr=args.lr)
    for it in range(args.iterations):
        opt.zero_grad()
        y = model(z)
        loss = task.data_loss(y, obs)
        loss.backward()
        opt.step()
    with torch.no_grad():
        y = model(z).detach().clamp(0,1)
    return y

def main():
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    obs = load_image(args.image).to(device)
    extra = {}
    if args.task == 'superres': extra['scale'] = args.scale
    if args.task == 'inpaint': extra['mask'] = load_mask(args.mask, size=obs.shape[-2:]).to(device)
    task = make_task(args.task, **extra)

    outs = []
    for s in tqdm(range(args.seeds), desc='Ensemble'):
        y = train_one(1000+s, args, obs, task, device)
        outs.append(y)
        save_image(y, os.path.join(args.save_dir, f'seed_{s:02d}.png'))

    Y = torch.stack(outs, dim=0)
    if args.aggregate == 'mean':
        agg = Y.mean(0, keepdim=False)
    else:
        agg = Y.median(0, keepdim=False).values
    save_image(agg, os.path.join(args.save_dir, 'ensemble.png'))
    print('Saved:', os.path.join(args.save_dir, 'ensemble.png'))

if __name__ == '__main__':
    main()
