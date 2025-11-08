import argparse, os, json, torch
from dip.utils.image_io import load_image, save_image
from dip.models import make_model
from dip.tasks.common import make_task
from dip.optim.get_optimizer import get_optimizer
from dip.operators.masks import load_mask
from dip.utils.metrics import psnr, total_variation_l1, high_freq_energy
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse():
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
    ap.add_argument('--optimizer', default='sgd', choices=['sgd','adam','rmsprop'])
    ap.add_argument('--lrs', nargs='+', type=float, required=True)
    ap.add_argument('--iterations', type=int, default=2000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', default='runs/lr_bias')
    return ap.parse_args()

def one_run(lr, args, obs, task, device, gt=None):
    model = make_model(args.net, in_ch=args.in_ch, out_ch=obs.shape[1], depth=args.depth, features=args.features).to(device)
    z = torch.randn(1, args.in_ch, obs.shape[-2], obs.shape[-1], device=device) * 0.1
    opt = get_optimizer(model.parameters(), name=args.optimizer, lr=lr, momentum=0.9)
    history = []
    best_psnr = -1e9
    best_img = None
    for it in range(args.iterations):
        opt.zero_grad()
        y = model(z)
        loss = task.data_loss(y, obs)
        loss.backward()
        opt.step()
        with torch.no_grad():
            yv = model(z).clamp(0,1)
            rep = {'iter': it+1, 'loss': float(loss.item())}
            rep['tv'] = float(total_variation_l1(yv).item())
            rep['hf'] = float(high_freq_energy(yv).item())
            rep['psnr_obs'] = float(psnr(yv, obs).item())
            if gt is not None:
                rep['psnr_gt'] = float(psnr(yv, gt).item())
                if rep['psnr_gt'] > best_psnr:
                    best_psnr = rep['psnr_gt']; best_img = yv.detach().cpu()
            history.append(rep)
    return history, best_img

def main():
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    obs = load_image(args.image).to(device)
    gt = load_image(args.gt).to(device) if args.gt else None
    extra = {}
    if args.task == 'superres': extra['scale'] = args.scale
    if args.task == 'inpaint': extra['mask'] = load_mask(args.mask, size=obs.shape[-2:]).to(device)
    task = make_task(args.task, **extra)

    summaries = []
    for lr in args.lrs:
        history, best_img = one_run(lr, args, obs, task, device, gt=gt)
        with open(os.path.join(args.save_dir, f'lr_{lr}.json'), 'w') as f:
            json.dump(history, f, indent=2)
        if best_img is not None:
            save_image(best_img, os.path.join(args.save_dir, f'best_lr_{lr}.png'))
        # summarize
        end = history[-1]
        summaries.append({
            'lr': lr,
            'final_psnr_obs': end['psnr_obs'],
            'final_tv': end['tv'],
            'final_hf': end['hf'],
            'best_psnr_gt': max([h.get('psnr_gt', -1) for h in history])
        })

    # quick plots (Matplotlib; single plots and default colors)
    lrs = [s['lr'] for s in summaries]
    tvs = [s['final_tv'] for s in summaries]
    hfs = [s['final_hf'] for s in summaries]
    psnrs = [s['final_psnr_obs'] for s in summaries]

    plt.figure()
    plt.plot(lrs, tvs, marker='o')
    plt.xscale('log'); plt.xlabel('learning rate'); plt.ylabel('TV (L1)')
    plt.title('Lissage (TV) vs LR')
    plt.savefig(os.path.join(args.save_dir, 'tv_vs_lr.png'), dpi=150)

    plt.figure()
    plt.plot(lrs, hfs, marker='o')
    plt.xscale('log'); plt.xlabel('learning rate'); plt.ylabel('High-frequency energy')
    plt.title('Énergie HF vs LR')
    plt.savefig(os.path.join(args.save_dir, 'hf_vs_lr.png'), dpi=150)

    plt.figure()
    plt.plot(lrs, psnrs, marker='o')
    plt.xscale('log'); plt.xlabel('learning rate'); plt.ylabel('PSNR (obs)')
    plt.title('PSNR(obs) vs LR')
    plt.savefig(os.path.join(args.save_dir, 'psnr_vs_lr.png'), dpi=150)

    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)

if __name__ == '__main__':
    main()
