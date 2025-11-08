import argparse, os, math, json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dip.models import make_model
from dip.tasks.common import make_task
from dip.tasks.denoise import DenoiseTask
from dip.tasks.superres import SuperResTask
from dip.tasks.inpaint import InpaintTask
from dip.utils.seed import seed_everything
from dip.utils.image_io import load_image, save_image
from dip.utils.metrics import psnr, ssim_simple, total_variation_l1, high_freq_energy
from dip.utils.earlystop import EarlyStopper
from dip.optim.get_optimizer import get_optimizer
from dip.operators.masks import load_mask

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, required=True, choices=['denoise','superres','inpaint'])
    ap.add_argument('--image', type=str, required=True, help='path to observed image (noisy/LR/corrupted)')
    ap.add_argument('--gt', type=str, default=None, help='path to ground-truth (optional)')
    ap.add_argument('--mask', type=str, default=None, help='mask for inpainting (1=known,0=missing)')
    ap.add_argument('--scale', type=int, default=4, help='SR scale')
    ap.add_argument('--gray', action='store_true')
    ap.add_argument('--net', type=str, default='unet', choices=['unet','encdec','resunet','convnext'])
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--features', type=int, default=64)
    ap.add_argument('--skip', action='store_true', help='use skip connections (for unet/resunet)')
    ap.add_argument('--in_ch', type=int, default=32, help='channels of input code z')
    ap.add_argument('--noise_std', type=float, default=0.1, help='std of input noise z in [0,1] scale')
    ap.add_argument('--optimizer', type=str, default='adam', choices=['sgd','adam','rmsprop','lbfgs'])
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--iterations', type=int, default=2000)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', type=str, default='runs/exp')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--early_stop', type=int, default=0, help='patience; 0 disables')
    ap.add_argument('--z_jitter', type=float, default=0.0, help='re-sample small noise each step (DIP sampling trick)')
    return ap.parse_args()

def main():
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    # load observed and (optional) ground truth
    obs = load_image(args.image, gray=args.gray).to(device)
    gt = load_image(args.gt, gray=args.gray).to(device) if args.gt else None

    # build task
    extra = {}
    if args.task == 'superres':
        extra['scale'] = args.scale
    if args.task == 'inpaint':
        if args.mask is None:
            raise ValueError('Provide --mask for inpainting.')
        m = load_mask(args.mask, size=obs.shape[-2:])
        extra['mask'] = m.to(device)
    task = make_task(args.task, **extra)

    # model and code z
    model = make_model(args.net, in_ch=args.in_ch, out_ch=obs.shape[1], depth=args.depth, features=args.features, skip=args.skip).to(device)
    z = torch.randn(1, args.in_ch, obs.shape[-2], obs.shape[-1], device=device) * args.noise_std
    z = z.clamp(-1,1)

    # optimizer
    opt = get_optimizer(model.parameters(), name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # early stopper on PSNR (vs GT if given else vs OBS)
    stopper = EarlyStopper(mode='max', patience=args.early_stop) if args.early_stop > 0 else None
    best_img = None

    hist = []
    pbar = tqdm(range(args.iterations), ncols=100)
    for it in pbar:
        model.train()
        opt.zero_grad()

        zin = z
        if args.z_jitter > 0:
            zin = z + torch.randn_like(z) * args.z_jitter

        y = model(zin)
        loss = task.data_loss(y, obs)
        if args.optimizer == 'lbfgs':
            def closure():
                opt.zero_grad()
                y2 = model(zin)
                l = task.data_loss(y2, obs)
                l.backward()
                return l
            loss = opt.step(closure)
        else:
            loss.backward()
            opt.step()

        # metrics + logging
        with torch.no_grad():
            y_eval = model(z)
            rep = task.report(y_eval, obs, gt=gt)
            rep['iter'] = it+1
            rep['tv'] = float(total_variation_l1(y_eval).item())
            rep['hf_energy'] = float(high_freq_energy(y_eval).item())
            hist.append(rep)

            if (it+1) % args.log_every == 0 or it == args.iterations-1:
                pbar.set_description(f"loss={rep['loss']:.4f} psnr={rep['psnr']:.2f} tv={rep['tv']:.4f}")
                save_image(y_eval, os.path.join(args.save_dir, f'iter_{it+1:06d}.png'))

            if stopper is not None:
                if stopper.step(rep['psnr'], state=y_eval.detach().cpu(), iteration=it+1):
                    # early stop
                    break

    # save best / last
    final = hist[-1]
    save_image(model(z), os.path.join(args.save_dir, 'final.png'))
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    if stopper is not None and stopper.best_state is not None:
        from dip.utils.image_io import save_image as _save
        _save(stopper.best_state, os.path.join(args.save_dir, 'best.png'))
        with open(os.path.join(args.save_dir, 'best_iter.txt'), 'w') as f:
            f.write(str(stopper.best_iter))

if __name__ == '__main__':
    main()
