"""
Main experiment runner for Deep Image Prior experiments.

This module provides high-level functions to run complete experiments
across multiple images and architectures.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .config import ExperimentConfig, get_default_config
from .models import get_dip_network, get_deep_decoder_network, get_transformer_network
from .tasks import run_denoising, run_superresolution, run_inpainting
from .utils import count_parameters


def create_model(model_type: str, config: ExperimentConfig, img_size: Tuple[int, int]):
    """
    Create model based on type.
    
    Args:
        model_type: "dip", "deep_decoder", or "transformer"
        config: Experiment configuration
        img_size: Image size (H, W)
    
    Returns:
        (model, model_type_str) tuple
    """
    if model_type == "dip":
        model = get_dip_network(
            input_depth=config.dip.input_depth,
            num_channels=config.dip.num_channels,
            num_levels=config.dip.num_levels,
            skip_channels=config.dip.skip_channels,
        )
        return model, "dip"
    
    elif model_type == "deep_decoder":
        model = get_deep_decoder_network(
            output_size=img_size,
            input_channels=config.deep_decoder.input_channels,
            input_spatial_size=config.deep_decoder.input_spatial_size,
            channels=config.deep_decoder.channels,
            learnable_input=config.deep_decoder.learnable_input,
        )
        return model, "deep_decoder"
    
    elif model_type == "transformer":
        model = get_transformer_network(
            architecture=config.transformer.architecture,
            img_size=min(img_size),
            in_channels=config.transformer.input_channels,
            embed_dim=config.transformer.embed_dim,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            mlp_ratio=config.transformer.mlp_ratio,
        )
        return model, "transformer"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    model_type: str = "dip",
    config: Optional[ExperimentConfig] = None,
    tasks: List[str] = None,
    save_results: bool = True
) -> Dict:
    """
    Run complete experiment across all images.
    
    Args:
        model_type: "dip", "deep_decoder", or "transformer"
        config: Experiment configuration (uses default if None)
        tasks: List of tasks to run ["denoise", "superres", "inpaint"]
        save_results: Whether to save results to files
    
    Returns:
        Dictionary with all results and logs
    """
    if config is None:
        config = get_default_config()
    
    if tasks is None:
        tasks = ["denoise", "superres", "inpaint"]
    
    # Get model-specific config
    if model_type == "dip":
        model_config = config.dip
        output_subdir = "output_dip"
    elif model_type == "deep_decoder":
        model_config = config.deep_decoder
        output_subdir = "output_deep_decoder"
    elif model_type == "transformer":
        model_config = config.transformer
        output_subdir = "output_transformer"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    output_path = os.path.join(config.paths.base_path, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    # Results storage
    results = {
        "model_type": model_type,
        "config": asdict(model_config),
        "scores": {task: {"highres": [], "lowres": []} for task in tasks},
        "logs": {task: {"highres": {}, "lowres": {}} for task in tasks},
    }
    
    print(f"\n{'='*60}")
    print(f"Running {model_type.upper()} Experiments")
    print(f"{'='*60}")
    
    for resolution in config.paths.resolutions:
        print(f"\n--- Processing {resolution} images ---")
        
        for img_folder in config.paths.image_folders:
            img_dir = os.path.join(config.paths.images_path, resolution, img_folder)
            out_dir = os.path.join(output_path, resolution, img_folder)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"\n[{img_folder}]")
            
            # File paths
            clean_path = os.path.join(img_dir, "clean.jpg")
            noisy_path = os.path.join(img_dir, "noisy.jpg")
            lr_path = os.path.join(img_dir, "lr_x4.jpg")
            corrupted_path = os.path.join(img_dir, "corrupted.jpg")
            mask_path = os.path.join(img_dir, "mask.jpg")
            
            # Denoising
            if "denoise" in tasks and os.path.exists(noisy_path):
                print("  Running denoising...")
                model, mtype = create_model(model_type, config, (model_config.max_image_size, model_config.max_image_size))
                print(f"    Model parameters: {count_parameters(model):,}")
                
                psnr, logs = run_denoising(
                    model=model,
                    noisy_img_path=noisy_path,
                    output_path=os.path.join(out_dir, "denoised.png"),
                    clean_img_path=clean_path if os.path.exists(clean_path) else None,
                    num_iter=model_config.num_iter_denoise,
                    lr=model_config.learning_rate,
                    max_size=model_config.max_image_size,
                    log_every=config.log_every,
                    use_amp=model_config.use_mixed_precision,
                    model_type=mtype,
                )
                
                results["scores"]["denoise"][resolution].append(psnr)
                results["logs"]["denoise"][resolution][img_folder] = logs
            
            # Super-resolution
            if "superres" in tasks and os.path.exists(lr_path):
                print("  Running super-resolution...")
                model, mtype = create_model(model_type, config, (model_config.max_image_size, model_config.max_image_size))
                
                psnr, logs = run_superresolution(
                    model=model,
                    lowres_img_path=lr_path,
                    output_path=os.path.join(out_dir, "superres.png"),
                    clean_img_path=clean_path if os.path.exists(clean_path) else None,
                    up_factor=config.up_factor,
                    num_iter=model_config.num_iter_superres,
                    lr=model_config.learning_rate,
                    max_size=model_config.max_image_size,
                    log_every=config.log_every,
                    use_amp=model_config.use_mixed_precision,
                    model_type=mtype,
                )
                
                results["scores"]["superres"][resolution].append(psnr)
                results["logs"]["superres"][resolution][img_folder] = logs
            
            # Inpainting
            if "inpaint" in tasks and os.path.exists(corrupted_path) and os.path.exists(mask_path):
                print("  Running inpainting...")
                model, mtype = create_model(model_type, config, (model_config.max_image_size, model_config.max_image_size))
                
                psnr, logs = run_inpainting(
                    model=model,
                    corrupted_img_path=corrupted_path,
                    mask_path=mask_path,
                    output_path=os.path.join(out_dir, "inpainted.png"),
                    clean_img_path=clean_path if os.path.exists(clean_path) else None,
                    num_iter=model_config.num_iter_inpaint,
                    lr=model_config.learning_rate,
                    max_size=model_config.max_image_size,
                    log_every=config.log_every,
                    use_amp=model_config.use_mixed_precision,
                    model_type=mtype,
                )
                
                results["scores"]["inpaint"][resolution].append(psnr)
                results["logs"]["inpaint"][resolution][img_folder] = logs
    
    # Save results
    if save_results:
        results_path = os.path.join(output_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print_summary(results, tasks)
    
    return results


def print_summary(results: Dict, tasks: List[str]):
    """Print experiment summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - {results['model_type'].upper()}")
    print(f"{'='*60}")
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        for resolution in ["highres", "lowres"]:
            scores = [s for s in results["scores"][task][resolution] if s is not None]
            if scores:
                mean_psnr = np.mean(scores)
                std_psnr = np.std(scores)
                print(f"  {resolution.capitalize():10s}: {mean_psnr:.2f} ± {std_psnr:.2f} dB (n={len(scores)})")


def plot_psnr_curves(
    results: Dict,
    output_path: str,
    tasks: List[str] = None
):
    """
    Plot PSNR curves from experiment results.
    
    Args:
        results: Results dictionary from run_experiment
        output_path: Path to save plot
        tasks: Tasks to plot
    """
    if tasks is None:
        tasks = ["denoise", "superres", "inpaint"]
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]
    
    for ax, task in zip(axes, tasks):
        for resolution in ["highres", "lowres"]:
            # Collect all PSNR curves
            all_psnrs = []
            all_iters = None
            
            for img_folder, logs in results["logs"][task][resolution].items():
                if logs["psnr"] and logs["psnr"][0] is not None:
                    all_psnrs.append(logs["psnr"])
                    if all_iters is None:
                        all_iters = logs["iterations"]
            
            if all_psnrs and all_iters:
                # Compute mean and std
                psnrs_array = np.array(all_psnrs)
                mean_psnr = np.mean(psnrs_array, axis=0)
                std_psnr = np.std(psnrs_array, axis=0)
                
                color = 'blue' if resolution == 'highres' else 'red'
                label = f"{resolution.capitalize()} (n={len(all_psnrs)})"
                
                ax.plot(all_iters, mean_psnr, color=color, label=label)
                ax.fill_between(all_iters, mean_psnr - std_psnr, mean_psnr + std_psnr,
                               color=color, alpha=0.2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(task.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    model_type = results["model_type"].upper()
    lr = results["config"].get("learning_rate", "?")
    fig.suptitle(f"{model_type}: Mean PSNR vs Iteration - LR={lr}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
