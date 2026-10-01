import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

from grid_utils import (get_batch_slices, grid_reconstruction_gaussian_multi,
                        extend_edge, save_grid_as_map, read_voxel_size,
                        interpolate_back_to_native_voxel)
from data_processor import DateProcessor
from VQUnet import MMVQUnet

data_processor = DateProcessor(target_voxel_size=1.0)
__version__ = "1.0.0"
WINDOW_SIZE = 48
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_state_dicts', 'CryoCodex.pth')

BANNER = f"""
=====================================================================================

 ██████╗ ██████╗  ██╗   ██╗  ██████╗   ██████╗  ██████╗  ██████╗  ███████╗ ██╗  ██╗
██╔════╝ ██╔══██╗ ╚██╗ ██╔╝ ██╔═══██╗ ██╔════╝ ██╔═══██╗ ██╔══██╗ ██╔════╝ ╚██╗██╔╝
██║      ██████╔╝  ╚████╔╝  ██║   ██║ ██║      ██║   ██║ ██║  ██║ █████╗    ╚███╔╝
██║      ██╔══██╗   ╚██╔╝   ██║   ██║ ██║      ██║   ██║ ██║  ██║ ██╔══╝    ██╔██╗
╚██████╗ ██║  ██║    ██║    ╚██████╔╝ ╚██████╗ ╚██████╔╝ ██████╔╝ ███████╗ ██╔╝ ██╗
 ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═════╝   ╚═════╝  ╚═════╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═╝

                              CryoCodex v{__version__}
          Enhancement, local quality estimation and molecular mask prediction
                                                    By Bin Cheng, Yang lab.
=====================================================================================
"""


def parse_bool(value):
    """Parse an explicit True/False command line value (case-insensitive)."""
    if value.strip().lower() == 'true':
        return True
    if value.strip().lower() == 'false':
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")


def load_network(model, checkpoint_path, device='cuda', strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        got = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise RuntimeError(
            f"Unrecognized checkpoint format in '{checkpoint_path}': expected a dict with a "
            f"'model_state_dict' entry, got {got}"
        )
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load the checkpoint '{checkpoint_path}' into "
            f"{type(model).__name__} (strict={strict}): {e}"
        ) from e
    model.to(device)
    model.eval()
    return model


def restore_full_size(grid, crop_box, fill=0):
    """Put a cropped grid back on the uncropped canvas, the background getting `fill`."""
    canvas = np.full(crop_box['shape'], fill, dtype=np.float32)
    slices = tuple(slice(s, s + n) for s, n in zip(crop_box['start'][::-1], grid.shape))
    canvas[slices] = grid
    return canvas


def score_inference(in_map_path, output_path, model_path, batch_size, out_name, Net,
                    stride=12, if_reverse_interpolation=False, normal=False, keep_size=False,
                    show_logo=True, crop_check=False):
    if show_logo:
        print(f"\n{BANNER}\n")
    os.makedirs(output_path, exist_ok=True)
    print(f'Improving {in_map_path} now, out_name:{out_name}')
    device = torch.device("cuda:0")
    mode = 'normal' if normal else 'fast'
    shape, total_batch_num, seg, global_origin, voxel_size, crop_box = data_processor.pre_data_for_infer(
        stride, WINDOW_SIZE, in_map_path, mode=mode,
        crop_check_dir=output_path if (not normal and crop_check) else None)
    module = load_network(Net, checkpoint_path=model_path, device=device, strict=True)
    pbar = tqdm(total=total_batch_num, file=sys.stdout, position=0, leave=True)
    grid_batches = get_batch_slices(total_batch_num, batch_size)
    seg = seg.pin_memory()
    with torch.no_grad():
        outputs = np.zeros((seg.shape[0], 4, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        next_x = seg[grid_batches[0]].to(device, non_blocking=True)
        for i, grid_batch in enumerate(grid_batches):
            x = next_x
            if i + 1 < len(grid_batches):
                next_x = seg[grid_batches[i + 1]].to(device, non_blocking=True)
            out = module(x)
            out = torch.cat((out[0], out[1], out[2], out[3]), dim=1).cpu().numpy()
            outputs[grid_batch] = out
            pbar.update(grid_batch.stop - grid_batch.start)
    reconstructed = grid_reconstruction_gaussian_multi(outputs, shape, stride=stride, windows_size=WINDOW_SIZE)
    pred, score_o, mask, score_i = reconstructed
    pbar.close()
    binary_mask = (mask >= 0.5).astype(np.float32)
    kernel = np.zeros((3, 3, 3))
    kernel[:, 1, 1] = 1
    kernel[1, :, 1] = 1
    kernel[1, 1, :] = 1
    soft_mask = extend_edge(
        binary_mask, edge=15, kernel=kernel,
        ramp=np.cos(np.linspace(0, np.pi, 15)) * 0.5 + 0.5,
    )
    # Fade every output out at the boundary of the input mask, except the mask itself.
    pred *= soft_mask
    if keep_size:
        # Put the outputs back on the uncropped canvas. The density has already faded out towards
        # the crop boundary, so filling the background with zeros stays continuous. The mask stays
        # binary and the scores stay zero outside the crop.
        print(f"# Filling the cropped away background back in, output grid is {crop_box['shape']}")
        pred = restore_full_size(pred, crop_box)
        score_o = restore_full_size(score_o, crop_box)
        score_i = restore_full_size(score_i, crop_box)
        binary_mask = restore_full_size(binary_mask, crop_box)
        global_origin = crop_box['origin']
    if if_reverse_interpolation:
        native_voxel_size = read_voxel_size(in_map_path)
        if np.allclose(native_voxel_size, voxel_size, atol=1e-4):
            print(f"# Input map is already on the {np.asarray(voxel_size)[0]} Angstrom grid, "
                  f"skipping the interpolation back to the native voxel size")
        else:
            print(f"# Interpolating the outputs from the {np.asarray(voxel_size)[0]} Angstrom grid "
                  f"back to the native voxel size {native_voxel_size}")
            pred, new_origin, new_voxel_size = interpolate_back_to_native_voxel(
                pred, global_origin, voxel_size, native_voxel_size)
            score_o = interpolate_back_to_native_voxel(
                score_o, global_origin, voxel_size, native_voxel_size)[0]
            score_i = interpolate_back_to_native_voxel(
                score_i, global_origin, voxel_size, native_voxel_size)[0]
            binary_mask = interpolate_back_to_native_voxel(
                binary_mask, global_origin, voxel_size, native_voxel_size)[0]
            global_origin, voxel_size = new_origin, new_voxel_size
            binary_mask = (binary_mask >= 0.5).astype(np.float32)
            print(f"# Output grid is now {pred.shape} at {voxel_size[0]} Angstrom")

    # Clamp after resampling as interpolation can introduce negative scores.
    np.maximum(score_o, 0, out=score_o)
    np.maximum(score_i, 0, out=score_i)

    save_grid_as_map(grid=pred, global_origin=global_origin, voxel_size=voxel_size, save_path=output_path, name=out_name)
    save_grid_as_map(grid=score_o, global_origin=global_origin, voxel_size=voxel_size, save_path=output_path, name=out_name + '_out_score')
    save_grid_as_map(grid=binary_mask, global_origin=global_origin, voxel_size=voxel_size, save_path=output_path, name=out_name + '_in_mask')
    save_grid_as_map(grid=score_i, global_origin=global_origin, voxel_size=voxel_size, save_path=output_path, name=out_name + '_in_score')


def main():
    # `predict.sh` exports CRYOCODEX_PROG so the usage line names the launcher the user typed
    parser = argparse.ArgumentParser(
        prog=os.environ.get("CRYOCODEX_PROG") or os.path.basename(sys.argv[0]),
        description="CryoCodex: cryo-EM map enhancement with local quality scores",
    )
    parser.add_argument("--version",action="version",version=f"CryoCodex v{__version__}")
    parser.add_argument("-i", "--in_map_path", type=str, required=True, metavar="MAP",
                        help="Input EM density map (.mrc/.map)")
    parser.add_argument("-o", "--out_dir", type=str, required=True, metavar="DIR",
                        help="Directory to save the output maps")
    parser.add_argument("--normal", type=parse_bool, default=False, metavar="True|False",
                        help="Infer on the whole map in normal mode; False (the default) runs fast mode "
                             "instead, which crops the background away and is much quicker")
    parser.add_argument("--reverse_interpolation", "-a", type=parse_bool, default=False, metavar="True|False",
                        help="Resample the saved maps back to the voxel size of the input map; False (the "
                             "default) saves them on the 1.0 Angstrom grid")
    parser.add_argument("--crop_check", "-c", type=parse_bool, default=False, metavar="True|False",
                        help="Fast mode only: review the cropped region before the inference")
    parser.add_argument("--keep_size", type=parse_bool, default=False, metavar="True|False",
                               
                        help="Fast mode only: fill the cropped away background back in with zeros,"
                             " so the outputs span the whole input map; this restores the grid shape at the same voxel "
                             "size, unlike --reverse_interpolation which changes the voxel size")
    parser.add_argument("--out_name", "-n", type=str, default="cryocodex", help="Base name of the output maps")
    parser.add_argument("--stride", "-s", type=int, default=12, help="Sliding-window stride")
    parser.add_argument("--batch_size", "-b", type=int, default=9, help="Number of boxes in one batch")
    parser.add_argument("--gpu", "-g", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--no_logo", type=parse_bool, default=False, metavar="True|False",
                        help="Do not show the logo banner")
    parser.add_argument("--model_path", "-m", type=str, default=MODEL_PATH, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.crop_check and args.normal:
        parser.error("--crop_check is only available in fast mode, so it needs --normal False")
    if args.keep_size and args.normal:
        parser.error("--keep_size is only available in fast mode, so it needs --normal False")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    score_inference(
        in_map_path=args.in_map_path,
        output_path=args.out_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        out_name=args.out_name,
        Net=MMVQUnet(drop_path_rate=0.),
        stride=args.stride,
        if_reverse_interpolation=args.reverse_interpolation,
        normal=args.normal,
        keep_size=args.keep_size,
        show_logo=not args.no_logo,
        crop_check=args.crop_check,
    )


if __name__ == "__main__":
    main()
