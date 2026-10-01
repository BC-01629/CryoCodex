import numpy as np
import torch
import numexpr as ne

from grid_utils import (map_regridder, make_cubic, get_auto_mask,
                        grid_segmentation)


class DateProcessor:
    def __init__(self, target_voxel_size=1.0):
        self.target_voxel_size = target_voxel_size

    def normalize_optimized(self, grid, q99999, threads=4):
        ne.set_num_threads(threads)
        ne.evaluate("where(grid/q99999 > 1, 1, where(grid/q99999 < 0, 0, grid/q99999))",
                    out=grid, casting='same_kind')
        return grid

    def gen_mask(self, map_in_path):
        grid, global_origin, voxel_size, q99999 = map_regridder(
            map_in_path, target_voxel_size=self.target_voxel_size, return_cubic=False)
        padd_grid, start_coords, end_coords = make_cubic(grid)
        mask_ft_cubic = get_auto_mask(padd_grid, voxel_size[0])
        mask_ft = self.inverse_cubic(mask_ft_cubic, start_coords, end_coords)
        nor = self.normalize_optimized(grid, q99999)
        return nor, mask_ft, global_origin, voxel_size

    def inverse_cubic(self, nbox, start_coords, end_coords):
        return nbox[start_coords[0]:end_coords[0],
                    start_coords[1]:end_coords[1],
                    start_coords[2]:end_coords[2]]

    def crop_by_mask(self, grid, mask_ft, pad=5, bounds=None):
        assert grid.shape == mask_ft.shape
        shape = np.array(grid.shape)
        if bounds is None:
            coords = np.argwhere(mask_ft > 0)
            if coords.shape[0] == 0:
                print('# Mask cropping complete')
                return grid, mask_ft, np.zeros(3, dtype=int)
            min_idx = coords.min(axis=0) - pad
            max_idx = coords.max(axis=0) + pad
            min_idx = np.maximum(min_idx, 0)
            max_idx = np.minimum(max_idx, shape - 1)
        else:
            from crop_check import validate_bounds
            bounds = validate_bounds(bounds, grid.shape)
            min_idx = np.array([bounds[a][0] for a in 'zyx'])
            max_idx = np.array([bounds[a][1] for a in 'zyx'])
        cropped_grid = grid[min_idx[0]:max_idx[0] + 1,
                            min_idx[1]:max_idx[1] + 1,
                            min_idx[2]:max_idx[2] + 1]
        cropped_mask = mask_ft[min_idx[0]:max_idx[0] + 1,
                               min_idx[1]:max_idx[1] + 1,
                               min_idx[2]:max_idx[2] + 1]
        print('# Mask cropping complete')
        return cropped_grid, cropped_mask, min_idx[::-1]

    def cut_box(self, stride, grid, window_size=48):
        padding = [(0, 0)] * (grid.ndim - 3) + [(0, max(0, window_size - s)) for s in grid.shape[-3:]]
        grid = np.pad(grid, padding)
        shape = np.array(grid.shape[-3:])
        segmentation = grid_segmentation(grid, stride=stride, windows_size=window_size)
        segmentation = torch.stack(segmentation, dim=0)
        segmentation = segmentation[:, None] * 100
        print('# Box cutting complete, start inference')
        return segmentation.shape[0], segmentation, shape

    def pre_data_for_infer(self, stride, window_size, map_in_path, mode, crop_check_dir=None):
        """Prepare the inference boxes.

        mode == 'fast'   : crop the background away to speed up the inference
        mode == 'normal' : infer on the whole map, no cropping

        Returns the last item as the crop bookkeeping needed to put the outputs back on the
        uncropped canvas: {'shape': uncropped grid shape, 'start': xyz start, 'origin': uncropped
        origin}, or None in normal mode where nothing was cropped.
        """
        grid, mask, global_origin, voxel_size = self.gen_mask(map_in_path)
        crop_box = None
        if mode == 'fast':
            full_shape = grid.shape
            full_origin = np.asarray(global_origin)
            bounds = None
            if crop_check_dir is not None:
                from crop_check import review_crop
                bounds = review_crop(grid, mask, voxel_size, crop_check_dir)
            grid, mask, start = self.crop_by_mask(grid, mask, bounds=bounds)
            global_origin = np.asarray(global_origin) + start * np.asarray(voxel_size)
            crop_box = {'shape': full_shape, 'start': np.asarray(start), 'origin': full_origin}
        elif mode == 'normal':
            print('# Normal mode: keeping the whole map, no background cropping')
        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'fast' or 'normal'")
        total_batch_num, seg, shape = self.cut_box(stride, grid, window_size)
        return shape, total_batch_num, seg, global_origin, voxel_size, crop_box
