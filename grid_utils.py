"""Grid utilities for CryoCodex, with code from EMReady2 and model-angelo.
Copyright (c) 2025 Huang Laboratory.
Copyright (c) 2022 Kiarash Jamali.
MIT license; see THIRD_PARTY_NOTICES.md for full terms and attribution.
"""

import os
import numpy as np
import torch
import mrcfile as mrc
from numba import jit, prange
from scipy.ndimage import binary_dilation


# From EMReady2, interp3d._get_w
@jit(nopython=True)
def _get_w(x, w):
    a = -0.5
    intx = int(np.floor(x))
    d1 = 1.0 + (x - intx)
    d2 = d1 - 1.0
    d3 = 1.0 - d2
    d4 = d3 + 1.0
    w[0] = a * np.abs(d1**3) - 5 * a * d1**2 + 8 * a * np.abs(d1) - 4 * a
    w[1] = (a + 2) * np.abs(d2**3) - (a + 3) * d2**2 + 1
    w[2] = (a + 2) * np.abs(d3**3) - (a + 3) * d3**2 + 1
    w[3] = a * np.abs(d4**3) - 5 * a * d4**2 + 8 * a * np.abs(d4) - 4 * a


# From EMReady2, interp3d.Interp3D
class Interp3D:
    def __init__(self):
        self.mapout = None
        self.pextx = None
        self.pexty = None
        self.pextz = None

    def cubic(self, mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx):
        pextx = int(np.floor(xpix * (nx - 1) / apix)) + 1
        pexty = int(np.floor(ypix * (ny - 1) / apix)) + 1
        pextz = int(np.floor(zpix * (nz - 1) / apix)) + 1
        self.mapout = np.zeros((pextz, pexty, pextx), dtype=np.float32)
        self.pextx = pextx
        self.pexty = pexty
        self.pextz = pextz
        self.mapout = self._cubic_interp(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, self.mapout)
        return self.mapout

    @staticmethod
    @jit(parallel=True, nopython=True)
    def _cubic_interp(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, mapout):
        for indz in prange(pextz):
            for indy in prange(pexty):
                for indx in prange(pextx):
                    gx = (indx * apix + shiftx) / xpix
                    gy = (indy * apix + shifty) / ypix
                    gz = (indz * apix + shiftz) / zpix
                    intx = int(np.floor(gx))
                    inty = int(np.floor(gy))
                    intz = int(np.floor(gz))
                    if intz >= 0 and intz + 1 < nz and inty >= 0 and inty + 1 < ny and intx >= 0 and intx + 1 < nx:
                        wz = np.zeros(4, dtype=np.float32)
                        wy = np.zeros(4, dtype=np.float32)
                        wx = np.zeros(4, dtype=np.float32)
                        _get_w(gz, wz)
                        _get_w(gy, wy)
                        _get_w(gx, wx)
                        for i in range(4):
                            for j in range(4):
                                for k in range(4):
                                    if (intz + i - 1 >= 0 and intz + i - 1 < nz and
                                            inty + j - 1 >= 0 and inty + j - 1 < ny and
                                            intx + k - 1 >= 0 and intx + k - 1 < nx):
                                        mapout[indz, indy, indx] += wz[i] * wy[j] * wx[k] * mapin[intz + i - 1, inty + j - 1, intx + k - 1]
        return mapout

    def inverse_cubic(self, mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx):
        """Resample mapin from the (zpix, ypix, xpix) grid back onto the (zpix_o, ypix_o, xpix_o) grid."""
        pextx = int(np.ceil(xpix * (nx - 1) / xpix_o)) + 1
        pexty = int(np.ceil(ypix * (ny - 1) / ypix_o)) + 1
        pextz = int(np.ceil(zpix * (nz - 1) / zpix_o)) + 1
        self.mapout = np.zeros((pextz, pexty, pextx), dtype=np.float32)
        self.pextx = pextx
        self.pexty = pexty
        self.pextz = pextz
        self.mapout = self._inverse_cubic_interp(mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, self.mapout)
        return self.mapout

    @staticmethod
    @jit(parallel=True, nopython=True)
    def _inverse_cubic_interp(mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, mapout):
        for indz in prange(pextz):
            for indy in prange(pexty):
                for indx in prange(pextx):
                    gx = (indx * xpix_o + shiftx) / xpix
                    gy = (indy * ypix_o + shifty) / ypix
                    gz = (indz * zpix_o + shiftz) / zpix
                    intx = int(np.floor(gx))
                    inty = int(np.floor(gy))
                    intz = int(np.floor(gz))
                    if intz >= 0 and intz + 1 < nz and inty >= 0 and inty + 1 < ny and intx >= 0 and intx + 1 < nx:
                        wz = np.zeros(4, dtype=np.float32)
                        wy = np.zeros(4, dtype=np.float32)
                        wx = np.zeros(4, dtype=np.float32)
                        _get_w(gz, wz)
                        _get_w(gy, wy)
                        _get_w(gx, wx)
                        for i in range(4):
                            for j in range(4):
                                for k in range(4):
                                    if (intz + i - 1 >= 0 and intz + i - 1 < nz and
                                            inty + j - 1 >= 0 and inty + j - 1 < ny and
                                            intx + k - 1 >= 0 and intx + k - 1 < nx):
                                        mapout[indz, indy, indx] += wz[i] * wy[j] * wx[k] * mapin[intz + i - 1, inty + j - 1, intx + k - 1]
        return mapout

    def del_mapout(self):
        self.mapout = None


def read_voxel_size(map_file):
    """Read the voxel size straight from the MRC header, without loading the map."""
    with mrc.open(map_file, mode='r') as mrc1:
        return np.asarray([mrc1.voxel_size.x, mrc1.voxel_size.y, mrc1.voxel_size.z], dtype=np.float32)


# Adapted from EMReady2, utils.inverse_map
def interpolate_back_to_native_voxel(grid, origin, voxel_size, native_voxel_size):
    """Resample a grid that sits on `voxel_size` back onto the native grid of the input map.

    Inverse of the resampling done by map_regridder/parse_map, using the inverse cubic
    spline of Interp3D.inverse_cubic. `origin` and `voxel_size` are (x, y, z); `grid` is
    (z, y, x). Returns (grid, origin, voxel_size) on the native grid.
    """
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    native_voxel_size = np.asarray(native_voxel_size, dtype=np.float32)
    origin_shift = (np.round(origin / native_voxel_size) - origin / native_voxel_size) * native_voxel_size
    nz, ny, nx = grid.shape
    interp3d = Interp3D()
    interp3d.inverse_cubic(
        np.ascontiguousarray(grid, dtype=np.float32),
        voxel_size[2], voxel_size[1], voxel_size[0],
        native_voxel_size[2], native_voxel_size[1], native_voxel_size[0],
        origin_shift[2], origin_shift[1], origin_shift[0],
        nz, ny, nx,
    )
    return interp3d.mapout, origin + origin_shift, native_voxel_size


# Adapted from EMReady2, utils.parse_map
def parse_map(map_file, ignorestart=False, apix=1.0, origin_shift=None):
    interp3d = Interp3D()
    mrc1 = mrc.open(map_file, mode='r')
    map = np.asarray(mrc1.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc1.voxel_size.x, mrc1.voxel_size.y, mrc1.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc1.header.nxstart, mrc1.header.nystart, mrc1.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc1.header.origin.x, mrc1.header.origin.y, mrc1.header.origin.z], dtype=np.float32)
    ncrs = (mrc1.header.nx, mrc1.header.ny, mrc1.header.nz)
    angle = np.asarray([mrc1.header.cellb.alpha, mrc1.header.cellb.beta, mrc1.header.cellb.gamma], dtype=np.float32)

    try:
        assert (angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc1.close()
        exit()

    mapcrs = np.subtract([mrc1.header.mapc, mrc1.header.mapr, mrc1.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])
    map = np.transpose(map, axes=2 - sort[::-1])
    mrc1.close()

    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    if apix is not None:
        try:
            assert (voxel_size[0] == voxel_size[1] == voxel_size[2] == apix and origin_shift is None)
        except AssertionError:
            interp3d.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}".format(voxel_size, target_voxel_size))
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
            voxel_size = target_voxel_size

    assert (np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int64)))
    return map, origin, nxyz, voxel_size


# Adapted from model-angelo, grid.make_cubic
def make_cubic(box):
    bz = np.array(box.shape)
    s = np.max(box.shape)
    s += s % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=int), bz
    nbox = np.zeros((s, s, s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0]:c[0] + bz[0], c[1]:c[1] + bz[1], c[2]:c[2] + bz[2]] = box
    return nbox, c, c + bz


def map_regridder(map_file, ignorestart=False, target_voxel_size=1.0, origin_shift=None, return_cubic=True):
    if map_file.endswith("map") or map_file.endswith("mrc"):
        grid_np, global_origin, _, voxel_size = parse_map(map_file, ignorestart, target_voxel_size, origin_shift)
        try:
            assert np.all(np.abs(np.round(global_origin / voxel_size) - global_origin / voxel_size) < 1e-4)
        except AssertionError:
            origin_shift = (np.round(global_origin / voxel_size) - global_origin / voxel_size) * voxel_size
            grid_np, global_origin, _, voxel_size = parse_map(map_file, ignorestart=False, apix=1.0, origin_shift=origin_shift)
            assert np.all(np.abs(np.round(global_origin / voxel_size) - global_origin / voxel_size) < 1e-4)
        grid_np = grid_np.astype(np.float32)
        q = np.quantile(grid_np[grid_np > 0], 0.99999)
        if return_cubic:
            grid_np, shift, _ = make_cubic(grid_np)
            global_origin = global_origin - shift[::-1] * voxel_size
        return grid_np, global_origin, voxel_size, q
    else:
        raise RuntimeError(f"File {map_file} is not a cryo-em density map file format.")


def get_fourier_shells(f):
    (z, y, x) = f.shape
    Z, Y, X = np.meshgrid(
        np.linspace(-z // 2, z // 2 - 1, z),
        np.linspace(-y // 2, y // 2 - 1, y),
        np.linspace(0, x - 1, x),
        indexing="ij",
    )
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.ifftshift(R, axes=(0, 1))
    return R


def apply_lowpass_filter_to_map(grid, voxel_size, lowpass_ang, filter_edge_width=2, use_cosine_kernel=True):
    grid_ft = np.fft.rfftn(np.fft.fftshift(grid))
    spectral_radius = get_fourier_shells(grid_ft)
    ori_size = grid.shape[0]
    ires_filter = round((ori_size * voxel_size) / lowpass_ang)
    filter_edge_halfwidth = filter_edge_width // 2
    edge_low = max(0, (ires_filter - filter_edge_halfwidth) / ori_size)
    edge_high = min(grid_ft.shape[0], (ires_filter + filter_edge_halfwidth) / ori_size)
    edge_width = edge_high - edge_low
    res = spectral_radius / ori_size
    scale_spectrum = np.zeros_like(res)
    scale_spectrum[res < edge_low] = 1
    if use_cosine_kernel:
        scale_spectrum[(res >= edge_low) & (res <= edge_high)] = 0.5 + 0.5 * np.cos(
            np.pi * (res[(res >= edge_low) & (res <= edge_high)] - edge_low) / edge_width
        )
    grid_ft *= scale_spectrum
    grid = np.fft.ifftshift(np.fft.irfftn(grid_ft))
    return grid


def get_spherical_mask(grid):
    ls = np.linspace(-grid.shape[0] // 2, grid.shape[0] // 2, grid.shape[0])
    rz = ls[:, None, None]
    ry = ls[None, :, None]
    rx = ls[None, None, :]
    r = np.sqrt(rz * rz + ry * ry + rx * rx)
    mask = r < (grid.shape[0] / 2 + 1)
    return mask


def extend_edge(binary_mask, edge, kernel, ramp):
    smooth_mask = np.copy(binary_mask)
    prev_mask = binary_mask.astype(bool)
    for i in range(edge):
        mask = binary_dilation(prev_mask, structure=kernel, iterations=1)
        skin = mask & ~prev_mask
        prev_mask = mask
        smooth_mask[skin] = ramp[i]
    return smooth_mask


def get_mask_from_grid(grid, ini_threshold=0.01, extend_inimask=3, width_soft_edge=3):
    binary_mask = np.zeros_like(grid, dtype=np.float32)
    binary_mask[grid > ini_threshold] = 1
    kernel = np.zeros((3, 3, 3))
    kernel[:, 1, 1] = 1
    kernel[1, :, 1] = 1
    kernel[1, 1, :] = 1
    if extend_inimask > 0:
        binary_mask = extend_edge(binary_mask, extend_inimask, kernel, np.ones((extend_inimask,)))
    if width_soft_edge > 0:
        binary_mask = extend_edge(binary_mask, width_soft_edge, kernel, ramp=np.cos(np.linspace(0, np.pi, width_soft_edge)) * 0.5 + 0.5)
    return binary_mask


# From model-angelo, grid.get_auto_mask and every function it calls above.
def get_auto_mask(grid, voxel_size):
    lowpass_grid = apply_lowpass_filter_to_map(grid, voxel_size, 15)
    s_mask = get_spherical_mask(lowpass_grid)
    lowpass_grid[~s_mask] = 0
    threshold = np.quantile(lowpass_grid[lowpass_grid > 0], q=0.97)
    extend_mask_value = max(5, round(0.01 * grid.shape[0]) + 1)
    mask_grid = get_mask_from_grid(lowpass_grid, threshold, extend_inimask=extend_mask_value, width_soft_edge=extend_mask_value)
    return mask_grid


# Adapted from EMReady2, utils.write_map
def save_grid_as_map(grid, global_origin, voxel_size, save_path, name):
    print('saving')
    mrc_file = os.path.join(save_path, f"{name}.mrc")
    with mrc.new(mrc_file, overwrite=True) as mrc1:
        mrc1.set_data(np.asarray(grid, dtype=np.float32))
        mrc1.voxel_size = tuple(voxel_size)
        origin = tuple(np.float32(x) for x in global_origin)
        mrc1.header.origin['x'] = 0
        mrc1.header.origin['y'] = 0
        mrc1.header.origin['z'] = 0
        mrc1.header["mapc"] = 1
        mrc1.header["mapr"] = 2
        mrc1.header["maps"] = 3
        mrc1.header.nxstart = np.round(origin[0] / voxel_size[0])
        mrc1.header.nystart = np.round(origin[1] / voxel_size[1])
        mrc1.header.nzstart = np.round(origin[2] / voxel_size[2])
    print('path:', mrc_file)
    return mrc_file


def get_batch_slices(num_total, batch_size):
    if num_total <= batch_size:
        return [slice(0, num_total)]
    num_batches = num_total // batch_size
    batches = [slice(i * batch_size, (i + 1) * batch_size) for i in range(num_batches)]
    if num_total % batch_size > 0:
        batches.append(slice(num_batches * batch_size, num_total))
    return batches


def get_indices(total, windows_size, stride):
    assert (total >= windows_size)
    indices = list(range(0, total - windows_size, stride)) + [total - windows_size]
    return sorted(set(indices))


def make_3d_gaussian(windows_size, sigma=None):
    sigma = sigma or windows_size / 6
    ax = np.arange(windows_size) - windows_size // 2
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel / kernel.max()


def grid_segmentation(grid, windows_size=48, stride=12):
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    if grid.dtype == torch.float64:
        grid = grid.float()
    Z, Y, X = grid.shape
    zs = get_indices(Z, windows_size, stride)
    ys = get_indices(Y, windows_size, stride)
    xs = get_indices(X, windows_size, stride)
    return [grid[i:i + windows_size, j:j + windows_size, k:k + windows_size]
            for i in zs for j in ys for k in xs]


def grid_reconstruction_gaussian_multi(blocks, grid_shape, windows_size=48, stride=12, margin=2):
    B, C, W, _, _ = blocks.shape
    Z, Y, X = grid_shape
    res = np.zeros((C, Z, Y, X), dtype=np.float32)
    weight_sum = np.zeros(grid_shape, dtype=np.float32)
    G = make_3d_gaussian(windows_size)
    G = G[margin:-margin, margin:-margin, margin:-margin]
    zs = get_indices(Z, windows_size, stride)
    ys = get_indices(Y, windows_size, stride)
    xs = get_indices(X, windows_size, stride)
    b_idx = 0
    for i in zs:
        for j in ys:
            for k in xs:
                block = blocks[b_idx, :, margin:-margin, margin:-margin, margin:-margin]
                slc = (
                    slice(i + margin, i + windows_size - margin),
                    slice(j + margin, j + windows_size - margin),
                    slice(k + margin, k + windows_size - margin)
                )
                res[:, slc[0], slc[1], slc[2]] += block * G[None]
                weight_sum[slc] += G
                b_idx += 1
    reconstructed = res / (weight_sum[None] + 1e-8)
    return reconstructed
