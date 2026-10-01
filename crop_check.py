"""Headless crop-inspection projections for CryoCodex."""

from pathlib import Path
import math
import shutil
import textwrap

import numpy as np


BG = '#f5f7fa'
INK = '#17212b'
BLUE = '#245A8D'
GUIDE = '#64a9df'
MUTED = '#667085'
FRAME = '#526477'
VIEW = '#d6e3f0'

INFO_X_SHIFT = -5
INFO_PADDING_X = 30
INFO_PADDING_Y = 30
LAYOUT_GAP = 12

MINOR_TICK = 8
MAJOR_TICK = 15
TICK_NUMBER_GAP = 10
AXIS_NAME_GAP = 30


def mask_bounds(mask, pad=5):
    fg = mask > 0
    out = {}
    for axis, name in enumerate('zyx'):
        other = tuple(i for i in range(3) if i != axis)
        idx = np.flatnonzero(fg.any(axis=other))
        out[name] = (
            (max(0, int(idx[0]) - pad), min(mask.shape[axis] - 1, int(idx[-1]) + pad))
            if idx.size else (0, mask.shape[axis] - 1)
        )
    return out


def validate_bounds(bounds, shape):
    if not isinstance(bounds, dict) or set(bounds) != set('xyz'):
        raise ValueError('Provide X, Y, and Z ranges.')
    out = {}
    for axis, size in zip('xyz', shape[::-1]):
        lo, hi = bounds[axis]
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < size:
            raise ValueError(f'{axis.upper()}: require 0 <= min <= max <= {size - 1}.')
        out[axis] = (lo, hi)
    return out


def parse_range(text, current, size):
    if not text.strip():
        return current
    parts = text.strip().split(',')
    if len(parts) != 2:
        raise ValueError("Enter the range [min,max], e.g. 0,100.")
    try:
        lo, hi = map(int, parts)
    except ValueError as e:
        raise ValueError('Range values must be integers.') from e
    if not 0 <= lo <= hi < size:
        raise ValueError(f'Require 0 <= min <= max <= {size - 1}.')
    return lo, hi


def make_projections(grid):
    if grid.ndim != 3 or any(n == 0 for n in grid.shape):
        raise ValueError('Density must be a nonempty 3D array.')
    return {
        'xy': np.max(grid, axis=0),
        'xz': np.max(grid, axis=1),
        'yz': np.max(grid, axis=2),
    }


def density_thresholds(grid):
    if not np.isfinite(grid).all():
        raise ValueError('Density contains nonfinite values.')
    q99, q999 = np.quantile(grid, [.99, .999])
    return [
        ('top_0p1pct', 'Top 1/1000 voxels', float(q999)),
        ('top_1pct', 'Top 1/100 voxels', float(q99)),
    ]


def _font(size, bold=False):
    from PIL import ImageFont
    names = ('DejaVuSans-Bold.ttf', 'arialbd.ttf') if bold else ('DejaVuSans.ttf', 'arial.ttf')
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _label_step(axis_size, pixel_span, font):
    labels = range(0, axis_size, 10)
    max_w = max((font.getbbox(str(v))[2] for v in labels), default=20)
    px_per_10 = pixel_span * 10 / max(axis_size, 1)
    return 10 * max(1, math.ceil((max_w + 14) / max(px_per_10, 1)))


def _rotated_text(text, font, fill):
    from PIL import Image, ImageDraw
    box = font.getbbox(text)
    img = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8))
    ImageDraw.Draw(img).text((4, 4), text, font=font, fill=fill)
    return img.rotate(90, expand=True)


def _render_panel(view, density, bounds, threshold, scale, position,
                  show_u=True, show_v=True):
    from PIL import Image, ImageDraw

    u, v = view
    h, w = density.shape
    finite = density[np.isfinite(density) & (density > 0)]
    high = float(np.quantile(finite, .995)) if finite.size else 1.0
    values = np.nan_to_num(density, nan=0.0, posinf=high, neginf=0.0)
    visible = np.isfinite(density) & (density >= threshold)

    gray = np.zeros_like(density, dtype=np.uint8)
    if high <= threshold:
        gray[visible] = 255
    else:
        x = np.clip((values[visible] - threshold) / (high - threshold), 0, 1)
        gray[visible] = (55 + 200 * np.sqrt(x)).astype(np.uint8)

    rgb = np.repeat(gray[..., None], 3, axis=2)
    u0, u1 = bounds[u]
    v0, v1 = bounds[v]
    rgb[:v0] //= 3
    rgb[v1 + 1:] //= 3
    rgb[v0:v1 + 1, :u0] //= 3
    rgb[v0:v1 + 1, u1 + 1:] //= 3

    pw, ph = round(w * scale), round(h * scale)
    tick_font = _font(24, True)
    axis_font = _font(24, True)
    max_tick_w = max(tick_font.getbbox(str(v))[2] for v in range(0, max(w, h), 10))
    tick_h = max(tick_font.getbbox(str(v))[3] - tick_font.getbbox(str(v))[1]
                 for v in range(0, max(w, h), 10))

    number_offset = MAJOR_TICK + TICK_NUMBER_GAP
    horizontal_title_offset = number_offset + tick_h + AXIS_NAME_GAP
    vertical_title_offset = number_offset + max_tick_w + AXIS_NAME_GAP

    left = max(120 if position == 'bl' else 155, vertical_title_offset + 45)
    right = max(120, vertical_title_offset + 45)
    top = max(100 if position == 'tr' else 45,
              horizontal_title_offset + 35 if position == 'tr' else 45)
    bottom = max(45 if position == 'tr' else 100,
                 horizontal_title_offset + 35 if position != 'tr' else 45)

    canvas = Image.new('RGB', (pw + left + right, ph + top + bottom), BG)
    canvas.paste(
        Image.fromarray(np.flipud(rgb)).resize((pw, ph), Image.Resampling.NEAREST),
        (left, top),
    )
    draw = ImageDraw.Draw(canvas)

    sx, sy = pw / w, ph / h
    px = lambda i: left + (i + .5) * sx
    py = lambda i: top + ph - (i + .5) * sy

    outer_top = position == 'tr'
    outer_right = position in ('tr', 'br')
    axis_y = top if outer_top else top + ph
    axis_x = left + pw if outer_right else left
    yd = -1 if outer_top else 1
    xd = 1 if outer_right else -1

    draw.rectangle((left, top, left + pw, top + ph), outline=FRAME, width=2)

    u_step = _label_step(w, pw, tick_font)
    v_step = _label_step(h, ph, tick_font)

    for tick in range(0, w, 10):
        x = px(tick)
        major = tick % u_step == 0
        length = MAJOR_TICK if major else MINOR_TICK
        draw.line((x, axis_y, x, axis_y + yd * length), fill=INK, width=2)
        if major:
            draw.text(
                (x, axis_y + yd * number_offset), str(tick),
                anchor='mb' if outer_top else 'mt', font=tick_font, fill=INK,
            )

    for tick in range(0, h, 10):
        y = py(tick)
        major = tick % v_step == 0
        length = MAJOR_TICK if major else MINOR_TICK
        draw.line((axis_x, y, axis_x + xd * length, y), fill=INK, width=2)
        if major:
            draw.text(
                (axis_x + xd * number_offset, y), str(tick),
                anchor='lm' if outer_right else 'rm', font=tick_font, fill=INK,
            )

    draw.text(
        (left + pw / 2, axis_y + yd * horizontal_title_offset),
        f'{u.upper()} (voxel)',
        anchor='mb' if outer_top else 'mt', font=axis_font, fill=INK,
    )

    vlabel = _rotated_text(f'{v.upper()} (voxel)', axis_font, INK)
    if outer_right:
        title_x = axis_x + vertical_title_offset
    else:
        title_x = axis_x - vertical_title_offset - vlabel.width
    canvas.paste(vlabel, (round(title_x), round(top + ph / 2 - vlabel.height / 2)), vlabel)

    x0, x1 = px(u0), px(u1)
    y0, y1 = py(v1), py(v0)
    for x in (x0, x1):
        for y in np.arange(y0, y1 + 1, 18):
            draw.line((x, y, x, min(y + 11, y1)), fill=GUIDE, width=3)
    for y in (y0, y1):
        for x in np.arange(x0, x1 + 1, 18):
            draw.line((x, y, min(x + 11, x1), y), fill=GUIDE, width=3)

    guide_y = top + ph if outer_top else top
    edge_y = y1 if outer_top else y0
    for x in (x0, x1):
        draw.line((x, edge_y, x, guide_y), fill=GUIDE, width=2)

    guide_x = left if outer_right else left + pw
    edge_x = x0 if outer_right else x1
    for y in (y0, y1):
        draw.line((edge_x, y, guide_x, y), fill=GUIDE, width=2)

    endpoint_font = _font(25, True)
    if show_u:
        inner_y = top + ph if outer_top else top
        inward_y = 1 if outer_top else -1
        for sign, value in (('-', u0), ('+', u1)):
            x = px(value)
            draw.line((x, inner_y, x, inner_y + inward_y * 14), fill=BLUE, width=3)
            draw.text(
                (x, inner_y + inward_y * 18), f'{u.upper()}{sign}={value}',
                anchor='mt' if inward_y > 0 else 'mb', font=endpoint_font, fill=BLUE,
            )

    if show_v:
        inner_x = left if outer_right else left + pw
        inward_x = -1 if outer_right else 1
        for sign, value in (('+', v1), ('-', v0)):
            y = py(value)
            draw.line((inner_x, y, inner_x + inward_x * 15, y), fill=BLUE, width=3)
            draw.text(
                (inner_x + inward_x * 19, y), f'{v.upper()}{sign}={value}',
                anchor='rm' if inward_x < 0 else 'lm', font=endpoint_font, fill=BLUE,
            )

    draw.text(
        (left + pw - 14, top + ph - 12), view.upper(),
        anchor='rb', font=_font(31, True), fill=VIEW,
    )

    return canvas, {
        'left': left, 'top': top, 'right': left + pw, 'bottom': top + ph,
        'sx': sx, 'sy': sy, 'ph': ph,
    }



def _label_on_bg(draw, xy, text, font):
    box = draw.textbbox(xy, text, font=font, anchor='mm')
    pad_x, pad_y = 5, 3
    draw.rectangle((box[0] - pad_x, box[1] - pad_y,
                    box[2] + pad_x, box[3] + pad_y), fill=BG)
    draw.text(xy, text, anchor='mm', font=font, fill=BLUE)


def _shared_labels(draw, bounds, xy_g, yz_g, xz_g, xy_o, yz_o, xz_o):
    font = _font(25, True)

    y_top = xy_o[1] + xy_g['bottom']
    y_bottom = xz_o[1] + xz_g['top']
    y_mid = (y_top + y_bottom) / 2
    for sign, value in (('-', bounds['x'][0]), ('+', bounds['x'][1])):
        x = xy_o[0] + xy_g['left'] + (value + .5) * xy_g['sx']
        draw.line((x, y_top, x, y_bottom), fill=GUIDE, width=2)
        _label_on_bg(draw, (x, y_mid), f'X{sign}={value}', font)

    x_left = yz_o[0] + yz_g['right']
    x_right = xz_o[0] + xz_g['left']
    x_mid = (x_left + x_right) / 2
    for sign, value in (('+', bounds['z'][1]), ('-', bounds['z'][0])):
        y = yz_o[1] + yz_g['top'] + yz_g['ph'] - (value + .5) * yz_g['sy']
        draw.line((x_left, y, x_right, y), fill=GUIDE, width=2)
        _label_on_bg(draw, (x_mid, y), f'Z{sign}={value}', font)


def _info_items(bounds, voxel_size, threshold_label, threshold):
    spacing = '/'.join(f'{float(v):g}' for v in voxel_size)
    items = [
        (0, 'CryoCodex', _font(54, True), BLUE, False),
        (68, 'Crop inspection', _font(34, True), INK, False),
        (118, threshold_label, _font(28), MUTED, False),
        (158, f'Threshold: {threshold:.6g}', _font(27), MUTED, False),
    ]
    for i, axis in enumerate('xyz'):
        lo, hi = bounds[axis]
        items.append((220 + i * 54,
                      f'{axis.upper()}- = {lo}    {axis.upper()}+ = {hi}',
                      _font(37, True), BLUE, False))
    items += [
        (390, f'Voxel XYZ: {spacing} Å', _font(27), MUTED, False),
        (438, textwrap.fill('Cropping removes background to accelerate inference.', width=38),
         _font(26), MUTED, True),
        (500, textwrap.fill('Please check that the blue dashed box fully encloses the molecule.', width=38),
         _font(26, True), BLUE, True),
    ]
    return items


def _info_size(items):
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new('RGB', (1, 1), BG))
    right = bottom = 0
    for y, text, font, _, multiline in items:
        box = (draw.multiline_textbbox((0, y), text, font=font, spacing=5)
               if multiline else draw.textbbox((0, y), text, font=font))
        right, bottom = max(right, box[2]), max(bottom, box[3])
    return right + 2 * INFO_PADDING_X, bottom + 2 * INFO_PADDING_Y


def _draw_info(draw, x, y, items):
    for dy, text, font, fill, multiline in items:
        if multiline:
            draw.multiline_text((x, y + dy), text, font=font, fill=fill, spacing=5)
        else:
            draw.text((x, y + dy), text, font=font, fill=fill)


def render_projections(projections, bounds, voxel_size, directory, revision, thresholds):
    from PIL import Image, ImageDraw

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scale = max(2.5, 580 / max(max(p.shape) for p in projections.values()))
    files = []

    for suffix, threshold_label, threshold in thresholds:
        xy, xy_g = _render_panel('xy', projections['xy'], bounds, threshold, scale, 'tr', False, True)
        yz, yz_g = _render_panel('yz', projections['yz'], bounds, threshold, scale, 'bl', True, False)
        xz, xz_g = _render_panel('xz', projections['xz'], bounds, threshold, scale, 'br', False, False)

        items = _info_items(bounds, voxel_size, threshold_label, threshold)
        info_w, info_h = _info_size(items)
        margin = 5
        left_width = max(yz.width, info_w)
        top_height = max(xy.height, info_h)
        column2 = margin + left_width + LAYOUT_GAP
        row2 = margin + top_height + LAYOUT_GAP

        canvas = Image.new(
            'RGB',
            (column2 + max(xy.width, xz.width) + margin,
             row2 + max(yz.height, xz.height) + margin),
            BG,
        )

        xy_o = (column2, margin)
        yz_o = (column2 - LAYOUT_GAP - yz.width, row2)
        xz_o = (column2, row2)
        for panel, origin in ((xy, xy_o), (yz, yz_o), (xz, xz_o)):
            canvas.paste(panel, origin)

        draw = ImageDraw.Draw(canvas)
        _shared_labels(draw, bounds, xy_g, yz_g, xz_g, xy_o, yz_o, xz_o)

        content_w = info_w - 2 * INFO_PADDING_X
        content_h = info_h - 2 * INFO_PADDING_Y
        lx = margin + max(INFO_PADDING_X, (left_width - content_w) // 2) + INFO_X_SHIFT
        ly = margin + max(INFO_PADDING_Y, (top_height - content_h) // 2)
        _draw_info(draw, lx, ly, items)

        path = directory / f'projections_{suffix}.png'
        canvas.save(path)
        files.append(path)

    return files

def review_crop(grid, mask, voxel_size, directory):
    bounds = mask_bounds(mask)
    projections = make_projections(grid)
    thresholds = density_thresholds(grid)
    directory = Path(directory).resolve()
    revision = 0
    folders = []

    def write_images():
        print(f'\n{"=" * 12} Round {revision + 1} check {"=" * 12}', flush=True)
        folder = directory / f'round_{revision + 1}_check'
        folders.append(folder)
        paths = render_projections(projections, bounds, voxel_size, folder, revision, thresholds)
        print('\nCrop inspection images:', flush=True)
        for path in paths:
            print(f'  {path}', flush=True)
        print('Current ranges: ' + ', '.join(
            f'{a.upper()}-={bounds[a][0]} {a.upper()}+={bounds[a][1]}' for a in 'xyz'
        ), flush=True)

    write_images()
    print('Inspect the blue dashed box and edit only if necessary.', flush=True)

    try:
        while True:
            action = input('[Enter] Continue without changes  |  [e] Edit bounds: ').strip().lower()
            if action in ('', 'c'):
                print('No changes made. Continuing with the current crop bounds.', flush=True)
                print('\n=========== Crop check completed ===========', flush=True)
                for folder in folders:
                    shutil.rmtree(folder, ignore_errors=True)
                return bounds
            if action != 'e':
                print('Press Enter to continue, or enter e to edit the crop bounds.', flush=True)
                continue

            print("Edit crop bounds: enter two integers as min,max (e.g. 0,100).", flush=True)
            updated = dict(bounds)
            for axis, size in zip('xyz', grid.shape[::-1]):
                while True:
                    try:
                        current = bounds[axis]
                        print(
                            f'\nCurrent {axis.upper()} range: {current[0]},{current[1]}',
                            flush=True,
                        )
                        value = input(
                            f'New {axis.upper()} range (press Enter to keep current): '
                        )
                        updated[axis] = parse_range(value, current, size)
                        break
                    except ValueError as e:
                        print(e, flush=True)
            bounds = validate_bounds(updated, grid.shape)
            revision += 1
            write_images()

    except EOFError as e:
        raise RuntimeError('Crop inspection needs terminal input.') from e
