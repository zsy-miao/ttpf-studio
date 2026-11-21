"""Texture codec for TTPF: AMD-style GPU swizzle <-> linear, Pillow BCn decode, DDS in/out.
Needs numpy + Pillow.

Swizzle: inside each GPU tile an element's address bits interleave x/y in a fixed order
(a 256-byte micro tile that depends on element size, then alternating y/x). Recovered by
searching for the bit order that gives the smoothest images.
Tiles are 64 KB when mip 0 is >= 64 KB, else 4 KB; tiles are laid out row-major.
"""
import io
import struct

import numpy as np
from PIL import Image

# name: (bytes per element, texels per element side, pillow 'bcn' args or None)
FORMATS = {
    'RGBA8': (4, 1, None),
    'BGRA8': (4, 1, None),
    'RGBA16F': (8, 1, None),
    'BC1': (8, 4, ('RGBA', 1)),
    'BC2': (16, 4, ('RGBA', 2)),
    'BC3': (16, 4, ('RGBA', 3)),
    'BC4': (8, 4, ('L', 4)),
    'BC5': (16, 4, ('RGB', 5)),
    'BC6H': (16, 4, ('RGB', 6)),
    'BC7': (16, 4, ('RGBA', 7)),
}

# Descriptor format code >> 3 is a DXGI_FORMAT number.
DXGI = {10: 'RGBA16F', 28: 'RGBA8', 29: 'RGBA8', 71: 'BC1', 72: 'BC1', 74: 'BC2', 75: 'BC2', 77: 'BC3', 78: 'BC3',
        80: 'BC4', 81: 'BC4', 83: 'BC5', 84: 'BC5', 87: 'BGRA8', 91: 'BGRA8', 95: 'BC6H', 96: 'BC6H', 98: 'BC7', 99: 'BC7'}
DXGI_OF = {'RGBA16F': 10, 'RGBA8': 28, 'BC1': 71, 'BC2': 74, 'BC3': 77, 'BC4': 80, 'BC5': 83, 'BGRA8': 87, 'BC6H': 95, 'BC7': 98}
SRGB = {29, 72, 75, 78, 91, 99}
# Rarer formats must look clearly smoother to win.
ASPECT_PENALTY = 1.0  # per halving of the aspect ratio
FORMAT_PRIOR = {'BC7': 1.0, 'BC1': 1.0, 'BC5': 1.05, 'BC4': 1.15, 'BC3': 1.3, 'RGBA8': 1.15}
GUESS_ORDER = ('BC7', 'BC1', 'BC3', 'BC4', 'BC5', 'RGBA8')

# Two micro-tile families. 'S' = 64 KB standard (decals, UI, liveries);
# 'R' = 4 KB tiles with a different micro tile (most character textures). 4-byte R/S only seen as one order.
MICRO = {'S': {4: 'xxyyyx', 8: 'xyxxy', 16: 'xyxy'}, 'R': {4: 'xxyyyx', 8: 'xyyxx', 16: 'yyxx'}}
# 'S' / 'R' alone = tile size picked from mip size (64 KB if the mip is >= 64 KB, else 4 KB)
TILINGS = ('S', 'R4', 'S64', 'S4', 'R64', 'LIN')
TILING_NAMES = {'S': 'Standard (auto)', 'R4': '4 KB R', 'S64': '64 KB S', 'S4': '4 KB S', 'R64': '64 KB R', 'LIN': 'Linear'}


def pattern_for(bpe, tile_bytes, family='S'):
    micro = MICRO[family].get(bpe)
    if micro is None:
        return None  # shortcut: 1/2-byte formats unverified, treated as linear
    return micro + ('yx' * 8)[:(tile_bytes.bit_length() - 1) - 8]


def tiling_from_desc(b):
    """Descriptor word b -> tiling."""
    return 'R4' if (b >> 24) & 3 else 'S'


def coords(pattern):
    idx = np.arange(1 << len(pattern))
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    xi = yi = 0
    for k, c in enumerate(pattern):
        bit = (idx >> k) & 1
        if c == 'x':
            x |= bit << xi
            xi += 1
        else:
            y |= bit << yi
            yi += 1
    return x, y, 1 << xi, 1 << yi


def dims(fmt, w, h):
    bpe, blk, _ = FORMATS[fmt]
    return bpe, max(1, -(-w // blk)), max(1, -(-h // blk))


def mip_bytes(fmt, w, h):
    bpe, ew, eh = dims(fmt, w, h)
    return ew * eh * bpe


def resolve(fmt, w, h, tiling='S'):
    """Tiling name -> bit pattern (None = linear)."""
    bpe, ew, eh = dims(fmt, w, h)
    if tiling == 'LIN':
        return None
    family = tiling[0]
    tile = {'64': 65536, '4': 4096}.get(tiling[1:]) or (65536 if ew * eh * bpe >= 65536 else 4096)
    return pattern_for(bpe, tile, family)


def unswizzle(fmt, data, w, h, tiling='S'):
    """Swizzled bytes -> (eh, ew, bpe) uint8 array in row-major order."""
    bpe, ew, eh = dims(fmt, w, h)
    pattern = resolve(fmt, w, h, tiling)
    el = np.frombuffer(data[:len(data) // bpe * bpe], np.uint8).reshape(-1, bpe)
    if pattern is None:
        need = ew * eh
        if len(el) < need:
            el = np.concatenate([el, np.zeros((need - len(el), bpe), np.uint8)])
        return el[:need].reshape(eh, ew, bpe)
    x, y, tw, th = coords(pattern)
    tx, ty, per = -(-ew // tw), -(-eh // th), len(x)
    grid = np.zeros((ty * th, tx * tw, bpe), np.uint8)
    for t in range(min(tx * ty, len(el) // per)):
        grid[(t // tx) * th + y, (t % tx) * tw + x] = el[t * per:(t + 1) * per]
    if len(el) < tx * ty * per and len(el) % per:  # partial last tile
        t = len(el) // per
        n = len(el) % per
        grid[(t // tx) * th + y[:n], (t % tx) * tw + x[:n]] = el[t * per:]
    return grid[:eh, :ew]


def swizzle(fmt, lin, w, h, old, tiling='S'):
    """Inverse of unswizzle. Writes into a copy of `old` so tile padding / mip tail bytes are kept."""
    bpe, ew, eh = dims(fmt, w, h)
    pattern = resolve(fmt, w, h, tiling)
    if isinstance(lin, (bytes, bytearray, memoryview)):
        lin = np.frombuffer(lin, np.uint8)
    lin = np.asarray(lin, np.uint8).reshape(eh, ew, bpe)
    if pattern is None:
        flat = lin.reshape(-1).tobytes()
        return flat + old[len(flat):]
    x, y, tw, th = coords(pattern)
    tx, ty, per = -(-ew // tw), -(-eh // th), len(x)
    grid = np.zeros((ty * th, tx * tw, bpe), np.uint8)
    grid[:eh, :ew] = lin
    out = bytearray(old)
    need = tx * ty * per * bpe
    if len(out) < need:
        out += bytes(need - len(out))
    arr = np.frombuffer(out, np.uint8).reshape(-1, bpe).copy()
    for t in range(tx * ty):
        arr[t * per:(t + 1) * per] = grid[(t // tx) * th + y, (t % tx) * tw + x]
    return arr.tobytes()[:max(len(old), need)]


def lin_to_image(fmt, lin, w, h):
    bpe, blk, pil = FORMATS[fmt]
    eh, ew = lin.shape[:2]
    if fmt == 'RGBA16F':
        f = np.nan_to_num(np.ascontiguousarray(lin).view('<f2').reshape(eh, ew, 4).astype(np.float32))
        img = Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA')  # shortcut: HDR clipped to 0..1
    elif fmt == 'BGRA8':
        img = Image.fromarray(np.ascontiguousarray(lin.reshape(eh, ew, 4)[..., [2, 1, 0, 3]]), 'RGBA')
    elif pil is None:
        img = Image.frombytes('RGBA', (ew, eh), lin.tobytes())
    else:
        img = Image.frombytes(pil[0], (ew * blk, eh * blk), lin.tobytes(), 'bcn', pil[1])
    img = img.crop((0, 0, w, h)) if img.size != (w, h) else img
    return img.convert('RGBA')


def decode(fmt, data, w, h, tiling='S'):
    return lin_to_image(fmt, unswizzle(fmt, data, w, h, tiling), w, h)


def roughness(img):
    a = np.asarray(img.convert('RGB'), np.float32)
    if a.shape[0] < 2 or a.shape[1] < 2:
        return 0.0
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def guess_sizes(fmt, n):
    bpe, blk, _ = FORMATS[fmt]
    texels = 1 << ((n // bpe * blk * blk).bit_length() - 1)  # padded blobs (mip tail / tile padding): largest power of two that fits
    out, side = [], 4
    while side <= 16384:
        for r in (1, 2, 4, 8):
            for ww, hh in ((side, side // r), (side // r, side)):
                if ww >= 4 and hh >= 4 and ww * hh == texels and (ww, hh) not in out:
                    out.append((ww, hh))
        side *= 2
    return out


def guess(data):
    """(fmt, w, h, tiling) for a blob with no descriptor: the smoothest-looking candidate wins.
    shortcut: picks by image smoothness, so flat/noisy textures can be misread; the UI lets the user override."""
    best = None
    for fmt in GUESS_ORDER:
        for ww, hh in guess_sizes(fmt, len(data)):
            for tiling in ('S', 'R4'):
                # decode only the top rows (tiles are row-major, so they come first); pin the tile size the full texture would use
                til = tiling if tiling != 'S' else ('S64' if mip_bytes(fmt, ww, hh) >= 65536 else 'S4')
                try:
                    img = decode(fmt, data, ww, min(hh, 256), til)
                except Exception:
                    continue
                # score a native-resolution crop: downscaling would hide misplaced tiles
                score = roughness(img.crop((0, 0, min(ww, 512), min(hh, 256))))
                aspect = abs((ww // hh or 1).bit_length() - (hh // ww or 1).bit_length())
                score *= FORMAT_PRIOR[fmt] * (1 + ASPECT_PENALTY * aspect) * (1.02 if tiling != 'S' else 1)
                if best is None or score < best[0]:
                    best = (score, fmt, ww, hh, tiling)
    return best[1:] if best else None

# ------------------------------------------------------------------ DDS

def dds_bytes(fmt, w, h, lin, dxgi=None):
    dxgi = dxgi or DXGI_OF[fmt]
    data = np.ascontiguousarray(lin).tobytes()
    pf = struct.pack('<II4s5I', 32, 0x4, b'DX10', 0, 0, 0, 0, 0)
    hdr = struct.pack('<7I44x', 124, 0x81007, h, w, len(data), 0, 1) + pf + struct.pack('<5I', 0x1000, 0, 0, 0, 0)
    return b'DDS ' + hdr + struct.pack('<5I', dxgi, 3, 0, 1, 0) + data


FOURCC = {b'DXT1': 'BC1', b'DXT2': 'BC2', b'DXT3': 'BC2', b'DXT4': 'BC3', b'DXT5': 'BC3',
          b'ATI1': 'BC4', b'BC4U': 'BC4', b'ATI2': 'BC5', b'BC5U': 'BC5'}


def dds_read(raw):
    """-> (fmt, w, h, [mip level bytes...]) with mip data in linear (row-major) order."""
    if raw[:4] != b'DDS ':
        raise ValueError('Not a DDS file')
    h, w = struct.unpack_from('<II', raw, 12)
    mips = max(1, struct.unpack_from('<I', raw, 28)[0])
    pf_flags, fourcc, bits, rmask = struct.unpack_from('<I4sII', raw, 80)
    off = 128
    if fourcc == b'DX10':
        dx = struct.unpack_from('<I', raw, 128)[0]
        fmt = DXGI.get(dx)
        off = 148
        if not fmt:
            raise ValueError(f'Unsupported DXGI format {dx}')
    elif fourcc in FOURCC:
        fmt = FOURCC[fourcc]
    elif bits == 32 and pf_flags & 0x40:
        fmt = 'RGBA8' if rmask == 0xFF else 'BGRA8'
    else:
        raise ValueError(f'Unsupported DDS pixel format {fourcc!r}/{bits}bpp')
    levels = []
    for k in range(mips):
        n = mip_bytes(fmt, max(1, w >> k), max(1, h >> k))
        if off + n > len(raw):
            break
        levels.append(raw[off:off + n])
        off += n
    return fmt, w, h, levels


def image_to_lin(fmt, img):
    """Encode a PIL image to linear element bytes for the formats Pillow can encode."""
    w, h = img.size
    if fmt == 'RGBA8':
        return np.asarray(img.convert('RGBA'), np.uint8).tobytes()
    if fmt == 'BGRA8':
        return np.ascontiguousarray(np.asarray(img.convert('RGBA'), np.uint8)[..., [2, 1, 0, 3]]).tobytes()
    pf = {'BC1': 'DXT1', 'BC2': 'DXT3', 'BC3': 'DXT5', 'BC5': 'BC5'}.get(fmt)
    if not pf:
        raise ValueError(f'No {fmt} encoder available: export as DDS, edit in Paint.NET / GIMP / texconv, '
                         f'save as {fmt} DDS and use Import DDS.')
    if w % 4 or h % 4:
        raise ValueError('Block-compressed images need sizes that are multiples of 4.')
    b = io.BytesIO()
    (img.convert('RGB') if fmt == 'BC5' else img.convert('RGBA')).save(b, 'DDS', pixel_format=pf)
    return dds_read(b.getvalue())[3][0]
