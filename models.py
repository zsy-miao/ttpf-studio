"""3D models from RPF9 archives (RAGE "gen9" resources): find them, decode their meshes, export OBJ.

A resource is a run of consecutive zlib streams: first its "virtual" chunks (structures, names), then
its "physical" chunks (GPU buffers). Pointers 0x50000000+ address the joined virtual data, 0x60000000+
the joined physical data. The root's +0x08 points to a block map whose bytes +8 / +9 hold the number of
virtual / physical chunks.

Drawable (+0x50) -> LOD list -> models (+0x08 array, +0x10 count) -> geometries. A geometry has the
marker 0x1601 at +0x60, its vertex buffer (physical) at +0x68 and its size in 4-byte words at +0x58;
+0x40 leads to its meshlet buffer. Vertex buffers start with stream descriptors (offset / 2 << 8 | type);
positions are packed in 8 bytes: two 24-bit values and one 16-bit value, scaled from an origin (+0x60)
by a step (+0x70); +0x6C says which axis is in which slot. Meshlets store triangles as three 7-bit local
indices per 32-bit word, mapped to vertices through a 16-bit table.

Format knowledge credit: the 6Stars project (github.com/Ahmet-Dedeler/6Stars) documented this layout.
This file is an independent implementation.
"""
import struct
import zlib

GEOM_MARK = 0x1601


def u16(b, o):
    return struct.unpack_from('<H', b, o)[0]


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def u64(b, o):
    return struct.unpack_from('<Q', b, o)[0]


def f32(b, o):
    return struct.unpack_from('<f', b, o)[0]


def vptr(q):
    return q - 0x50000000 if 0x50000000 <= q < 0x58000000 else None


def pptr(q):
    return q - 0x60000000 if 0x60000000 <= q < 0x70000000 else None


def inflate(m, off, limit=64 << 20):
    d = zlib.decompressobj()
    out, q, n = [], off, 0
    while q < len(m):
        chunk = m[q:q + (1 << 16)]
        q += len(chunk)
        try:
            o = d.decompress(chunk)
        except zlib.error:
            return None
        out.append(o)
        n += len(o)
        if d.eof:
            return b''.join(out)
        if n > limit:
            return None
    return None


def stream_offsets(items):
    """Every zlib stream of an archive in file order (an item may span several 64 KB streams)."""
    offs = sorted(off for it in items if it.kind != 'video' for off, _ in it.parts)
    return offs


def _virtual(m, streams, i):
    """Join a resource's virtual chunks starting at stream i -> (virtual, next stream, physical chunk count) or None."""
    if i >= len(streams):
        return None
    v = inflate(m, streams[i])
    if not v or len(v) < 0x80 or u64(v, 0x60) == GEOM_MARK:
        return None
    bm = vptr(u64(v, 8))
    if bm is None or bm > (32 << 20):
        return None
    k = i + 1
    while len(v) <= bm + 10 and k < len(streams) and k - i < 32:
        nxt = inflate(m, streams[k])
        if nxt is None:
            return None
        v += nxt
        k += 1
    if bm + 10 > len(v):
        return None
    v_cnt, p_cnt = v[bm + 8], v[bm + 9]
    if not (1 <= v_cnt <= 32 and 1 <= p_cnt <= 64):
        return None
    while k - i < v_cnt and k < len(streams):
        v += inflate(m, streams[k]) or b''
        k += 1
    return v, k, p_cnt


def read_resource(m, streams, i):
    """Join a resource starting at stream index i -> (virtual, physical, streams used) or None."""
    r = _virtual(m, streams, i)
    if not r:
        return None
    v, k, p_cnt = r
    p = b''
    for _ in range(p_cnt):
        if k >= len(streams):
            break
        p += inflate(m, streams[k]) or b''
        k += 1
    return v, p, k - i


def model_name(v):
    for ext in (b'.#dr', b'.#ft', b'.#dd'):
        j = v.find(ext)
        if j > 1:
            s = j
            while s > 0 and (chr(v[s - 1]).isalnum() or v[s - 1] in b'_-'):
                s -= 1
            if 2 <= j - s <= 96 and v[s:j].isascii():
                return v[s:j].decode('ascii') + ext.decode()
    if len(v) >= 0xb0:
        no = vptr(u64(v, 0xa8))
        if no is not None and no < len(v):
            end = v.find(b'\0', no, no + 100)
            s = v[no:end]
            if 2 <= len(s) <= 96 and all(chr(c).isalnum() or c in b'_-.#' for c in s):
                return s.decode()
    return None


def meshlet_buffer(v, link):
    lo = vptr(link)
    if lo is None or lo + 8 > len(v):
        return None
    mo = vptr(u64(v, lo))
    if mo is None or mo + 0x20 > len(v):
        return None
    mb = pptr(u64(v, mo + 0x18))
    return None if mb is None else mb


def drawable_bases(v):
    """Where drawables start: the root itself, a fragment's drawable (+0x18), and the first entry of a
    drawable dictionary (+0x30 array, u16 count at +0x38; peds and clothing use these)."""
    out = [0]
    if len(v) >= 0x40:
        if vptr(u64(v, 0x18)) is not None:
            out.append(vptr(u64(v, 0x18)))
        arr, n = vptr(u64(v, 0x30)), u16(v, 0x38)
        if arr is not None and 0 < n <= 4096 and arr + 8 <= len(v) and vptr(u64(v, arr)) is not None:
            out.append(vptr(u64(v, arr)))  # shortcut: entries after the first are not shown
    return out


def geometries(v):
    """LOD0 geometries as (vertex buffer offset, vertex buffer bytes, meshlet buffer offset, drawable base, shader index),
    else every geometry found (base / shader None). Shadow-only copies are left out."""
    seen = set()
    for base in drawable_bases(v):
        if base + 0x58 > len(v):
            continue
        lod = vptr(u64(v, base + 0x50))
        if lod is None or lod + 0x10 > len(v):
            continue
        marr, mcnt = vptr(u64(v, lod)), u16(v, lod + 8)
        if marr is None or not (0 < mcnt <= 128) or marr + mcnt * 8 > len(v):
            continue
        found = []
        for mi in range(mcnt):
            mp = vptr(u64(v, marr + mi * 8))
            if mp is None or mp + 0x38 > len(v):
                continue
            garr, gcnt = vptr(u64(v, mp + 8)), u16(v, mp + 0x10)
            if garr is None or not (0 < gcnt <= 256) or garr + gcnt * 8 > len(v):
                continue
            smap = vptr(u64(v, mp + 0x20))
            if smap is not None and smap + gcnt * 2 > len(v):
                smap = None
            mask = v[mp + 0x34]  # bit 0: drawn in the main pass (others are shadow / auxiliary copies)
            for gi in range(gcnt):
                gp = vptr(u64(v, garr + gi * 8))
                if gp is None or gp + 0x70 > len(v) or u64(v, gp + 0x60) != GEOM_MARK:
                    continue
                vb, mb = pptr(u64(v, gp + 0x68)), meshlet_buffer(v, u64(v, gp + 0x40))
                if vb is not None and mb is not None:
                    found.append((vb, u32(v, gp + 0x58) * 4, mb, base, u16(v, smap + gi * 2) if smap is not None else gi, mask))
        if any(f[5] & 1 for f in found):
            found = [f for f in found if f[5] & 1]
        out = []
        for f in found:
            if (f[0], f[2]) not in seen:
                seen.add((f[0], f[2]))
                out.append(f[:5])
        if out:
            return out
    out = []
    for off in range(0x28, len(v) - 0x10, 8):  # fallback: every geometry marker in the resource
        if u64(v, off - 8) == GEOM_MARK:
            vb, mb = pptr(u64(v, off)), meshlet_buffer(v, u64(v, off - 0x28))
            if vb is not None and mb is not None and (vb, mb) not in seen:
                seen.add((vb, mb))
                out.append((vb, u32(v, off - 0x10) * 4, mb, None, None))
    return out


# shader parameter name hashes (joaat) for colour maps, best first
COLOUR_SLOTS = (0xc8e8c282, 0x346394ea, 0x0df47048)  # diffuseTex, vehicle body/detail slot, diffuseTex2
TINT_PALETTE = 0x2c8d7182
GLASS_SHADER = 0xe3ba3129
STREAMED = 0x980  # texture object that refers to a TTPF texture by 64-bit hash


def shader_info(v, base, idx):
    """-> dict(tex=hash or None, alpha=bool, glass=bool, paint=bool) for shader idx of the drawable at base."""
    info = dict(tex=None, alpha=False, glass=False, paint=False)
    if base is None or idx is None or base + 0x18 > len(v):
        return info
    sg = vptr(u64(v, base + 0x10))
    if sg is None or sg + 0x20 > len(v):
        return info
    sp, n = vptr(u64(v, sg + 0x10)), u16(v, sg + 0x18)
    if sp is None or idx >= n or sp + n * 8 > len(v):
        return info
    sh = vptr(u64(v, sp + idx * 8))
    if sh is None or sh + 0x40 > len(v) or v[sh + 4:sh + 8] != b'atem':
        return info
    pass_ = v[sh + 0x3c]
    info['alpha'] = pass_ in (2, 4, 8)
    info['glass'] = pass_ == 2 and u32(v, sh) == GLASS_SHADER
    params, slots = vptr(u64(v, sh + 0x20)), vptr(u64(v, sh + 0x10))
    if params is None or slots is None or params + 8 > len(v):
        return info
    found = {}
    for k in range(v[params + 1]):
        e = params + 8 + k * 8
        if e + 8 > len(v):
            break
        name, code = u32(v, e), u32(v, e + 4)
        if name == TINT_PALETTE:
            info['paint'] = True
        if name not in COLOUR_SLOTS or code % 4 or slots + code * 2 + 8 > len(v):
            continue
        tp = vptr(u64(v, slots + code * 2))
        if tp is None or tp + 0x38 > len(v) or u64(v, tp + 0x10) != STREAMED:
            continue
        ref = vptr(u64(v, tp + 0x30))
        if ref is not None and ref + 16 <= len(v):
            found[name] = u64(v, ref + 8)
    info['tex'] = next((found[n] for n in COLOUR_SLOTS if n in found), None)
    return info


def texture_index(folder, progress=lambda f: None):
    """{64-bit texture hash: (pack path, texture number)} for every .ttpf under folder.
    A pack's ETPF table is followed by its textures' hashes, sorted, in texture order."""
    import os
    packs = []
    for d, _, fs in os.walk(folder):
        packs += [os.path.join(d, f) for f in fs if f.lower().endswith('.ttpf')]
    out = {}
    for k, path in enumerate(sorted(packs)):
        progress(k / max(1, len(packs)))
        try:
            with open(path, 'rb') as f:
                head = f.read(0x400000)
        except OSError:
            continue
        at = head.find(b'ETPF', 0x18)
        if at < 0 or at + 16 > len(head):
            continue
        n = u32(head, at + 8)
        hb = at + 16 + n * 16
        if not 0 < n < 200000 or hb + n * 8 > len(head):
            continue
        hs = struct.unpack_from(f'<{n}Q', head, hb)
        if any(hs[j] > hs[j + 1] for j in range(n - 1)):
            continue  # not a sorted hash list: some other layout
        for j, h in enumerate(hs):
            out.setdefault(h, (path, j))
    return out


AXIS_SLOTS = {0: (2, 1, 0), 1: (2, 0, 1)}  # which world axis each packed slot holds (default x, y, z)


def decode_geometry(p, vb, vb_size, mb, verts, uvs, tris):
    """Append one geometry's vertices / UVs / triangles. Returns the number of triangles added."""
    if mb + 0x10 > len(p) or vb + 0x80 > len(p):
        return 0
    n_meshlets, n_tris, n_refs = u32(p, mb) & 0xFFFF, u32(p, mb + 4), u32(p, mb + 8)
    if not (n_meshlets and n_tris and n_refs):
        return 0
    tri_base = mb + 0x0C + n_meshlets * 16
    remap = tri_base + n_tris * 4
    if remap + n_refs * 2 > len(p):
        return 0
    desc = [u32(p, vb + 4 * k) for k in range(8)]
    pos_off = (desc[0] >> 8) << 1
    raw = desc[0] & 0xFF == 0x63  # plain float3 positions (glass and other small parts), not packed
    stride = 12 if raw else 8
    pos_end, uv = vb_size, None
    for h in desc[1:]:
        if not h:
            continue
        so, kind = (h >> 8) << 1, h & 0xFF
        if pos_off < so < pos_end:
            pos_end = so
        if so > pos_off and uv is None and kind in (0x28, 0x42):
            uv = (so, 4 if kind == 0x28 else 8)
    max_v = min((pos_end - pos_off) // stride, (len(p) - vb - pos_off) // stride)
    if max_v <= 0:
        return 0
    org = struct.unpack_from('<3f', p, vb + 0x60)
    step = struct.unpack_from('<3f', p, vb + 0x70)
    slots = AXIS_SLOTS.get(u32(p, vb + 0x6C), (0, 1, 2))
    base = len(verts)
    used, added, new_tris = -1, 0, []
    for mi in range(n_meshlets):
        mo = mb + 0x0C + mi * 16
        v0, t0 = u16(p, mo + 4), u16(p, mo + 6)
        t1 = u16(p, mo + 16 + 6) if mi + 1 < n_meshlets else n_tris
        for ti in range(t0, min(t1, n_tris)):
            w = u32(p, tri_base + ti * 4)
            r = (v0 + (w & 0x7F), v0 + ((w >> 7) & 0x7F), v0 + ((w >> 14) & 0x7F))
            if max(r) >= n_refs:
                continue
            i = [u16(p, remap + x * 2) for x in r]
            if max(i) < max_v and len(set(i)) == 3:
                new_tris.append((base + i[0], base + i[1], base + i[2]))
                used = max(used, *i)
                added += 1
    if not added:
        return 0
    start, new_verts = vb + pos_off, []
    for k in range(used + 1):
        o = start + k * stride
        if raw:
            new_verts.append(struct.unpack_from('<3f', p, o))
            continue
        a = p[o + 1] | p[o + 2] << 8 | p[o + 3] << 16          # 24-bit slot
        b = p[o] | p[o + 4] << 8 | p[o + 5] << 16              # 24-bit slot
        c = p[o + 6] | p[o + 7] << 8                           # 16-bit slot
        vals = (org[0] + a * step[0], org[1] + b * step[1], org[2] + c * step[2])
        xyz = [0.0, 0.0, 0.0]
        for slot, axis in enumerate(slots):
            xyz[axis] = vals[slot]
        new_verts.append(tuple(xyz))
    if any(not abs(x) < 1e5 for xyz in new_verts for x in xyz):
        return 0  # a layout we don't read right: leave the part out rather than wreck the whole model
    verts.extend(new_verts)
    tris.extend(new_tris)
    for k in range(used + 1):
        if uv and vb + uv[0] + (k + 1) * uv[1] <= len(p):
            uo = vb + uv[0] + k * uv[1]
            uvs.append((struct.unpack_from('<h', p, uo)[0] / 8192, struct.unpack_from('<h', p, uo + 2)[0] / 8192)
                       if uv[1] == 4 else struct.unpack_from('<2f', p, uo))
        else:
            uvs.append((0.0, 0.0))
    return added


def decode_model(m, streams, i):
    """-> dict(name, verts, uvs, tris, geometries, streams) or None."""
    r = read_resource(m, streams, i)
    if not r:
        return None
    v, p, used = r
    geoms = geometries(v)
    if not geoms or len(p) < 0x90:
        return None
    verts, uvs, tris, tri_mat, mats, mat_of = [], [], [], [], [], {}
    for vb, vb_size, mb, base, sidx in geoms:
        n = decode_geometry(p, vb, vb_size, mb, verts, uvs, tris)
        if not n:
            continue
        if (base, sidx) not in mat_of:
            mat_of[(base, sidx)] = len(mats)
            mats.append(shader_info(v, base, sidx))
        tri_mat += [mat_of[(base, sidx)]] * n
    if not tris:
        return None
    return dict(name=model_name(v), verts=verts, uvs=uvs, tris=tris, tri_mat=tri_mat, materials=mats,
                geometries=len(geoms), streams=used)


PAINT = (150, 28, 32)       # vehicle paint is a colour picked in game, not a texture
GLASS = (40, 55, 65)
PLAIN = (200, 200, 205)


class Textures:
    """Loads the TTPF textures a model's materials point to (packs stay open; images are cached)."""

    def __init__(self, index):
        self.index, self.packs, self.full, self.small = index or {}, {}, {}, {}

    def image(self, h):
        if h not in self.full:
            img = None
            if h in self.index:
                path, k = self.index[h]
                try:
                    import ttpf_tool
                    pk = self.packs.get(path) or self.packs.setdefault(path, ttpf_tool.Pack(path))
                    img = pk.image(pk.textures[k])[0]
                    img = img.convert('RGBA') if img else None
                except Exception:
                    img = None
            self.full[h] = img
        return self.full[h]

    def sampler(self, h):
        """Small RGB numpy array for colouring the preview, or None."""
        if h not in self.small:
            import numpy as np
            img = self.image(h)
            self.small[h] = np.asarray(img.convert('RGB').resize((128, 128))) if img else None
        return self.small[h]

    def close(self):
        for pk in self.packs.values():
            try:
                pk.close()
            except Exception:
                pass
        self.packs.clear()


def material_colour(mat, paint=PAINT):
    return GLASS if mat.get('glass') else paint if mat.get('paint') else PLAIN


def colourise(model, textures, paint=PAINT):
    """Per-triangle colours from each material's texture (sampled at the triangle's UV centre)."""
    import numpy as np
    t = np.asarray(model['tris'], dtype=np.int64)
    uv = np.asarray(model['uvs'], dtype=np.float64)
    tm = np.asarray(model.get('tri_mat') or [0] * len(t))
    out = np.empty((len(t), 3), dtype=np.float64)
    found = 0
    for mi, mat in enumerate(model.get('materials') or [{}]):
        sel = tm == mi
        if not sel.any():
            continue
        # body paint shaders: the colour is picked in game; their texture slots hold dirt / detail maps
        tex = textures.sampler(mat['tex']) if mat.get('tex') and not mat.get('paint') else None
        if tex is None:
            out[sel] = material_colour(mat, paint)
            continue
        found += 1
        c = uv[t[sel]].mean(1) % 1.0
        ys = np.clip((c[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
        xs = np.clip((c[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
        out[sel] = tex[ys, xs]
    model['colours'] = out
    model['textured'] = found
    return found


def write_obj(model, path, textures=None, paint=PAINT):
    """OBJ (+ .mtl and PNG textures next to it when textures are given)."""
    import os
    base = os.path.splitext(path)[0]
    stem = os.path.basename(base)
    mats = model.get('materials') or [{}]
    tm = model.get('tri_mat') or [0] * len(model['tris'])
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model["name"] or "model"} - exported by TTPF Studio\n')
        if textures is not None:
            f.write(f'mtllib {stem}.mtl\n')
        f.writelines(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in model['verts'])
        f.writelines(f'vt {u:.5f} {1 - v:.5f}\n' for u, v in model['uvs'])
        for mi in range(len(mats)):
            faces = [tri for tri, m in zip(model['tris'], tm) if m == mi]
            if not faces:
                continue
            if textures is not None:
                f.write(f'usemtl m{mi}\n')
            f.writelines(f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n' for a, b, c in faces)
    if textures is None:
        return
    with open(base + '.mtl', 'w', encoding='utf-8') as f:
        for mi, mat in enumerate(mats):
            r, g, b = (x / 255 for x in material_colour(mat, paint))
            f.write(f'newmtl m{mi}\nKd {r:.3f} {g:.3f} {b:.3f}\n')
            if mat.get('glass'):
                f.write('d 0.35\n')
            img = textures.image(mat['tex']) if mat.get('tex') and not mat.get('paint') else None
            if img:
                png = f'tex_{mat["tex"]:016x}.png'  # by hash: models exported together share them
                if not os.path.exists(os.path.join(os.path.dirname(path), png)):
                    img.save(os.path.join(os.path.dirname(path), png))
                f.write(f'Kd 1 1 1\nmap_Kd {png}\n')
            f.write('\n')


def is_root(m, off):
    """Cheap check on the first bytes of a stream: does it look like the start of a resource?"""
    try:
        h = zlib.decompressobj().decompress(m[off:off + 512], 0x70)
    except zlib.error:
        return False
    return len(h) >= 0x68 and vptr(u64(h, 8)) is not None and u64(h, 0x60) != GEOM_MARK


def find_models(m, streams, progress=lambda f: None, cancel=None):
    """[(stream index, name)] for every named drawable / fragment in the archive."""
    out, i, tick = [], 0, 0
    while i < len(streams):
        if i >= tick:
            tick = i + 2000
            progress(i / max(1, len(streams)))
            if cancel and cancel.is_set():
                break
        if not is_root(m, streams[i]):
            i += 1
            continue
        r = _virtual(m, streams, i)
        n = model_name(r[0]) if r else None
        if not n:
            i += 1
            continue
        out.append((i, n))
        i = r[1] + r[2]  # skip this resource's own chunks: a later one can look like a resource start too
    return out


def render(model, size=640, yaw=0.8, pitch=0.45, zoom=1.0, bg=(24, 26, 31)):
    """Flat-shaded preview as a PIL image (game is Z-up; shown Y-up). Uses model["colours"] if set."""
    import numpy as np
    from PIL import Image, ImageDraw
    v = np.asarray(model['verts'], dtype=np.float64)
    t = np.asarray(model['tris'], dtype=np.int64)
    img = Image.new('RGB', (size, size), bg)
    if not len(v) or not len(t):
        return img
    v = (v - (v.max(0) + v.min(0)) / 2)[:, [0, 2, 1]] * np.array([1, 1, -1])
    cy, sy_, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    w = v @ ry.T @ rx.T
    span = (np.linalg.norm(v, axis=1).max() * 2) or 1  # rotation-independent, so the model doesn't jump while turning
    sx = (w[:, 0] / span * 0.95 * zoom + 0.5) * size
    sy = (0.5 - w[:, 1] / span * 0.95 * zoom) * size
    tri = w[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = np.abs(n @ (np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])))
    col = model.get('colours')
    if col is not None and len(col) == len(t):
        col = np.clip(np.asarray(col) * (0.35 + 0.75 * shade)[:, None], 0, 255)
    else:
        col = None
    dr = ImageDraw.Draw(img)
    for k in np.argsort(tri[:, :, 2].mean(1)):
        a, b, c = t[k]
        if col is None:
            g = int(60 + 180 * shade[k])
            fill = (g, g, min(255, int(g * 1.05)))
        else:
            fill = tuple(int(x) for x in col[k])
        dr.polygon([(sx[a], sy[a]), (sx[b], sy[b]), (sx[c], sy[c])], fill=fill)
    return img
