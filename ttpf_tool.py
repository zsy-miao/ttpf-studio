"""TTPF Studio - view, export and replace textures in .ttpf texture packs (+ RPF9 header peek).
Run:  python ttpf_tool.py [file]      Self-test:  python ttpf_tool.py --selftest
Needs numpy + Pillow (texcodec.py next to this file).

Pack format (reverse engineered):
  'RDS5' header (0x18 B): magic, u32 entry_count, u32 virtual_size, u32 ?, u32 ?, u32 0
  entry_count x u64 index entries:
      bits 0-23  file offset / 256      bits 24-29 flags (32 = zlib)
      bits 30-47 virtual 4 KB page      bits 48-63 unknown (kept as-is)
  each entry -> one zlib stream (78 DA) = one mip level (or a packed mip tail), GPU-swizzled.
  'ETPF' block: u32 ?, u32 count, u32 size, then count x 16 B texture descriptors:
      u32 a: (w-1) | (h-1)<<14 | mips<<28    u32 b: DXGI_FORMAT<<3 | flags
      u32 page (low 20 bits)                 u32 ?
  A texture's mips are chained in virtual memory from smallest to largest: the blob at `page`,
  then the blob at page + len/4096, ... until one is as big as mip 0.
  Files whose header is zero or damaged open in "carved" mode: every 256-aligned zlib
  stream is listed and its format/size is guessed.
RPF9 archives: 0x40-byte header then an encrypted table of contents -> header info only.
"""
import mmap
import os
import queue
import shutil
import struct
import json
import sys
import threading
import time
import traceback
import zlib

import texcodec as tc

ALIGN = 256
PAGE = 4096


def inflate_at(m, p, limit=64 << 20):
    """Inflate a zlib stream starting at m[p] -> (data, compressed length), or None if it is not one.
    Feeds small chunks so a false candidate costs a few bytes, not a multi-MB copy."""
    d = zlib.decompressobj()
    parts, q, n, size, step = [], p, 0, len(m), 256
    try:
        while q < size:
            chunk = m[q:q + step]
            q += len(chunk)
            out = d.decompress(chunk)
            parts.append(out)
            n += len(out)
            if d.eof:
                return b''.join(parts), q - p - len(d.unused_data)
            if n > limit:
                return None
            step = 1 << 16
    except zlib.error:
        return None
    return None


class Entry:
    __slots__ = ('idx', 'off', 'raw', 'end', 'page', 'length')

    def __init__(self, idx, off, raw, page=None):
        self.idx, self.off, self.raw, self.page = idx, off, raw, page
        self.end = self.length = None


NOISY = 45  # roughness of a 256 px crop: normal textures score below this, wrong-data decodes far above


def group_chains(entries):
    """Consecutive blobs that grow 2x / 4x are one texture's mips, smallest first.
    shortcut: size-only rule, ~10% of textures fold into a neighbour's group; the 'All blobs' view still lists them."""
    out = []
    for e in entries:
        if out:
            a, b = out[-1][-1].length, e.length
            if b in (2 * a, 4 * a) or (len(out[-1]) == 1 and (a < b <= 4 * a or b == a)):
                out[-1].append(e)
                continue
        out.append([e])
    return out


class Texture:
    def __init__(self, n, fmt, w, h, mips=1, dxgi=None, page=None, guessed=False, tiling='S'):
        self.n, self.fmt, self.w, self.h, self.mips, self.tiling = n, fmt, w, h, mips, tiling
        self.dxgi, self.page, self.guessed = dxgi, page, guessed
        self.chain = []       # entries, smallest mip blob first
        self.status = 'missing'
        self.checked = False  # noise check done

    @property
    def name(self):
        return f'tex_{self.n:05d}'

    def mip0(self):
        if self.chain and self.fmt and self.chain[-1].length >= tc.mip_bytes(self.fmt, self.w, self.h):
            return self.chain[-1]
        return None

    def best(self):
        """(entry, w, h, level) of the largest mip we actually have."""
        if not self.chain or not self.fmt:
            return None
        e = self.chain[-1]
        for k in range(0, 14):
            w, h = max(1, self.w >> k), max(1, self.h >> k)
            if e.length >= tc.mip_bytes(self.fmt, w, h):
                return e, w, h, k
        return e, max(1, self.w >> 13), max(1, self.h >> 13), 13


class Pack:
    def __init__(self, path, progress=lambda f: None):
        self.path = path
        self.f = open(path, 'rb')
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = len(self.m)
        self.edits = {}  # id(entry) -> (entry, new decompressed bytes)
        self.carved = self.m[:4] != b'RDS5'
        self.progress = progress
        if self.carved:
            self.entries = self._carve()
            self.set_grouping(True)
        else:
            self.entries = self._table()
            self.textures = self._descriptors()

    # ---- index
    def _table(self):
        cnt = struct.unpack_from('<I', self.m, 4)[0]
        es = []
        for i in range(cnt):
            raw = struct.unpack_from('<Q', self.m, 0x18 + 8 * i)[0]
            if raw:
                es.append(Entry(i, (raw & 0xFFFFFF) * ALIGN, raw, ((raw >> 24) & 0xFFFFFF) >> 6))
        es.sort(key=lambda e: e.off)
        for a, b in zip(es, es[1:] + [None]):
            a.end = b.off if b else self.size  # slot = gap to next blob
        self.pages = {e.page: e for e in es}
        return es

    def _carve(self):
        es, p, last = [], 0, -1
        while (p := self.m.find(b'\x78\xda', p)) != -1:
            if p % ALIGN:
                p += 1
                continue
            if p >> 24 != last:
                last = p >> 24
                self.progress(p / self.size)
            r = inflate_at(self.m, p, 0x800000)
            if r and len(r[0]) >= 64:
                e = Entry(None, p, None)
                e.end = p + r[1]  # shortcut: carved slot = exact stream size, so edits must compress no larger
                e.length = len(r[0])
                es.append(e)
                p += -(-r[1] // ALIGN) * ALIGN
            else:
                p += ALIGN
        self.pages = {}
        return es

    def set_grouping(self, on):
        """Headerless packs: one texture per mip chain (on) or one per blob (off)."""
        self.grouped = on
        self.textures = []
        for i, g in enumerate(group_chains(self.entries) if on else [[e] for e in self.entries]):
            t = Texture(i, None, 0, 0, mips=len(g))
            t.chain, t.status = g, 'unknown'
            self.textures.append(t)

    def _descriptors(self):
        at = self.m.find(b'ETPF', 0x18, min(self.size, 0x4000000))
        if at < 0:
            raise ValueError('No ETPF texture table found (file may be truncated).')
        n = struct.unpack_from('<I', self.m, at + 8)[0]
        out = []
        for k in range(n):
            a, b, p, _ = struct.unpack_from('<IIII', self.m, at + 16 + k * 16)
            fmt = tc.DXGI.get((b & 0x3FF) >> 3)
            t = Texture(k, fmt, (a & 0x3FFF) + 1, ((a >> 14) & 0x3FFF) + 1, a >> 28, (b & 0x3FF) >> 3, p & 0xFFFFF,
                        tiling=tc.tiling_from_desc(b))
            if fmt:
                self._chain(t)
            else:
                t.status = 'unsupported'
            out.append(t)
            if k % 200 == 0:
                self.progress(k / n)
        return out

    def _chain(self, t):
        need = tc.mip_bytes(t.fmt, t.w, t.h)
        q = t.page
        while q in self.pages and len(t.chain) < 16:
            e = self.pages[q]
            if e.length is None:
                blob = self.blob(e)
                e.length = len(blob) if blob is not None else 0
            if not e.length:
                break
            t.chain.append(e)
            if e.length >= need:
                break
            q += -(-e.length // PAGE)
        t.status = 'full' if t.mip0() else ('low' if t.chain else 'missing')

    # ---- blobs
    def blob(self, e):
        if id(e) in self.edits:
            return self.edits[id(e)][1]
        b = self.m[e.off:e.end]
        if b[:2] != b'\x78\xda':
            return None
        try:
            return zlib.decompressobj().decompress(b)
        except zlib.error:
            return None

    def set_blob(self, e, data):
        self.edits[id(e)] = (e, bytes(data))

    def revert(self, t):
        for e in t.chain:
            self.edits.pop(id(e), None)

    def edited(self, t):
        return any(id(e) in self.edits for e in t.chain)

    # ---- textures
    def ensure_format(self, t):
        if t.fmt is None and t.chain:
            g = tc.guess(self.blob(t.chain[-1]))
            if g:
                t.fmt, t.w, t.h, t.tiling = g
                t.guessed = True
                t.status = 'guess'
            else:
                t.status = 'unsupported'  # size fits no power-of-two texture (e.g. a packed mip tail)

    def image(self, t):
        self.ensure_format(t)
        b = t.best()
        if not b:
            return None, None
        e, w, h, k = b
        data = self.blob(e)
        img = tc.decode(t.fmt, data, w, h, t.tiling)
        if not t.guessed and not t.checked and k == 0:
            # damaged packs: a broken mip chain can land on another texture's blob, which decodes as noise
            t.checked = True
            r = tc.roughness(img.crop((0, 0, 256, 256)))
            g = tc.guess(data) if r > NOISY else None
            if g:
                alt = tc.decode(g[0], data, g[1], g[2], g[3])
                if 0.5 < tc.roughness(alt.crop((0, 0, 256, 256))) < 0.3 * r:  # clearly smoother, and not a flat blank
                    t.fmt, t.w, t.h, t.tiling = g
                    t.chain, t.guessed, t.status = [e], True, 'fixed'
                    return alt, 0
        return img, k

    def linear(self, t):
        e, w, h, k = t.best()
        return tc.unswizzle(t.fmt, self.blob(e), w, h, t.tiling), w, h

    def replace(self, t, fmt, w, h, levels):
        """Write mip 0 (and any further mips that have their own blob) from linear data."""
        if tc.FORMATS[fmt][:2] != tc.FORMATS[t.fmt][:2] or (fmt in ('BGRA8', 'RGBA8')) != (t.fmt in ('BGRA8', 'RGBA8')):
            raise ValueError(f'Format mismatch: file is {fmt}, texture is {t.fmt}.')
        b = t.best()
        if not b:
            raise ValueError('This texture has no data in the file to replace.')
        e, bw, bh, k = b
        if (w, h) != (bw, bh):
            raise ValueError(f'Image is {w}x{h}, texture needs {bw}x{bh}.')
        if fmt != t.fmt and {fmt, t.fmt} == {'RGBA8', 'BGRA8'}:
            import numpy as np
            levels = [np.frombuffer(L, np.uint8).reshape(-1, 4)[:, [2, 1, 0, 3]].tobytes() for L in levels]
        if self.carved:
            levels = levels[:1]  # carved chains are a size-based guess: only touch the blob we decoded
        written = 0
        own = t.chain[1:] if len(t.chain) > 1 else t.chain  # chain[0] is the packed mip tail
        for lvl, data in enumerate(levels):
            i = len(t.chain) - 1 - lvl
            if lvl >= len(own) or i < len(t.chain) - len(own):
                break
            ce = t.chain[i]
            mw, mh = max(1, w >> lvl), max(1, h >> lvl)
            old = self.blob(ce)
            if len(data) < tc.mip_bytes(t.fmt, mw, mh):
                break
            self.set_blob(ce, tc.swizzle(t.fmt, data, mw, mh, old, t.tiling))
            written += 1
        return written

    # ---- save
    def save_as(self, out, progress=lambda f: None):
        if os.path.abspath(out) == os.path.abspath(self.path):
            raise ValueError('Save to a new file; the original is kept as a backup.')
        shutil.copyfile(self.path, out)
        with open(out, 'r+b') as f:
            for i, (e, data) in enumerate(self.edits.values()):
                progress(i / max(1, len(self.edits)))
                comp = zlib.compress(data, 9)
                if len(comp) <= e.end - e.off:
                    f.seek(e.off)
                    f.write(comp + b'\0' * (e.end - e.off - len(comp)))
                    continue
                if self.carved:
                    raise ValueError(f'Edited blob @0x{e.off:x} compresses bigger than its slot and this file '
                                     'has no index to repoint. Try an image with less detail.')
                f.seek(0, 2)
                new = -(-f.tell() // ALIGN) * ALIGN
                if new // ALIGN >= 1 << 24:
                    raise ValueError('File too big to append (24-bit offset limit).')
                f.seek(new)
                f.write(comp + b'\0' * (-len(comp) % ALIGN))
                f.seek(0x18 + 8 * e.idx)
                f.write(struct.pack('<Q', (e.raw & ~0xFFFFFF) | (new // ALIGN)))

    def close(self):
        self.m.close()
        self.f.close()


class Item:
    __slots__ = ('n', 'kind', 'off', 'clen', 'length', 'name', 'info', 'parts')

    def __init__(self, kind, off, clen, length, name, info=''):
        self.kind, self.off, self.clen, self.length, self.name, self.info = kind, off, clen, length, name, info
        self.n = 0
        self.parts = [(off, clen)]  # big files are stored as several 64 KB zlib chunks back to back


ITEM_KINDS = {'xml': 'XML', 'text': 'Text', 'video': 'Video', 'psin': 'PSIN data', 'skel': 'Skeleton', 'bin': 'Binary'}


def classify(data):
    """-> (kind, short name tag, extension) for a carved stream."""
    import re
    body = data[3:] if data[:3] == b'\xef\xbb\xbf' else data
    head = body[:512]
    if head and sum(32 <= c < 127 or c in (9, 10, 13) for c in head) / len(head) > 0.95:
        if head.lstrip()[:1] == b'<':
            m_ = re.search(rb'<(?![?!])([A-Za-z_][\w:.\-]*)', body[:4096])
            return 'xml', (m_.group(1).decode('latin1') if m_ else 'xml')[:48], 'xml'
        first = head.split(b'\n', 1)[0].strip()[:32]
        return 'text', re.sub(r'[^\w\-]+', '_', first.decode('latin1')).strip('_') or 'text', 'txt'
    if body[:4] == b'PSIN':
        return 'psin', 'psin', 'psin'
    if b'SKEL_' in data:  # RAGE resource with a skeleton: bone names are stored in clear (rigged model / skeleton data)
        return 'skel', 'skeleton', 'bin'
    return 'bin', data[:4].hex(), 'bin'


class Archive:
    """RPF9 archive: the file list is encrypted, but much of the data inside is not. Carves what it can:
    zlib streams (16-byte aligned) classified as XML / text / PSIN / binary, and Bink videos (size from their header)."""
    ALIGN = 16

    def __init__(self, path, progress=lambda f: None):
        import re
        self.path = path
        self.f = open(path, 'rb')
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = len(self.m)
        self.encrypted = self.m[:4] == b'9FPR'
        items = []
        p, last = 0, -1
        while (p := self.m.find(b'\x78\xda', p)) != -1:
            if p % self.ALIGN:
                p += 1
                continue
            if p >> 26 != last:
                last = p >> 26
                progress(0.85 * p / self.size)
            r = inflate_at(self.m, p)
            if r and len(r[0]) >= 32:
                kind, tag, ext = classify(r[0])
                prev = items[-1] if items else None
                fresh = r[0][:3] == b'\xef\xbb\xbf' or r[0].lstrip()[:5] == b'<?xml' or r[0][:4] == b'PSIN'
                if (prev and prev.kind != 'video' and not fresh and prev.length % 65536 == 0
                        and p == prev.parts[-1][0] + -(-prev.parts[-1][1] // self.ALIGN) * self.ALIGN):
                    prev.parts.append((p, r[1]))  # continuation chunk of the previous file
                    prev.clen += r[1]
                    prev.length += len(r[0])
                else:
                    items.append(Item(kind, p, r[1], len(r[0]), f'{tag}.{ext}'))
                p += -(-r[1] // self.ALIGN) * self.ALIGN
            else:
                p += self.ALIGN
        for magic in (b'KB2', b'BIK'):
            p = 0
            while (p := self.m.find(magic, p)) != -1:
                if p % self.ALIGN or p + 28 > self.size:
                    p += 1
                    continue
                ver = self.m[p + 3]
                size = struct.unpack_from('<I', self.m, p + 4)[0] + 8
                frames, = struct.unpack_from('<I', self.m, p + 8)
                w, h = struct.unpack_from('<II', self.m, p + 20)
                if 97 <= ver <= 122 and 64 < size <= self.size - p and 0 < frames < 10 ** 6 and 16 <= w <= 8192 and 16 <= h <= 8192:
                    ext = 'bk2' if magic == b'KB2' else 'bik'
                    items.append(Item('video', p, size, size, f'video_{w}x{h}.{ext}', f'{w}×{h}, {frames} frames'))
                    p += size
                else:
                    p += 1
            progress(0.95)
        items.sort(key=lambda it: it.off)
        for i, it in enumerate(items):
            it.n = i
            it.name = f'{i:05d}_{it.name}'
        self.items = items
        self.header = rpf_info(path)
        self._text = {}  # item n -> lower-cased text, filled by search()

    def search(self, q):
        """Numbers of the XML / text items whose content contains q (case-insensitive)."""
        q = q.lower()
        hits = []
        for it in self.items:
            if it.kind not in ('xml', 'text'):
                continue
            t = self._text.get(it.n)
            if t is None:
                t = self._text[it.n] = self.data(it).decode('utf-8', 'replace').lower()
            if q in t:
                hits.append(it.n)
        return hits

    @classmethod
    def from_items(cls, path, items):
        """Re-open an archive from an earlier scan (the library index) without scanning it again."""
        a = cls.__new__(cls)
        a.path = path
        a.f = open(path, 'rb')
        a.m = mmap.mmap(a.f.fileno(), 0, access=mmap.ACCESS_READ)
        a.size = len(a.m)
        a.encrypted = a.m[:4] == b'9FPR'
        a.items = items
        for i, it in enumerate(items):
            it.n = i
        a.header = rpf_info(path)
        a._text = {}
        return a

    def data(self, it):
        if it.kind == 'video':
            return self.m[it.off:it.off + it.clen]
        out = []
        for off, _ in it.parts:
            r = inflate_at(self.m, off)
            out.append(r[0] if r else b'')
        return b''.join(out)

    def close(self):
        self.m.close()
        self.f.close()


FILES_HEADER = ['archive', 'item', 'kind', 'size', 'compressed', 'file_offset', 'details', 'file']


def dump_archive(path, out_dir, cancel=None, progress=lambda f: None):
    """Write every carved item of an RPF to out_dir/<Kind>/<name> plus out_dir/index.csv. Returns a Counter."""
    import collections
    import csv
    arc = Archive(path, lambda f: progress(f * 0.5))
    stats = collections.Counter()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'files.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(FILES_HEADER)
        for i, it in enumerate(arc.items):
            if cancel and cancel.is_set():
                break
            if i % 50 == 0:
                progress(0.5 + 0.5 * i / max(1, len(arc.items)))
            rel = os.path.join(ITEM_KINDS[it.kind], it.name)
            os.makedirs(os.path.join(out_dir, ITEM_KINDS[it.kind]), exist_ok=True)
            with open(os.path.join(out_dir, rel), 'wb') as g:
                g.write(arc.data(it))
            stats[ITEM_KINDS[it.kind]] += 1
            wr.writerow([os.path.basename(path), it.name, ITEM_KINDS[it.kind], it.length, it.clen, f'0x{it.off:x}', it.info, rel])
    arc.close()
    return stats


BINK_PLAYERS = ('binkpl64.exe', 'binkplay.exe', 'radvideo64.exe', 'radvideo.exe')
VIDEO_CACHE = os.path.join(os.environ.get('TEMP') or os.path.expanduser('~'), 'TTPF Studio videos')


def bink_info(path):
    """(frames, width, height, fps) from a Bink / Bink 2 header."""
    with open(path, 'rb') as f:
        h = f.read(44)
    if h[:3] not in (b'KB2', b'BIK'):
        raise ValueError('Not a Bink video.')
    frames, = struct.unpack_from('<I', h, 8)
    w, hh, fn, fd = struct.unpack_from('<IIII', h, 20)
    return frames, w, hh, (fn / fd if fd else 30.0)


def find_rad_converter(saved=None, player=None):
    """radvideo64.exe (RAD Video Tools' converter): saved path, next to the player, or the usual folders."""
    if saved and os.path.isfile(saved):
        return saved
    dirs = [os.path.dirname(player)] if player else []
    dirs += [os.path.join(b, 'RADVideo') for b in filter(None, (os.environ.get(k) for k in ('ProgramFiles(x86)', 'ProgramFiles')))]
    for d in dirs:
        for exe in ('radvideo64.exe', 'radvideo.exe'):
            if os.path.isfile(os.path.join(d, exe)):
                return os.path.join(d, exe)
    return None


def find_ffmpeg(saved=None):
    return saved if saved and os.path.isfile(saved) else shutil.which('ffmpeg')


def _close_done_window(pid):
    """RAD's converter waits on a 'Done!' window after finishing; close it the way the X button would."""
    if os.name != 'nt':
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if 'Done' in buf.value:
                found.append(hwnd)
        return True
    user32.EnumWindows(each, 0)
    for hwnd in found:
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    return bool(found)


def convert_bink_to_mp4(src, dst, rad, ffmpeg, progress=lambda f, msg: None, cancel=None):
    """Bink / Bink 2 -> MP4 (H.264 + AAC). RAD Video Tools decodes to an uncompressed AVI, ffmpeg compresses it.
    shortcut: needs temporary disk space for the raw AVI (~6 MB per 1080p frame); a pipe-based path would avoid it."""
    import subprocess
    frames, w, h, _ = bink_info(src)
    need = frames * w * h * 3 + (64 << 20)
    os.makedirs(VIDEO_CACHE, exist_ok=True)
    free = shutil.disk_usage(VIDEO_CACHE).free
    if need > free * 0.9:
        raise ValueError(f'This video needs {human(need)} of temporary disk space, only {human(free)} is free.')
    tmp = os.path.join(VIDEO_CACHE, f'convert_{os.getpid()}_{time.time_ns()}.avi')
    no_window = 0x08000000 if os.name == 'nt' else 0
    try:
        proc = subprocess.Popen([rad, 'binkconv', os.path.abspath(src), tmp])
        last, still = -1, time.time()
        while proc.poll() is None:
            if cancel and cancel.is_set():
                _close_done_window(proc.pid)
                proc.terminate()
                raise RuntimeError('Cancelled.')
            size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if size != last:
                last, still = size, time.time()
            elif time.time() - still > 300 and not _close_done_window(proc.pid):
                proc.terminate()
                raise RuntimeError('RAD Video Tools stopped making progress.')
            progress(0.75 * min(1.0, size / need), 'Decoding the Bink video…')
            _close_done_window(proc.pid)
            time.sleep(0.4)
        if not os.path.exists(tmp) or os.path.getsize(tmp) < 4096:
            raise RuntimeError('RAD Video Tools produced no output (is it the full RAD Video Tools install?).')
        cmd = [ffmpeg, '-v', 'error', '-y', '-i', tmp, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-progress', 'pipe:1', '-nostats', dst]
        ff = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=no_window)
        for line in ff.stdout:
            if line.startswith('frame='):
                done = int(line.split('=', 1)[1] or 0)
                progress(0.75 + 0.25 * min(1.0, done / max(1, frames)), f'Encoding MP4… frame {done}/{frames}')
            if cancel and cancel.is_set():
                ff.terminate()
                raise RuntimeError('Cancelled.')
        err = ff.stderr.read()
        if ff.wait():
            raise RuntimeError('ffmpeg failed: ' + err.strip()[-400:])
        return dst
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def copy_image_to_clipboard(img):
    """Put a PIL image on the Windows clipboard (transparent parts become white)."""
    import ctypes
    import io
    from ctypes import wintypes
    from PIL import Image
    rgba = img.convert('RGBA')
    flat = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    flat.alpha_composite(rgba)
    buf = io.BytesIO()
    flat.convert('RGB').save(buf, 'BMP')
    dib = buf.getvalue()[14:]  # CF_DIB = the BMP without its 14-byte file header
    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.GlobalAlloc.restype, k32.GlobalAlloc.argtypes = wintypes.HGLOBAL, [wintypes.UINT, ctypes.c_size_t]
    k32.GlobalLock.restype, k32.GlobalLock.argtypes = ctypes.c_void_p, [wintypes.HGLOBAL]
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    u32.SetClipboardData.restype, u32.SetClipboardData.argtypes = wintypes.HANDLE, [wintypes.UINT, wintypes.HANDLE]
    h = k32.GlobalAlloc(0x0002, len(dib))  # GMEM_MOVEABLE
    ptr = k32.GlobalLock(h)
    ctypes.memmove(ptr, dib, len(dib))
    k32.GlobalUnlock(h)
    if not u32.OpenClipboard(None):
        raise OSError('The clipboard is busy, try again.')
    try:
        u32.EmptyClipboard()
        if not u32.SetClipboardData(8, h):
            raise OSError('Could not copy the image.')
    finally:
        u32.CloseClipboard()


SHORTCUTS = """Ctrl+O              Open a pack, archive or video
Ctrl+Shift+O        Open a folder of packs
Ctrl+F              Search the list (tick "in files" to search inside XML / text)
Ctrl+Click, Shift+Click   Select several items, then export / save / convert them together
Ctrl+C              Copy the current texture as an image
Ctrl+E              Export the current texture as PNG
Ctrl+G              Switch between viewer and thumbnail grid
Ctrl+S              Save the modded pack
Mouse wheel         Zoom    ·    Drag: pan    ·    Double-click: fit
F1                  This list"""


class BinkEmbed:
    """RAD's Bink player running as a child window of ours (Windows only), driven through its own controls,
    read from its window procedure: Space (WM_CHAR) toggles pause; a click on the bottom 16 px of the picture
    seeks to frame ((x - 8) * frames + (w - 8) / 2 - 1) // (w - 8) + 1. It rescales the picture on resize."""

    def __init__(self, exe, path, frames, fps, vw, vh, screen_rect):
        import subprocess
        self.frames, self.fps, self.vw, self.vh = max(1, frames), fps or 30.0, vw, vh
        x, y, w, h = screen_rect
        self.proc = subprocess.Popen([exe, os.path.normpath(path), f'/W{w}', f'/H{h}', f'/X{x}', f'/Y{y}'])
        self.hwnd, self.cw, self.ch = None, w, h
        self.paused, self.base, self.t0 = False, 0.0, time.monotonic()

    @property
    def duration(self):
        return self.frames / self.fps

    def alive(self):
        return self.proc.poll() is None

    def _post(self, msg, wparam, lparam):
        import ctypes
        if self.hwnd:
            ctypes.windll.user32.PostMessageW(self.hwnd, msg, wparam, lparam)

    def attach(self, parent):
        """Adopt the player's window into `parent` once it exists. Returns True when done."""
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def each(h, _):
            owner = wintypes.DWORD()
            u32.GetWindowThreadProcessId(h, ctypes.byref(owner))
            if owner.value == self.proc.pid and u32.IsWindowVisible(h):
                buf = ctypes.create_unicode_buffer(256)
                u32.GetWindowTextW(h, buf, 256)
                if 'Bink Video Player' in buf.value and 'Help' not in buf.value:
                    found.append(h)
            return True
        u32.EnumWindows(each, 0)
        if not found:
            return False
        h = found[0]
        u32.SetWindowLongW(h, -16, 0x40000000 | 0x10000000)  # WS_CHILD | WS_VISIBLE: no caption, no border
        u32.SetParent(h, parent)
        u32.EnableWindow(h, False)  # clicks fall through to our panel, so our play/pause state stays true
        self.hwnd, self.t0 = h, time.monotonic()
        self._post(0x0007, 0, 0)  # WM_SETFOCUS: it pauses on losing focus (which embedding causes); this resumes it
        return True

    def place(self, x, y, w, h):
        import ctypes
        if self.hwnd:  # SWP_NOZORDER | SWP_FRAMECHANGED: also drops the frame left over from the old window style
            ctypes.windll.user32.SetWindowPos(self.hwnd, 0, x, y, w, h, 0x0004 | 0x0020)
        self.cw, self.ch = w, h

    def position(self):
        return min(self.base if self.paused else self.base + time.monotonic() - self.t0, self.duration)

    def toggle_pause(self):
        self._post(0x0102, 0x20, 0)  # WM_CHAR ' '
        if self.paused:
            self.t0, self.paused = time.monotonic(), False
        else:
            self.base, self.paused = self.position(), True

    def seek(self, frac):
        frame = max(1, min(self.frames, int(max(0.0, min(1.0, frac)) * self.frames) + 1))
        xv = 8 + (frame - 0.5) * (self.vw - 8) / self.frames  # inverse of the player's strip-to-frame mapping
        xv = min(max(xv, 9), self.vw - 9)
        x, y = int(xv * self.cw / self.vw), self.ch - 2
        lp = (y << 16) | (x & 0xFFFF)
        if not self.paused:
            self._post(0x0102, 0x20, 0)  # its seek strip only jumps reliably while paused: pause first
        self._post(0x0201, 1, lp)  # WM_LBUTTONDOWN
        self._post(0x0202, 0, lp)  # WM_LBUTTONUP
        if not self.paused:
            self._post(0x0102, 0x20, 0)  # ...then carry on playing from the new spot
        self.base, self.t0 = (frame - 1) / self.fps, time.monotonic()

    def stop(self):
        import subprocess
        self._post(0x0010, 0, 0)  # WM_CLOSE
        try:
            self.proc.wait(2)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


def fit_rect(vw, vh, w, h):
    """Largest vw:vh rectangle centred in w x h -> (x, y, width, height)."""
    if vw <= 0 or vh <= 0 or w <= 0 or h <= 0:
        return 0, 0, max(1, w), max(1, h)
    s = min(w / vw, h / vh)
    fw, fh = max(1, int(vw * s)), max(1, int(vh * s))
    return (w - fw) // 2, (h - fh) // 2, fw, fh


def fmt_time(sec):
    sec = int(max(0, sec))
    return f'{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}' if sec >= 3600 else f'{sec // 60}:{sec % 60:02d}'


def find_bink_player(saved=None):
    """RAD Video Tools player: the path the user picked before, else the usual install folders, else PATH."""
    if saved and os.path.isfile(saved):
        return saved
    for base in filter(None, (os.environ.get(k) for k in ('ProgramFiles(x86)', 'ProgramFiles', 'LOCALAPPDATA'))):
        for sub in ('RADVideo', 'RAD Video Tools', os.path.join('Programs', 'RADVideo')):
            for exe in BINK_PLAYERS:
                if os.path.isfile(os.path.join(base, sub, exe)):
                    return os.path.join(base, sub, exe)
    return next(filter(None, (shutil.which(e) for e in BINK_PLAYERS)), None)


def rpf_info(path):
    with open(path, 'rb') as f:
        h = f.read(0x40)
    if h[:4] != b'9FPR':
        return f'Not a readable RPF9 archive (first bytes {h[:4].hex(" ")}).\nThe file is probably incomplete or zero-filled.'
    v = struct.unpack_from('<IIII', h, 4)
    return ('RPF9 archive header\n\n'
            f'field 0x04   0x{v[0]:x}\nfield 0x08   0x{v[1]:x}\nfield 0x0C   0x{v[2]:x}\n'
            f'0x10         {h[0x10:0x18].hex(" ")}\n0x18         {h[0x18:0x28].hex(" ")}\n'
            f'0x28         {h[0x28:0x40].hex(" ")}\n\n'
            'Everything after 0x40 (the file list) is encrypted. Listing or extracting\n'
            'needs keys from the game executable, which is not part of this dump.')


# ====================================================================== bulk dump
CSV_HEADER = ['pack', 'texture', 'format', 'width', 'height', 'tiling', 'role', 'status', 'mip', 'file_offset',
              'blob_sha1', 'png', 'duplicate_of']


def pack_name(path, root):
    """Output folder name: the file name for packs directly in root, the relative path for nested ones
    (game folders reuse names like ttdb_vfx.ttpf in several sub-packs)."""
    rel = os.path.splitext(os.path.relpath(path, root))[0]
    return rel.replace(os.sep, '__').replace('/', '__')


def dump_folder(src, out, headerless=True, skip_blank=True, progress=lambda msg, frac: None, cancel=None, workers=None,
                root=None, rpf=False):
    """Export every texture in every pack under `src` (a folder or a list of pack paths) to `out`.
    PNGs go to out/<pack>/<role>/, each pack gets out/<pack>/index.csv and they are merged into out/index.csv.
    Finished packs are marked with a .done file, so running again resumes where it stopped. Returns a stats Counter."""
    import collections
    import concurrent.futures as cf
    import csv
    import hashlib
    paths = src if isinstance(src, list) else [p for p, kind, _ in scan_folder(src)
                                               if kind == 'indexed' or (headerless and kind == 'headerless')
                                               or (rpf and kind in ('rpf', 'rpf_noheader'))]
    os.makedirs(out, exist_ok=True)
    stats = collections.Counter()
    seen = {}  # blob sha1 -> png already written (same texture stored in several packs)
    root = root or (src if not isinstance(src, list) else os.path.dirname(os.path.commonpath(paths)) if len(paths) > 1
                    else os.path.dirname(paths[0]))
    names = [pack_name(p, root) for p in paths]
    for pname in names:  # resume: remember what finished packs already exported
        if os.path.exists(os.path.join(out, pname, '.done')) and os.path.exists(os.path.join(out, pname, 'index.csv')):
            with open(os.path.join(out, pname, 'index.csv'), newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row['png']:
                        seen.setdefault(row['blob_sha1'], row['png'])
    workers = workers or max(2, (os.cpu_count() or 4) - 1)
    if True:
        for fi, path in enumerate(paths):
            if cancel and cancel.is_set():
                break
            pname = names[fi]
            pdir = os.path.join(out, pname)
            if os.path.exists(os.path.join(pdir, '.done')):
                stats['packs skipped (already done)'] += 1
                continue
            progress(f'Opening {pname}  ({fi + 1}/{len(paths)})', fi / len(paths))
            if path.lower().endswith('.rpf'):
                try:
                    res = dump_archive(path, pdir, cancel, lambda f, fi=fi, pn=pname: progress(
                        f'{pn}: carving archive  ({fi + 1}/{len(paths)} packs)', (fi + f) / len(paths)))
                except Exception as ex:
                    stats['packs failed'] += 1
                    log_error(f'dump: {path}: {ex}\n')
                    continue
                for k, v in res.items():
                    stats['RPF ' + k] += v
                if not (cancel and cancel.is_set()):
                    open(os.path.join(pdir, '.done'), 'w').close()
                    stats['packs done'] += 1
                continue
            try:
                pk = Pack(path)
            except Exception as ex:
                stats['packs failed'] += 1
                log_error(f'dump: {path}: {ex}\n')
                continue
            todo = [t for t in pk.textures if t.status not in ('missing', 'unsupported')]
            os.makedirs(pdir, exist_ok=True)
            fcsv = open(os.path.join(pdir, 'index.csv'), 'w', newline='', encoding='utf-8')
            wr = csv.writer(fcsv)
            wr.writerow(CSV_HEADER)

            def one(t):
                if cancel and cancel.is_set():
                    return None
                try:
                    img, k = pk.image(t)
                except Exception:
                    return t, None, None, None
                if img is None:
                    return t, None, None, None
                if skip_blank and all(lo == hi for lo, hi in img.getextrema()[:3]):
                    return t, img, 'blank', None
                e = t.best()[0]
                sha = hashlib.sha1(pk.blob(e)).hexdigest()
                return t, img, texture_role(img, t.fmt), (sha, k, e.off)

            with cf.ThreadPoolExecutor(workers) as ex:
                for j, res in enumerate(ex.map(one, todo)):
                    if j % 25 == 0:
                        progress(f'{pname}: {j}/{len(todo)} textures  ({fi + 1}/{len(paths)} packs)',
                                 (fi + j / max(1, len(todo))) / len(paths))
                    if res is None:
                        continue
                    t, img, role, meta = res
                    if img is None:
                        stats['failed to decode'] += 1
                        continue
                    if role == 'blank':
                        stats['blank, skipped'] += 1
                        continue
                    sha, k, off = meta
                    png, dup = '', seen.get(sha, '')
                    if dup:
                        stats['duplicates, skipped'] += 1
                    else:
                        rel = os.path.join(pname, role, f'{t.name}_{t.fmt}_{img.width}x{img.height}.png')
                        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                        img.save(os.path.join(out, rel), compress_level=1)
                        seen[sha] = png = rel
                        stats['exported'] += 1
                        stats[role] += 1
                    wr.writerow([pname, t.name, t.fmt, img.width, img.height, t.tiling, role, t.status, k,
                                 f'0x{off:x}', sha, png, dup])
            pk.close()
            fcsv.close()
            if not (cancel and cancel.is_set()):
                open(os.path.join(pdir, '.done'), 'w').close()
                stats['packs done'] += 1
    with open(os.path.join(out, 'index.csv'), 'w', newline='', encoding='utf-8') as f:  # merged catalogue
        wr = csv.writer(f)
        wr.writerow(CSV_HEADER)
        for pname in sorted(os.listdir(out)):  # every pack catalogue in out, also from earlier or parallel runs
            part = os.path.join(out, pname, 'index.csv')
            if os.path.exists(part):
                with open(part, newline='', encoding='utf-8') as pf:
                    wr.writerows(list(csv.reader(pf))[1:])
    archives = [d_ for d_ in sorted(os.listdir(out)) if os.path.exists(os.path.join(out, d_, 'files.csv'))]
    if archives:
        with open(os.path.join(out, 'files.csv'), 'w', newline='', encoding='utf-8') as f:  # merged archive catalogue
            wr = csv.writer(f)
            wr.writerow(FILES_HEADER)
            for d_ in archives:
                with open(os.path.join(out, d_, 'files.csv'), newline='', encoding='utf-8') as pf:
                    wr.writerows(list(csv.reader(pf))[1:])
    progress('Finished' if not (cancel and cancel.is_set()) else 'Stopped', 1.0)
    return stats


# ====================================================================== app support
APP_DIR = os.path.join(os.environ.get('APPDATA') or os.path.expanduser('~'), 'TTPF Studio')
SETTINGS = os.path.join(APP_DIR, 'settings.json')
LOG = os.path.join(APP_DIR, 'error.log')


def load_settings():
    try:
        with open(SETTINGS, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(s):
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(SETTINGS, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=1)
    except OSError:
        pass  # settings are a convenience; never block the app over them


def log_error(text):
    try:
        os.makedirs(APP_DIR, exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(time.strftime('\n--- %Y-%m-%d %H:%M:%S\n') + text)
    except OSError:
        pass


def scan_folder(d, limit=3000):
    """All .ttpf / .rpf files under d (8 levels deep), with what their header says they are."""
    out = []
    base = d.rstrip('\\/').count(os.sep)
    for root_, dirs, files in os.walk(d):
        if root_.count(os.sep) - base >= 8:
            dirs[:] = []
        for f in files:
            ext = f.lower().rsplit('.', 1)[-1]
            if ext not in ('ttpf', 'rpf'):
                continue
            p = os.path.join(root_, f)
            try:
                with open(p, 'rb') as fh:
                    head = fh.read(4)
                size = os.path.getsize(p)
            except OSError:
                continue
            if ext == 'ttpf':
                kind = 'indexed' if head == b'RDS5' else 'headerless'
            else:
                kind = 'rpf' if head == b'9FPR' else 'rpf_noheader'
            out.append((p, kind, size))
            if len(out) >= limit:
                break
    rank = {'indexed': 0, 'headerless': 1, 'rpf': 2, 'rpf_noheader': 3}
    return sorted(out, key=lambda r: (rank[r[1]], r[0].lower()))


def texture_role(img, fmt):
    """Rough guess at what a texture is for, so odd-looking data maps get explained."""
    import numpy as np
    a = np.asarray(img.convert('RGBA').resize((64, 64)), np.float32).reshape(-1, 4)
    m, sd = a.mean(0), a.std(0)
    if fmt == 'BC5' or (abs(m[0] - 128) < 30 and abs(m[1] - 128) < 30 and m[2] > 170):
        return 'Normal map'
    if fmt == 'BC4' or (np.abs(a[:, 0] - a[:, 1]).mean() < 3 and np.abs(a[:, 1] - a[:, 2]).mean() < 3):
        return 'Greyscale'
    live = [i for i in range(3) if sd[i] > 4]
    if len(live) >= 2:
        c = np.corrcoef(a[:, live].T)
        if np.nanmax(np.abs(c[np.triu_indices(len(live), 1)])) < 0.5:
            return 'Packed data'
    return 'Colour'


def human(n):
    return (f'{n / 2**30:.1f} GB' if n >= 2**30 else f'{n / 2**20:.0f} MB' if n >= 2**20
            else f'{n / 1024:.0f} KB' if n >= 1024 else f'{n} B')


def make_icon(n=256):
    """Flat mark: a 3x3 tile grid (a texture page) whose lit tiles spell a T."""
    from PIL import Image, ImageDraw
    s = n * 4  # draw large, downscale for smooth edges
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.2), fill=(20, 22, 28, 255))
    m, g = int(s * 0.19), int(s * 0.045)
    cell = (s - 2 * m - 2 * g) / 3
    for r in range(3):
        for c in range(3):
            x, y = m + c * (cell + g), m + r * (cell + g)
            lit = r == 0 or c == 1
            d.rounded_rectangle((x, y, x + cell, y + cell), radius=int(cell * 0.16),
                                fill=(91, 140, 255, 255) if lit else (52, 56, 66, 255))
    return img.resize((n, n), Image.LANCZOS)


def install():
    """Copy the app to %LOCALAPPDATA%\\Programs\\TTPF Studio and add Start menu + Desktop shortcuts."""
    import subprocess
    dest = os.path.join(os.environ['LOCALAPPDATA'], 'Programs', 'TTPF Studio')
    os.makedirs(dest, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ('ttpf_tool.py', 'texcodec.py', 'library.py', 'models.py'):
        if os.path.abspath(os.path.join(here, f)) != os.path.abspath(os.path.join(dest, f)):
            shutil.copy2(os.path.join(here, f), os.path.join(dest, f))
    ico = os.path.join(dest, 'icon.ico')
    make_icon(256).save(ico, sizes=[(x, x) for x in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
    pyw = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    if not os.path.exists(pyw):
        pyw = sys.executable
    ps = ("$s = New-Object -ComObject WScript.Shell\n"
          "foreach ($p in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) {\n"
          "  $f = Join-Path $p 'TTPF Studio.lnk'; $l = $s.CreateShortcut($f)\n"
          "  $l.TargetPath = $env:T_PYW; $l.Arguments = '\"' + $env:T_SCRIPT + '\"'; $l.WorkingDirectory = $env:T_DEST\n"
          "  $l.IconLocation = $env:T_ICO + ',0'; $l.Description = 'View, export and replace textures in .ttpf packs'\n"
          "  $l.Save(); Write-Output $f }")
    env = dict(os.environ, T_PYW=pyw, T_SCRIPT=os.path.join(dest, 'ttpf_tool.py'), T_DEST=dest, T_ICO=ico)
    r = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    print('Installed to', dest)
    print(r.stdout.strip())


# ====================================================================== GUI
C = dict(bg='#14161b', panel='#1c1f26', panel2='#232731', line='#2c313c', text='#e6e8ee', dim='#8b93a7',
         accent='#5b8cff', accent2='#3f6fe0', ok='#4cc38a', warn='#e5a83b', bad='#e5534b', sel='#2a3550')
STATUS = {'full': ('●', 'ok', 'Full resolution'), 'low': ('◐', 'warn', 'Low-res mips only'),
          'missing': ('○', 'dim', 'No data in this file'), 'unsupported': ('○', 'dim', 'Unsupported format'),
          'unknown': ('◌', 'dim', 'Not analysed yet'), 'guess': ('◆', 'accent', 'Format guessed'),
          'fixed': ('◆', 'accent', 'Auto-corrected: data did not match its table entry')}


ITEM_ICON = {'xml': '‹›', 'text': '≡', 'video': '▶', 'psin': '◆', 'skel': '✦', 'bin': '■'}
ITEM_COL = {'xml': '#7aa2ff', 'text': '#c8ccd6', 'video': '#e5a83b', 'psin': '#4cc38a', 'skel': '#c792ea', 'bin': '#8b93a7'}


def hexdump(data, limit=4096):
    rows = []
    for o in range(0, min(len(data), limit), 16):
        chunk = data[o:o + 16]
        rows.append(f'{o:08x}  {chunk.hex(" "):<48}  ' + ''.join(chr(c) if 32 <= c < 127 else '·' for c in chunk))
    return '\n'.join(rows)


def run_gui(path=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title('TTPF Studio')
    root.geometry('1440x880')
    root.minsize(1000, 620)
    root.configure(bg=C['bg'])
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    FONT = ('Segoe UI', 10)
    st = dict(pack=None, tex=None, img=None, zoom=1.0, ox=0, oy=0, chan='RGBA', photo=None, job=None, drag=None, level=0,
              view='viewer', folder=None, gen=0, arc=None, item=None, path=None, mode='tex')  # gen: bumps when the texture list is rebuilt, so stale thumbnails are dropped
    settings = load_settings()
    if settings.get('geometry'):
        root.geometry(settings['geometry'])
    if settings.get('zoomed'):
        root.state('zoomed')
    st['icon'] = ImageTk.PhotoImage(make_icon(64))
    root.iconphoto(True, st['icon'])

    def report(exc, val, tb):
        log_error(''.join(traceback.format_exception(exc, val, tb)))
        messagebox.showerror('Something went wrong', f'{val}\n\nDetails were written to\n{LOG}')
    root.report_callback_exception = report

    # ---------- styles
    s = ttk.Style()
    s.theme_use('clam')
    s.configure('.', background=C['panel'], foreground=C['text'], fieldbackground=C['panel2'], bordercolor=C['line'],
                lightcolor=C['panel'], darkcolor=C['panel'], troughcolor=C['panel'], font=FONT, focuscolor=C['accent'])
    s.configure('TFrame', background=C['panel'])
    s.configure('Bg.TFrame', background=C['bg'])
    s.configure('Head.TFrame', background=C['bg'])
    s.configure('TLabel', background=C['panel'], foreground=C['text'])
    s.configure('Dim.TLabel', foreground=C['dim'])
    s.configure('Head.TLabel', background=C['bg'], foreground=C['text'], font=('Segoe UI Semibold', 15))
    s.configure('HeadDim.TLabel', background=C['bg'], foreground=C['dim'])
    s.configure('Sec.TLabel', foreground=C['dim'], font=('Segoe UI Semibold', 9))
    s.configure('Val.TLabel', font=('Consolas', 10))
    s.configure('TButton', background=C['panel2'], foreground=C['text'], borderwidth=0, padding=(12, 7))
    s.map('TButton', background=[('pressed', C['line']), ('active', C['line']), ('disabled', C['panel'])],
          foreground=[('disabled', C['dim'])])
    s.configure('Accent.TButton', background=C['accent'], foreground='white')
    s.map('Accent.TButton', background=[('pressed', C['accent2']), ('active', C['accent2']), ('disabled', C['panel2'])])
    s.configure('Seg.TButton', padding=(9, 4), background=C['panel2'])
    s.configure('SegOn.TButton', padding=(9, 4), background=C['accent'], foreground='white')
    s.map('SegOn.TButton', background=[('active', C['accent2'])])
    s.configure('Treeview', background=C['panel'], fieldbackground=C['panel'], foreground=C['text'], rowheight=26,
                borderwidth=0)
    s.map('Treeview', background=[('selected', C['sel'])], foreground=[('selected', 'white')])
    s.configure('Treeview.Heading', background=C['panel2'], foreground=C['dim'], borderwidth=0, padding=(6, 5),
                font=('Segoe UI Semibold', 9))
    s.map('Treeview.Heading', background=[('active', C['line'])])
    s.configure('TEntry', fieldbackground=C['panel2'], foreground=C['text'], insertcolor=C['text'], padding=6, borderwidth=0)
    s.configure('TCombobox', fieldbackground=C['panel2'], background=C['panel2'], foreground=C['text'], arrowcolor=C['dim'],
                padding=4, borderwidth=0)
    s.map('TCombobox', fieldbackground=[('readonly', C['panel2'])], foreground=[('readonly', C['text'])])
    root.option_add('*TCombobox*Listbox.background', C['panel2'])
    root.option_add('*TCombobox*Listbox.foreground', C['text'])
    root.option_add('*TCombobox*Listbox.selectBackground', C['accent'])
    s.configure('Vertical.TScrollbar', background=C['panel2'], troughcolor=C['panel'], borderwidth=0, arrowcolor=C['dim'])
    s.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['panel2'], borderwidth=0, thickness=6)
    s.configure('TCheckbutton', background=C['panel'], foreground=C['text'], indicatorcolor=C['panel2'])
    s.map('TCheckbutton', indicatorcolor=[('selected', C['accent'])], background=[('active', C['panel'])])

    # ---------- header
    head = ttk.Frame(root, style='Head.TFrame', padding=(16, 12))
    head.pack(fill='x')
    ttk.Label(head, text='TTPF Studio', style='Head.TLabel').pack(side='left')
    file_lbl = ttk.Label(head, text='No file open', style='HeadDim.TLabel')
    file_lbl.pack(side='left', padx=(18, 8))
    chip = tk.Label(head, text='', bg=C['bg'], fg='white', font=('Segoe UI Semibold', 9), padx=8, pady=2)
    chip.pack(side='left')
    save_btn = ttk.Button(head, text='Save modded pack…', style='Accent.TButton')
    save_btn.pack(side='right')
    edits_lbl = ttk.Label(head, text='', style='HeadDim.TLabel')
    edits_lbl.pack(side='right', padx=10)
    open_btn = ttk.Button(head, text='Open file…')
    open_btn.pack(side='right', padx=(0, 8))
    folder_btn = ttk.Button(head, text='Open folder…')
    folder_btn.pack(side='right', padx=(0, 6))
    lib_btn = ttk.Button(head, text='Library', style='Accent.TButton')
    lib_btn.pack(side='right', padx=(0, 6))
    mine_btn = ttk.Button(head, text='Export everything…')
    mine_btn.pack(side='right', padx=(0, 6))
    help_btn = ttk.Button(head, text='?', width=3)
    help_btn.pack(side='right', padx=(0, 6))

    body = ttk.Frame(root, style='Bg.TFrame', padding=(12, 0, 12, 0))
    body.pack(fill='both', expand=True)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    # ---------- left: list
    left = ttk.Frame(body, padding=10)
    left.grid(row=0, column=0, sticky='nsw', padx=(0, 10))
    # packs in the opened folder (hidden until a folder is opened)
    pk_frame = ttk.Frame(left)
    pk_head = ttk.Frame(pk_frame)
    pk_head.pack(fill='x')
    ttk.Label(pk_head, text='PACKS', style='Sec.TLabel').pack(side='left')
    folder_lbl = ttk.Label(pk_head, text='', style='Dim.TLabel')
    folder_lbl.pack(side='left', padx=8)
    pk_tf = ttk.Frame(pk_frame)
    pk_tf.pack(fill='x', pady=(6, 14))
    packs = ttk.Treeview(pk_tf, columns=('st', 'name', 'size'), show='headings', selectmode='browse', height=8)
    for c, t_, wd, an in (('st', 'Type', 104, 'w'), ('name', 'File', 130, 'w'), ('size', 'Size', 68, 'e')):
        packs.heading(c, text=t_)
        packs.column(c, width=wd, anchor=an, stretch=c == 'name')
    for tag, col in (('indexed', C['ok']), ('headerless', C['warn']), ('rpf', C['accent']), ('rpf_noheader', C['accent'])):
        packs.tag_configure(tag, foreground=col)
    pk_vs = ttk.Scrollbar(pk_tf, command=packs.yview)
    packs.configure(yscrollcommand=pk_vs.set)
    packs.pack(side='left', fill='x', expand=True)
    pk_vs.pack(side='right', fill='y')
    tex_lbl = ttk.Label(left, text='TEXTURES', style='Sec.TLabel')
    tex_lbl.pack(anchor='w')
    fl = ttk.Frame(left)
    fl.pack(fill='x', pady=(6, 8))
    q_var = tk.StringVar()
    q_entry = ttk.Entry(fl, textvariable=q_var, width=18)
    q_entry.pack(side='left', fill='x', expand=True)
    show_var = tk.StringVar(value='Has data')
    show_cb = ttk.Combobox(fl, textvariable=show_var, state='readonly', width=15,
                           values=('Has data', 'Full res only', 'All'))
    show_cb.pack(side='left', padx=(6, 0))
    deep_var = tk.BooleanVar(value=False)
    deep_cb = ttk.Checkbutton(left, text='Search inside files (XML / text)', variable=deep_var)
    tf = ttk.Frame(left)
    tf.pack(fill='both', expand=True)
    cols = ('st', 'name', 'size', 'fmt')
    tree = ttk.Treeview(tf, columns=cols, show='headings', selectmode='extended', height=20)
    for c, t, wd, an in (('st', '', 28, 'center'), ('name', 'Name', 170, 'w'), ('size', 'Size', 84, 'e'),
                         ('fmt', 'Format', 64, 'w')):
        tree.heading(c, text=t)
        tree.column(c, width=wd, anchor=an, stretch=c == 'name')
    for k, (_, col, _) in STATUS.items():
        tree.tag_configure(k, foreground=C[col] if col != 'dim' else C['dim'])
    tree.tag_configure('edited', foreground=C['accent'])
    for k_, col_ in ITEM_COL.items():
        tree.tag_configure('k_' + k_, foreground=col_)
    vs = ttk.Scrollbar(tf, command=tree.yview)
    tree.configure(yscrollcommand=vs.set)
    tree.pack(side='left', fill='both', expand=True)
    vs.pack(side='right', fill='y')
    count_lbl = ttk.Label(left, text='', style='Dim.TLabel')
    count_lbl.pack(anchor='w', pady=(8, 0))

    # ---------- center: viewer
    mid = ttk.Frame(body, padding=0)
    mid.grid(row=0, column=1, sticky='nsew')
    vbar = ttk.Frame(mid, padding=(10, 8))
    vbar.pack(fill='x')
    view_btns = {}
    for v_, label_ in (('viewer', 'Viewer'), ('grid', 'Grid')):
        b = ttk.Button(vbar, text=label_, style='Seg.TButton', width=7, command=lambda v=v_: set_view(v))
        b.pack(side='left', padx=(0, 2))
        view_btns[v_] = b
    ttk.Frame(vbar, width=18).pack(side='left')
    chan_btns = {}
    for ch in ('RGBA', 'RGB', 'R', 'G', 'B', 'A', 'Split'):
        b = ttk.Button(vbar, text=ch, style='Seg.TButton', width=5, command=lambda c=ch: set_chan(c))
        b.pack(side='left', padx=(0, 2))
        chan_btns[ch] = b
    ttk.Button(vbar, text='1:1', style='Seg.TButton', width=4, command=lambda: set_zoom(1.0)).pack(side='right')
    bg_btn = ttk.Button(vbar, style='Seg.TButton', width=9)
    bg_btn.pack(side='right', padx=(0, 8))
    ttk.Button(vbar, text='Fit', style='Seg.TButton', width=4, command=lambda: fit()).pack(side='right', padx=2)
    zoom_lbl = ttk.Label(vbar, text='', style='Dim.TLabel')
    zoom_lbl.pack(side='right', padx=10)
    canvas = tk.Canvas(mid, bg='#0f1115', highlightthickness=0, cursor='crosshair')
    canvas.pack(fill='both', expand=True)
    pix_lbl = ttk.Label(mid, text='', style='Dim.TLabel', padding=(10, 6))
    pix_lbl.pack(fill='x')
    YT_BG, YT_RED = '#0b0c0f', '#ff2d46'
    vid_frame = tk.Frame(mid, bg='#000000')
    vid_host = tk.Frame(vid_frame, bg='#000000', cursor='hand2')
    vid_host.pack(fill='both', expand=True)
    vid_clip = tk.Frame(vid_host, bg='#000000', cursor='hand2')  # exactly the visible picture; clips the player's overlay
    vid_msg = tk.Label(vid_host, bg='#000000', fg='#c8ccd6', font=('Segoe UI', 12), justify='center', cursor='hand2')
    vid_msg.place(relx=0.5, rely=0.5, anchor='center')
    ctl = tk.Frame(vid_frame, bg=YT_BG)
    ctl.pack(fill='x')
    seek_cv = tk.Canvas(ctl, height=20, bg=YT_BG, highlightthickness=0, cursor='hand2')
    seek_cv.pack(fill='x', padx=12, pady=(6, 0))
    ctl_row = tk.Frame(ctl, bg=YT_BG)
    ctl_row.pack(fill='x', padx=6, pady=(0, 8))

    def ctl_button(text, tip, side='left', size=15):
        b = tk.Label(ctl_row, text=text, bg=YT_BG, fg='white', font=('Segoe UI Symbol', size), padx=9, cursor='hand2')
        b.pack(side=side)
        b.bind('<Enter>', lambda e: (b.configure(fg=YT_RED), status.configure(text=tip)))
        b.bind('<Leave>', lambda e: b.configure(fg='white'))
        return b
    vb_play = ctl_button('▶', 'Play / pause  (Space or K)')
    vb_back = ctl_button('↺', 'Back 10 seconds  (J)', size=14)
    vb_fwd = ctl_button('↻', 'Forward 10 seconds  (L)', size=14)
    vb_time = tk.Label(ctl_row, text='0:00 / 0:00', bg=YT_BG, fg='#e6e8ee', font=('Segoe UI', 10), padx=8)
    vb_time.pack(side='left')
    vb_full = ctl_button('⛶', 'Full screen  (F, Esc to leave)', side='right')
    vb_ext = ctl_button('↗', 'Open in a separate RAD Video Tools window', side='right', size=13)
    vb_mp4 = tk.Label(ctl_row, text='MP4', bg=YT_BG, fg='white', font=('Segoe UI Semibold', 9), padx=9, cursor='hand2')
    vb_mp4.pack(side='right')
    vb_mp4.bind('<Enter>', lambda e: (vb_mp4.configure(fg=YT_RED), status.configure(text='Convert this video to MP4')))
    vb_mp4.bind('<Leave>', lambda e: vb_mp4.configure(fg='white'))
    txt_frame = ttk.Frame(mid)
    txt = tk.Text(txt_frame, bg='#0f1115', fg=C['text'], insertbackground=C['text'], font=('Consolas', 10), relief='flat',
                  padx=16, pady=12, wrap='none', borderwidth=0, highlightthickness=0, selectbackground=C['sel'])
    t_vs = ttk.Scrollbar(txt_frame, command=txt.yview)
    t_hs = ttk.Scrollbar(txt_frame, orient='horizontal', command=txt.xview)
    txt.configure(yscrollcommand=t_vs.set, xscrollcommand=t_hs.set)
    t_hs.pack(side='bottom', fill='x')
    t_vs.pack(side='right', fill='y')
    txt.pack(side='left', fill='both', expand=True)
    for tag_, col_ in (('tag', '#7aa2ff'), ('attr', '#e5a83b'), ('com', C['dim']), ('head', C['ok'])):
        txt.tag_configure(tag_, foreground=col_)
    txt.tag_configure('hit', background='#6b5512', foreground='white')
    txt.tag_raise('hit')
    gf = ttk.Frame(mid)
    grid_cv = tk.Canvas(gf, bg='#0f1115', highlightthickness=0, yscrollincrement=24)
    g_vs = ttk.Scrollbar(gf, command=grid_cv.yview)
    grid_cv.pack(side='left', fill='both', expand=True)
    g_vs.pack(side='right', fill='y')

    # ---------- right: inspector
    right = ttk.Frame(body, padding=14, width=300)
    right.grid(row=0, column=2, sticky='nse', padx=(10, 0))
    right.grid_propagate(False)
    right.columnconfigure(1, weight=1)
    r = 0

    def section(title):
        nonlocal r
        ttk.Label(right, text=title, style='Sec.TLabel').grid(row=r, column=0, columnspan=2, sticky='w', pady=(14 if r else 0, 6))
        r += 1

    section('PROPERTIES')
    props = {}
    for key in ('Name', 'Status', 'Looks like', 'Format', 'Tiling', 'Size', 'Showing', 'Mips', 'Blobs', 'Offset'):
        ttk.Label(right, text=key, style='Dim.TLabel').grid(row=r, column=0, sticky='w', pady=1)
        v = ttk.Label(right, text='—', style='Val.TLabel')
        v.grid(row=r, column=1, sticky='w', padx=(10, 0), pady=1)
        props[key] = v
        r += 1

    section('DECODE AS')
    ov = ttk.Frame(right)
    ov.grid(row=r, column=0, columnspan=2, sticky='ew')
    r += 1
    fmt_var, size_var = tk.StringVar(), tk.StringVar()
    fmt_cb = ttk.Combobox(ov, textvariable=fmt_var, state='readonly', width=8, values=tc.GUESS_ORDER + ('BC2', 'BC6H', 'BGRA8'))
    fmt_cb.pack(side='left')
    size_cb = ttk.Combobox(ov, textvariable=size_var, state='readonly', width=12)
    size_cb.pack(side='left', padx=6)
    til_var = tk.StringVar()
    til_cb = ttk.Combobox(right, textvariable=til_var, state='readonly', values=[tc.TILING_NAMES[k] for k in tc.TILINGS])
    til_cb.grid(row=r, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    r += 1
    ov_note = ttk.Label(right, text='Auto-detected. Change Tiling if an image looks scrambled, '
                        'or Format / Size if it looks like noise.', style='Dim.TLabel', wraplength=260)
    ov_note.grid(row=r, column=0, columnspan=2, sticky='w', pady=(4, 0))
    r += 1

    section('EXPORT')
    ex = ttk.Frame(right)
    ex.grid(row=r, column=0, columnspan=2, sticky='ew')
    r += 1
    btn_png = ttk.Button(ex, text='PNG')
    btn_png.pack(side='left', fill='x', expand=True)
    btn_dds = ttk.Button(ex, text='DDS')
    btn_dds.pack(side='left', fill='x', expand=True, padx=(6, 0))
    btn_all = ttk.Button(right, text='Export all to folder…')
    btn_all.grid(row=r, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    r += 1

    section('REPLACE')
    im = ttk.Frame(right)
    im.grid(row=r, column=0, columnspan=2, sticky='ew')
    r += 1
    btn_ipng = ttk.Button(im, text='Import PNG')
    btn_ipng.pack(side='left', fill='x', expand=True)
    btn_idds = ttk.Button(im, text='Import DDS')
    btn_idds.pack(side='left', fill='x', expand=True, padx=(6, 0))
    btn_rev = ttk.Button(right, text='Revert this texture')
    btn_rev.grid(row=r, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    r += 1
    ttk.Label(right, text='BC7 / BC4 / BC6H: edit an exported DDS in Paint.NET, GIMP or texconv, keep the '
              'same format and size, then Import DDS. Edits are only written when you save.',
              style='Dim.TLabel', wraplength=262, justify='left').grid(row=r, column=0, columnspan=2, sticky='w', pady=(10, 0))
    r += 1

    right_arc = ttk.Frame(body, padding=14, width=300)
    right_arc.grid_propagate(False)
    right_arc.columnconfigure(1, weight=1)
    ttk.Label(right_arc, text='FILE', style='Sec.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 6))
    aprops = {}
    for i_, key in enumerate(('Name', 'Type', 'Size', 'Stored', 'Chunks', 'Offset', 'Details'), start=1):
        ttk.Label(right_arc, text=key, style='Dim.TLabel').grid(row=i_, column=0, sticky='nw', pady=1)
        v = ttk.Label(right_arc, text='—', style='Val.TLabel', wraplength=180, justify='left')
        v.grid(row=i_, column=1, sticky='w', padx=(10, 0), pady=1)
        aprops[key] = v
    ttk.Label(right_arc, text='ACTIONS', style='Sec.TLabel').grid(row=8, column=0, columnspan=2, sticky='w', pady=(16, 6))
    a_play = ttk.Button(right_arc, text='▶  Play video', style='Accent.TButton')
    a_play.grid(row=9, column=0, columnspan=2, sticky='ew')
    a_conv = ttk.Button(right_arc, text='Convert to MP4…')
    a_conv.grid(row=10, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    a_save = ttk.Button(right_arc, text='Save this file…')
    a_save.grid(row=11, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    a_type = ttk.Button(right_arc, text='Save all of this type…')
    a_type.grid(row=12, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    a_all = ttk.Button(right_arc, text='Save everything in this archive…')
    a_all.grid(row=13, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    ttk.Label(right_arc, text='ARCHIVE', style='Sec.TLabel').grid(row=14, column=0, columnspan=2, sticky='w', pady=(16, 6))
    a_head = ttk.Button(right_arc, text='Header details')
    a_head.grid(row=15, column=0, columnspan=2, sticky='ew')
    a_models = ttk.Button(right_arc, text='3D models in this archive…', style='Accent.TButton')
    a_models.grid(row=16, column=0, columnspan=2, sticky='ew', pady=(6, 0))
    ttk.Label(right_arc, text='RPF9 file lists are encrypted, so original names and folders are not available. '
              'These files are recovered from the data that is not encrypted; names come from their content '
              '(XML root element, video size). Bink 2 videos play in RAD Video Tools, not VLC.',
              style='Dim.TLabel', wraplength=262, justify='left').grid(row=17, column=0, columnspan=2, sticky='w', pady=(12, 0))

    # ---------- status bar
    sb = ttk.Frame(root, style='Head.TFrame', padding=(16, 6))
    sb.pack(fill='x')
    status = ttk.Label(sb, text='Open a .ttpf texture pack (or an .rpf to read its header).', style='HeadDim.TLabel')
    status.pack(side='left')
    prog = ttk.Progressbar(sb, length=220, mode='determinate')

    # ---------- background jobs (tk is single-threaded: workers post to a queue)
    jobs = queue.Queue()

    def run_bg(label, work, done):
        if st['job']:
            return messagebox.showinfo('Busy', 'Please wait for the current task to finish.')
        st['job'] = label
        status.configure(text=label)
        prog.pack(side='right')
        prog['value'] = 0

        def wrap():
            try:
                res = work(lambda f: jobs.put(('p', f)))
                jobs.put(('ok', done, res))
            except Exception as ex_:
                jobs.put(('err', label, ex_))
        threading.Thread(target=wrap, daemon=True).start()

    def pump():
        try:
            while True:
                msg = jobs.get_nowait()
                if msg[0] == 'p':
                    prog['value'] = max(0, min(100, msg[1] * 100))
                    continue
                if msg[0] == 'msg':
                    status.configure(text=msg[1])
                    continue
                st['job'] = None
                prog.pack_forget()
                if msg[0] == 'ok':
                    msg[1](msg[2])
                else:
                    status.configure(text='Failed: ' + str(msg[2]))
                    messagebox.showerror(msg[1], str(msg[2]))
        except queue.Empty:
            pass
        root.after(80, pump)
    pump()

    # ---------- list
    def size_text(t):
        return f'{t.w}×{t.h}' if t.w else f'{t.chain[0].length // 1024} KB' if t.chain else ''

    def refill():
        p = st['pack']
        tree.delete(*tree.get_children())
        if st['mode'] == 'arc' and st['arc']:
            a, q, mode, n = st['arc'], q_var.get().strip().lower(), show_var.get(), 0
            hits = None
            if q and deep_var.get():
                status.configure(text=f'Searching inside {sum(i.kind in ("xml", "text") for i in a.items)} files for "{q}"…')
                root.update_idletasks()
                hits = set(a.search(q))
            for it in a.items:
                kind = ITEM_KINDS[it.kind]
                if mode != 'All' and kind != mode:
                    continue
                row = (ITEM_ICON[it.kind], it.name, human(it.length), kind)
                if hits is not None:
                    if it.n not in hits:
                        continue
                elif q and q not in ' '.join(row).lower():
                    continue
                tree.insert('', 'end', iid=str(it.n), values=row, tags=('k_' + it.kind,))
                n += 1
            count_lbl.configure(text=f'{n} shown  ·  {len(a.items)} files recovered'
                                     + (f'  ·  "{q}" found in {len(hits)} files' if hits is not None else ''))
            if hits is not None:
                status.configure(text=f'"{q}" found inside {len(hits)} files')
            return
        if not p:
            return
        q = q_var.get().strip().lower()
        mode = show_var.get()
        n = 0
        for i, t in enumerate(p.textures):
            if mode in ('Has data', 'One per texture') and t.status in ('missing', 'unsupported'):
                continue
            if mode == 'Full res only' and t.status not in ('full', 'guess', 'unknown', 'fixed'):
                continue
            row = (STATUS[t.status][0], t.name, size_text(t), t.fmt or '?')
            if q and q not in ' '.join(map(str, row)).lower():
                continue
            tree.insert('', 'end', iid=str(i), values=row, tags=('edited',) if p.edited(t) else (t.status,))
            n += 1
        full = sum(t.status in ('full', 'guess', 'unknown', 'fixed') for t in p.textures)
        count_lbl.configure(text=f'{n} shown  ·  {len(p.textures)} total  ·  {full} with full-size data')
        if st['view'] == 'grid':
            grid_layout()

    def update_row(t):
        iid = str(t.n)
        if tree.exists(iid):
            tree.item(iid, values=(STATUS[t.status][0], t.name, size_text(t), t.fmt or '?'),
                      tags=('edited',) if st['pack'].edited(t) else (t.status,))
        n = len(st['pack'].edits)
        edits_lbl.configure(text=f'{n} unsaved change{"s" * (n != 1)}' if n else '')

    def on_query(*_):
        if st.get('q_after'):
            root.after_cancel(st['q_after'])
        st['q_after'] = root.after(350 if deep_var.get() and st['mode'] == 'arc' else 0, refill)
    q_var.trace_add('write', on_query)
    deep_var.trace_add('write', on_query)
    def on_show(_=None):
        p = st['pack']
        if p and p.carved and p.grouped != (show_var.get() == 'One per texture'):
            p.set_grouping(not p.grouped)
            st['gen'] += 1
            thumbs.clear()
            thumb_req.clear()
            st.update(tex=None, img=None)
            gstate['sel'] = None
        refill()
    show_cb.bind('<<ComboboxSelected>>', on_show)

    # ---------- viewer
    checker = {}

    def checker_bg(w, h):
        mode_ = settings.get('viewer_bg', 'checker')
        if mode_ != 'checker':
            return Image.new('RGB', (w, h), '#000000' if mode_ == 'black' else '#ffffff')
        key = (w, h)
        if key not in checker:
            checker.clear()
            tile_ = Image.new('RGB', (16, 16), '#3a3d45')
            tile_.paste('#2c2f36', (0, 0, 8, 8))
            tile_.paste('#2c2f36', (8, 8, 16, 16))
            bg = Image.new('RGB', (w, h))
            for yy in range(0, h, 16):
                for xx in range(0, w, 16):
                    bg.paste(tile_, (xx, yy))
            checker[key] = bg
        return checker[key]

    def channel_view(img):
        ch = st['chan']
        if ch == 'RGBA':
            return img
        if ch == 'RGB':
            return img.convert('RGB').convert('RGBA')
        band = img.getchannel(ch)
        return Image.merge('RGBA', (band, band, band, Image.new('L', img.size, 255)))

    def disp():
        """What the viewer shows: the texture, or a 2x2 sheet of its R/G/B/A channels in Split mode."""
        img = st['img']
        if img is None or st['chan'] != 'Split':
            return img
        if st.get('split_src') is not img:
            w, h = img.size
            sheet = Image.new('L', (w * 2, h * 2))
            for k, band in enumerate(img.split()):
                sheet.paste(band, ((k % 2) * w, (k // 2) * h))
            st['split'], st['split_src'] = sheet.convert('RGBA'), img
        return st['split']

    def redraw():
        canvas.delete('all')
        img = disp()
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if img is None and st['pack']:
            canvas.create_text(cw // 2, ch // 2, text='Select a texture', fill=C['dim'], font=('Segoe UI', 13))
            return
        if img is None:
            y = ch // 2 - 120
            canvas.create_image(cw // 2, y, image=st['icon'])
            canvas.create_text(cw // 2, y + 58, text='TTPF Studio', fill=C['text'], font=('Segoe UI Semibold', 22))
            canvas.create_text(cw // 2, y + 92, fill=C['dim'], font=('Segoe UI', 11),
                               text='Open a game folder (Ctrl+Shift+O) or a single .ttpf / .rpf file (Ctrl+O)')
            rec = [f for f in settings.get('recent', []) if os.path.exists(f)][:6]
            if rec:
                canvas.create_text(cw // 2, y + 138, text='RECENT', fill=C['dim'], font=('Segoe UI Semibold', 9))
            for k, f in enumerate(rec):
                tag = f'recent{k}'
                canvas.create_text(cw // 2, y + 164 + k * 26, text=os.path.basename(f), fill=C['accent'],
                                   font=('Segoe UI', 11, 'underline'), tags=(tag,), activefill='white')
                canvas.tag_bind(tag, '<Button-1>', lambda e, f=f: open_file(f))
            return
        z = st['zoom']
        iw, ih = img.size
        dw, dh = iw * z, ih * z
        x0, y0 = (cw - dw) / 2 + st['ox'], (ch - dh) / 2 + st['oy']
        # visible part only, so huge textures stay fast
        vx0, vy0 = max(0, int(-x0 / z)), max(0, int(-y0 / z))
        vx1, vy1 = min(iw, int((cw - x0) / z) + 1), min(ih, int((ch - y0) / z) + 1)
        if vx1 <= vx0 or vy1 <= vy0:
            return
        crop = img.crop((vx0, vy0, vx1, vy1))
        crop = crop if st['chan'] == 'Split' else channel_view(crop)
        tw, th = max(1, int((vx1 - vx0) * z)), max(1, int((vy1 - vy0) * z))
        crop = crop.resize((tw, th), Image.NEAREST if z >= 1 else Image.BILINEAR)
        bg = checker_bg(tw, th).copy()
        bg.paste(crop, (0, 0), crop)
        st['photo'] = ImageTk.PhotoImage(bg)
        px, py = x0 + vx0 * z, y0 + vy0 * z
        canvas.create_image(px, py, image=st['photo'], anchor='nw')
        canvas.create_rectangle(x0 - 1, y0 - 1, x0 + dw, y0 + dh, outline=C['line'])
        if st['chan'] == 'Split':
            canvas.create_line(x0 + dw / 2, y0, x0 + dw / 2, y0 + dh, fill=C['bg'], width=3)
            canvas.create_line(x0, y0 + dh / 2, x0 + dw, y0 + dh / 2, fill=C['bg'], width=3)
            for k, (name, col) in enumerate((('R  red', '#ff6b6b'), ('G  green', '#5fd38d'), ('B  blue', '#6ea8ff'), ('A  alpha', '#d0d4dc'))):
                lx, ly = x0 + (k % 2) * dw / 2 + 10, y0 + (k // 2) * dh / 2 + 10
                canvas.create_rectangle(lx - 4, ly - 2, lx + 74, ly + 20, fill=C['bg'], outline='')
                canvas.create_text(lx, ly + 9, text=name, anchor='w', fill=col, font=('Segoe UI Semibold', 10))
        zoom_lbl.configure(text=f'{z * 100:.0f}%')

    def fit():
        img = disp()
        if img is None:
            return
        cw, ch = canvas.winfo_width() - 40, canvas.winfo_height() - 40
        st['zoom'] = max(0.02, min(cw / img.width, ch / img.height, 8))
        st['ox'] = st['oy'] = 0
        redraw()

    def set_zoom(z, about=None):
        img = disp()
        if img is None:
            return
        old = st['zoom']
        z = max(0.02, min(z, 32))
        if about:
            cx, cy = canvas.winfo_width() / 2 + st['ox'], canvas.winfo_height() / 2 + st['oy']
            st['ox'] += (about[0] - cx) * (1 - z / old)
            st['oy'] += (about[1] - cy) * (1 - z / old)
        else:
            st['ox'] = st['oy'] = 0
        st['zoom'] = z
        redraw()

    def set_chan(c):
        st['chan'] = c
        for k, b in chan_btns.items():
            b.configure(style='SegOn.TButton' if k == c else 'Seg.TButton')
        fit() if (c == 'Split') != (st.get('was_split', False)) else redraw()
        st['was_split'] = c == 'Split'
    set_chan('RGBA')

    canvas.bind('<Configure>', lambda _: redraw())
    canvas.bind('<MouseWheel>', lambda e: set_zoom(st['zoom'] * (1.2 if e.delta > 0 else 1 / 1.2), (e.x, e.y)))
    canvas.bind('<ButtonPress-1>', lambda e: st.update(drag=(e.x, e.y)))

    def on_drag(e):
        if st['drag']:
            st['ox'] += e.x - st['drag'][0]
            st['oy'] += e.y - st['drag'][1]
            st['drag'] = (e.x, e.y)
            redraw()
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<Double-Button-1>', lambda _: fit())

    def on_move(e):
        img, d = st['img'], disp()
        if img is None:
            return
        z = st['zoom']
        x0 = (canvas.winfo_width() - d.width * z) / 2 + st['ox']
        y0 = (canvas.winfo_height() - d.height * z) / 2 + st['oy']
        x, y = int((e.x - x0) / z), int((e.y - y0) / z)
        if 0 <= x < d.width and 0 <= y < d.height:
            x, y = x % img.width, y % img.height
            rr, gg, bb, aa = img.getpixel((x, y))
            pix_lbl.configure(text=f'x {x}  y {y}     R {rr}  G {gg}  B {bb}  A {aa}     #{rr:02x}{gg:02x}{bb:02x}')
    canvas.bind('<Motion>', on_move)

    # ---------- selection
    def show_props(t, level=None):
        p = st['pack']
        b = t.best() if t.fmt else None
        props['Name'].configure(text=t.name)
        props['Status'].configure(text=STATUS[t.status][2] + (' · edited' if p.edited(t) else ''),
                                  foreground=C[STATUS[t.status][1]] if STATUS[t.status][1] != 'dim' else C['dim'])
        props['Format'].configure(text=(t.fmt or '?') + (' (sRGB)' if t.dxgi in tc.SRGB else '') + (' · guessed' if t.guessed else ''))
        props['Tiling'].configure(text=tc.TILING_NAMES.get(t.tiling, t.tiling) if t.fmt else '—')
        props['Size'].configure(text=f'{t.w} × {t.h}' if t.w else '?')
        props['Showing'].configure(text=f'{b[1]} × {b[2]}' + (f'  (mip {b[3]})' if b[3] else '') if b else '—')
        props['Mips'].configure(text=str(t.mips) if not p.carved else '—')
        props['Blobs'].configure(text=f'{len(t.chain)}' + (f'  ({sum(e.length or 0 for e in t.chain) // 1024} KB)' if t.chain else ''))
        props['Offset'].configure(text=f'0x{t.chain[-1].off:x}' if t.chain else '—')
        props['Looks like'].configure(text=texture_role(st['img'], t.fmt) if st['img'] is not None else '—')
        for w_ in (fmt_cb, size_cb):
            w_.configure(state='readonly' if t.chain else 'disabled')
        fmt_var.set(t.fmt or '')
        size_var.set(f'{t.w} × {t.h}' if t.w else '')
        if t.fmt and t.chain:
            sizes = tc.guess_sizes(t.fmt, t.chain[-1].length)
            size_cb.configure(values=[f'{a} × {b_}' for a, b_ in sizes])
        til_var.set(tc.TILING_NAMES.get(t.tiling, t.tiling))
        til_cb.configure(state='readonly' if t.fmt else 'disabled')
        has = bool(b)
        for w_ in (btn_png, btn_dds, btn_copy, btn_ipng, btn_idds):
            w_.state(['!disabled'] if has else ['disabled'])
        btn_rev.state(['!disabled'] if p.edited(t) else ['disabled'])

    def load_texture(t, refit=True):
        p = st['pack']
        st['tex'] = t
        try:
            img, lvl = p.image(t)
        except Exception as ex_:
            img, lvl = None, None
            status.configure(text=f'Could not decode {t.name}: {ex_}')
        st['img'] = img
        update_row(t)
        show_props(t)
        if refit:
            fit()
        else:
            redraw()
        if img is not None:
            role = texture_role(img, t.fmt)
            tip = {'Packed data': '  ·  packed data: each channel is a separate map, press Split to see them',
                   'Normal map': '  ·  normal map: surface bumps stored as directions, blue/purple is normal'}.get(role, '')
            status.configure(text=f'{t.name}  ·  {t.fmt}  ·  {img.width}×{img.height}' + tip
                             + ('  ·  auto-detected, change it under Decode As if it looks wrong' if t.guessed else ''))
        elif t.status == 'missing':
            status.configure(text=f'{t.name}: this file has no data for it (that part of the file is empty).')

    def show_item(it):
        import re
        a = st['arc']
        st['item'] = it
        data = a.data(it)
        aprops['Name'].configure(text=it.name)
        aprops['Type'].configure(text=ITEM_KINDS[it.kind], foreground=ITEM_COL[it.kind])
        aprops['Size'].configure(text=human(len(data)))
        aprops['Stored'].configure(text=human(it.clen) + ('' if it.kind == 'video' else '  (zlib)'))
        aprops['Chunks'].configure(text=str(len(it.parts)) if it.kind != 'video' else '—')
        aprops['Offset'].configure(text=f'0x{it.off:x}')
        aprops['Details'].configure(text=it.info or '—')
        for b_ in (a_play, a_conv):
            b_.state(['!disabled'] if it.kind == 'video' else ['disabled'])
        if it.kind == 'video':
            if st['vid'] is not it:
                show_video_panel(it)
            status.configure(text=f'{it.name}  ·  Video  ·  {it.info}  ·  Space to play')
            return
        if st['vid']:
            hide_video_panel()
            txt_frame.pack(fill='both', expand=True)
        txt.configure(state='normal')
        txt.delete('1.0', 'end')
        if it.kind in ('xml', 'text'):
            body_ = data[3:] if data[:3] == b'\xef\xbb\xbf' else data
            cap = 1 << 20  # the Text widget slows down on multi-MB files; the saved file is always complete
            shown = body_[:cap].decode('utf-8', 'replace')
            txt.insert('1.0', shown)
            if len(body_) > cap:
                txt.insert('end', f'\n\n… {human(len(body_) - cap)} more. Use "Save this file" to get all of it.', 'com')
            if it.kind == 'xml':
                head_ = shown[:400_000]
                for pat, tag_ in ((r'<!--.*?-->', 'com'), (r'</?[\w:.\-]+|/?>', 'tag'), (r'\s[\w:.\-]+(?==)', 'attr')):
                    for m_ in re.finditer(pat, head_, re.S):
                        txt.tag_add(tag_, f'1.0+{m_.start()}c', f'1.0+{m_.end()}c')
        else:
            if it.kind == 'video':
                txt.insert('end', f'Bink {"2 " if it.name.endswith(".bk2") else ""}video  ·  {it.info}  ·  {human(it.length)}\n', 'head')
                txt.insert('end', 'Press "Play video" to watch it in your video player (it is converted first, then cached). '
                                  'Needs RAD Video Tools + ffmpeg.\n\n', 'com')
            elif it.kind == 'skel':
                start = data.find(b'SKEL_')
                bones = list(dict.fromkeys(b.decode() for b in re.findall(rb'[A-Za-z][A-Za-z0-9_]{3,40}', data[start:])))
                txt.insert('end', f'Skeleton / rigged model resource (RAGE in-memory layout). {len(bones)} names stored in clear:\n\n', 'com')
                txt.insert('end', '\n'.join(bones[:400]) + '\n\nFirst 4 KB:\n\n', 'head')
            elif it.kind == 'psin':
                txt.insert('end', 'PSIN data: a compiled game resource (format not decoded yet). First 4 KB:\n\n', 'com')
            else:
                txt.insert('end', 'Binary data (format unknown). First 4 KB:\n\n', 'com')
            txt.insert('end', hexdump(data))
        q = q_var.get().strip()
        if q and deep_var.get() and it.kind in ('xml', 'text'):
            pos, n_hits = '1.0', 0
            while n_hits < 2000:
                pos = txt.search(q, pos, 'end', nocase=True)
                if not pos:
                    break
                end = f'{pos}+{len(q)}c'
                txt.tag_add('hit', pos, end)
                if not n_hits:
                    txt.see(pos)
                pos, n_hits = end, n_hits + 1
        txt.configure(state='disabled')
        status.configure(text=f'{it.name}  ·  {ITEM_KINDS[it.kind]}  ·  {human(len(data))}')

    def selected():
        """Selected list rows as ints, in list order."""
        return [int(i) for i in tree.selection()]

    def update_bulk_labels():
        n = len(tree.selection())
        if st['mode'] == 'arc' and st['arc']:
            vids = sum(st['arc'].items[i].kind == 'video' for i in selected())
            a_save.configure(text=f'Save {n} selected files…' if n > 1 else 'Save this file…')
            a_conv.configure(text=f'Convert {vids} videos to MP4…' if vids > 1 else 'Convert to MP4…')
            if vids > 1:
                a_conv.state(['!disabled'])
        else:
            btn_all.configure(text=f'Export {n} selected to folder…' if n > 1 else 'Export all to folder…')

    def on_select(_=None):
        sel = tree.selection()
        if not sel:
            return
        cur_ = tree.focus() if tree.focus() in sel else sel[-1]  # the row just clicked
        if st['mode'] == 'arc':
            show_item(st['arc'].items[int(cur_)])
        else:
            if st['view'] == 'grid':
                grid_select(int(cur_))
            load_texture(st['pack'].textures[int(cur_)], refit=st['view'] == 'viewer')
        update_bulk_labels()
    tree.bind('<<TreeviewSelect>>', on_select)

    def on_override(_=None):
        t = st['tex']
        if not t or not t.chain:
            return
        fmt = fmt_var.get()
        sizes = tc.guess_sizes(fmt, t.chain[-1].length)
        size_cb.configure(values=[f'{a} × {b}' for a, b in sizes])
        try:
            w, h = map(int, size_var.get().replace(' ', '').split('×'))
        except ValueError:
            w = h = 0
        if (w, h) not in sizes:
            w, h = (sizes[0] if sizes else (0, 0))
            for a, b in sizes:
                if a == b:
                    w, h = a, b
        if not w:
            return status.configure(text=f'{fmt} does not fit a blob of {t.chain[0].length} bytes.')
        t.fmt, t.w, t.h, t.guessed = fmt, w, h, False
        t.status = 'guess'
        load_texture(t)
    fmt_cb.bind('<<ComboboxSelected>>', on_override)
    size_cb.bind('<<ComboboxSelected>>', on_override)

    def on_tiling(_=None):
        t = st['tex']
        if t:
            t.tiling = {v: k for k, v in tc.TILING_NAMES.items()}[til_var.get()]
            load_texture(t, refit=False)
    til_cb.bind('<<ComboboxSelected>>', on_tiling)

    # ---------- grid view: thumbnails are decoded on a worker thread, newest requests (what you see) first
    CELL, THUMB = 156, 120
    thumbs, thumb_req = {}, set()
    thumb_in, thumb_out = queue.LifoQueue(), queue.Queue()
    gstate = dict(order=[], pos={}, cols=1, pad=0, sel=None, width=0)

    def thumb_worker():
        while True:
            p, t, gen = thumb_in.get()
            if gen != st['gen']:
                continue
            try:
                img, _ = p.image(t)
                if img is not None:
                    img = img.copy()
                    img.thumbnail((THUMB, THUMB))
                    bg = Image.new('RGB', img.size, '#2c2f36')
                    bg.paste(img, (0, 0), img)
                    img = bg
            except Exception:
                img = None
            thumb_out.put((gen, t, img))
    threading.Thread(target=thumb_worker, daemon=True).start()

    def cell_xy(k):
        return gstate['pad'] + (k % gstate['cols']) * CELL, (k // gstate['cols']) * CELL

    def place_thumb(i):
        k = gstate['pos'].get(i)
        if k is None:
            return
        x, y = cell_xy(k)
        grid_cv.delete(f'i{i}')
        if thumbs.get(i):
            grid_cv.create_image(x + CELL // 2, y + 6 + (CELL - 30) // 2, image=thumbs[i], tags=(f'i{i}',))
        else:
            grid_cv.create_text(x + CELL // 2, y + 66, text='no preview', fill=C['dim'], font=('Segoe UI', 9), tags=(f'i{i}',))

    def grid_layout():
        grid_cv.delete('all')
        p = st['pack']
        order = [int(i) for i in tree.get_children()] if p else []
        w = max(CELL, grid_cv.winfo_width())
        cols = max(1, w // CELL)
        gstate.update(order=order, pos={i: k for k, i in enumerate(order)}, cols=cols, pad=(w - cols * CELL) // 2, width=w)
        for k, i in enumerate(order):
            t = p.textures[i]
            x, y = cell_xy(k)
            grid_cv.create_rectangle(x + 5, y + 5, x + CELL - 5, y + CELL - 5, fill=C['panel'], width=2,
                                     outline=C['accent'] if i == gstate['sel'] else C['panel'], tags=(f'c{i}',))
            grid_cv.create_text(x + CELL // 2, y + CELL - 17, text=f'{t.name}  ·  {size_text(t)}', fill=C['dim'],
                                font=('Segoe UI', 8), tags=(f't{i}',))
            if t.status in ('missing', 'unsupported'):
                grid_cv.create_text(x + CELL // 2, y + 66, text='no data in file', fill=C['dim'], font=('Segoe UI', 9))
            elif i in thumbs:
                place_thumb(i)
        rows = -(-len(order) // cols)
        grid_cv.configure(scrollregion=(0, 0, w, max(rows * CELL + 10, grid_cv.winfo_height())))
        if not order:
            grid_cv.create_text(w // 2, 80, text='No textures to show' if p else 'Open a pack first', fill=C['dim'],
                                font=('Segoe UI', 12))
        request_visible()

    def request_visible():
        p = st['pack']
        if not p or st['view'] != 'grid' or not gstate['order']:
            return
        top, h, cols = grid_cv.canvasy(0), grid_cv.winfo_height(), gstate['cols']
        k0, k1 = int(top // CELL) * cols, min(len(gstate['order']), (int((top + h) // CELL) + 2) * cols)
        for k in reversed(range(k0, k1)):  # LIFO queue: push bottom first so the top row decodes first
            i = gstate['order'][k]
            t = p.textures[i]
            if i not in thumbs and i not in thumb_req and t.status not in ('missing', 'unsupported'):
                thumb_req.add(i)
                thumb_in.put((p, t, st['gen']))

    def pump_thumbs():
        try:
            for _ in range(30):
                gen, t, img = thumb_out.get_nowait()
                if gen != st['gen']:
                    continue
                thumbs[t.n] = ImageTk.PhotoImage(img) if img is not None else None
                if t.guessed or t.status == 'fixed':
                    update_row(t)
                    grid_cv.itemconfigure(f't{t.n}', text=f'{t.name}  ·  {size_text(t)}')
                if st['view'] == 'grid':
                    place_thumb(t.n)
        except queue.Empty:
            pass
        root.after(50, pump_thumbs)
    pump_thumbs()

    def grid_hit(e):
        x, y = grid_cv.canvasx(e.x) - gstate['pad'], grid_cv.canvasy(e.y)
        col, k = int(x // CELL), int(y // CELL) * gstate['cols'] + int(x // CELL)
        return gstate['order'][k] if 0 <= col < gstate['cols'] and 0 <= k < len(gstate['order']) else None

    def grid_select(i):
        if gstate['sel'] is not None:
            grid_cv.itemconfigure(f'c{gstate["sel"]}', outline=C['panel'])
        gstate['sel'] = i
        grid_cv.itemconfigure(f'c{i}', outline=C['accent'])

    def grid_click(e):
        i = grid_hit(e)
        if i is not None:
            grid_select(i)
            tree.selection_set(str(i))
            tree.see(str(i))

    def grid_open(e):
        if grid_hit(e) is not None:
            set_view('viewer')

    grid_cv.configure(yscrollcommand=lambda a, b: (g_vs.set(a, b), root.after_idle(request_visible)))
    grid_cv.bind('<MouseWheel>', lambda e: grid_cv.yview_scroll(int(-e.delta / 120 * 4), 'units'))
    grid_cv.bind('<Configure>', lambda e: grid_layout() if abs(e.width - gstate['width']) >= CELL // 2 else request_visible())
    grid_cv.bind('<Button-1>', grid_click)
    grid_cv.bind('<Double-Button-1>', grid_open)

    def set_view(v):
        if st['mode'] == 'arc':
            return
        st['view'] = settings['view'] = v
        for k, b in view_btns.items():
            b.configure(style='SegOn.TButton' if k == v else 'Seg.TButton')
        for b in chan_btns.values():
            b.state(['disabled'] if v == 'grid' else ['!disabled'])
        if v == 'grid':
            canvas.pack_forget()
            pix_lbl.pack_forget()
            gf.pack(fill='both', expand=True)
            gstate['sel'] = int(tree.selection()[0]) if tree.selection() else None
            def show_grid():
                grid_layout()
                k = gstate['pos'].get(gstate['sel'])
                rows = -(-len(gstate['order']) // gstate['cols'])
                grid_cv.yview_moveto(max(0, (k // gstate['cols']) - 1) / rows if k is not None and rows else 0)
                request_visible()
            root.after(30, show_grid)
        else:
            gf.pack_forget()
            canvas.pack(fill='both', expand=True)
            pix_lbl.pack(fill='x')
            root.after(30, fit if st['img'] is not None else redraw)

    # ---------- folder of packs
    def open_folder(d=None, then=None):
        d = d or filedialog.askdirectory(title='Open a folder with .ttpf / .rpf files', initialdir=st['folder'] or settings.get('folder'))
        if not d:
            return
        d = os.path.normpath(d)
        rows = scan_folder(d)
        st['folder'] = settings['folder'] = d
        save_settings(settings)
        packs.delete(*packs.get_children())
        label = {'indexed': '● Indexed', 'headerless': '◐ No header', 'rpf': '■ RPF', 'rpf_noheader': '□ RPF, no header'}
        for p_, kind, size in rows:
            packs.insert('', 'end', iid=p_, values=(label[kind], os.path.relpath(p_, d), human(size)), tags=(kind,))
        folder_lbl.configure(text=f'{os.path.basename(d)}  ·  {len(rows)} files')
        if not pk_frame.winfo_ismapped():
            pk_frame.pack(fill='x', before=tex_lbl)
        n_idx = sum(r_[1] == 'indexed' for r_ in rows)
        status.configure(text=f'{len(rows)} packs found: {n_idx} indexed, '
                              f'{sum(r_[1] == "headerless" for r_ in rows)} headerless, {sum(r_[1] in ("rpf", "rpf_noheader") for r_ in rows)} RPF.')
        target = then if then and packs.exists(then) else next((r_[0] for r_ in rows if r_[1] == 'indexed'), None)
        if target:
            packs.selection_set(target)
            packs.see(target)

    def on_pack_select(_=None):
        sel = packs.selection()
        if sel and os.path.normcase(st['path'] or '') != os.path.normcase(sel[0]):
            open_file(sel[0])
    packs.bind('<<TreeviewSelect>>', on_pack_select)

    # ---------- open
    def set_mode(m):
        st['mode'] = m
        if m == 'arc':
            right.grid_remove()
            right_arc.grid(row=0, column=2, sticky='nse', padx=(10, 0))
            for w_ in (vbar, canvas, pix_lbl, gf):
                w_.pack_forget()
            txt_frame.pack(fill='both', expand=True)
            tex_lbl.configure(text='FILES')
            deep_cb.pack(anchor='w', after=fl, pady=(0, 8))
            tree.heading('fmt', text='Type')
            save_btn.state(['disabled'])
        else:
            right_arc.grid_remove()
            right.grid()
            deep_cb.pack_forget()
            deep_var.set(False)
            hide_video_panel()
            txt_frame.pack_forget()
            vbar.pack(fill='x')
            tex_lbl.configure(text='TEXTURES')
            tree.heading('fmt', text='Format')
            save_btn.state(['!disabled'])
            set_view(st['view'])

    def close_current():
        hide_video_panel()
        for k_ in ('pack', 'arc'):
            if st[k_]:
                st[k_].close()
                st[k_] = None

    def remember(fn):
        st['path'] = fn
        settings['file'] = fn
        settings['recent'] = [fn] + [f for f in settings.get('recent', []) if f != fn][:9]
        save_settings(settings)
        root.title(f'{os.path.basename(fn)} · TTPF Studio')
        file_lbl.configure(text=os.path.basename(fn))

    def select_row(select):
        iid = str(select)
        if select is not None and tree.exists(iid):
            tree.selection_set(iid)
            tree.focus(iid)
            tree.see(iid)
            return True
        return False

    def open_archive(fn, items=None, select=None):
        def done_arc(a):
            close_current()
            st.update(arc=a, tex=None, img=None, item=None)
            st['gen'] += 1
            remember(fn)
            set_mode('arc')
            chip.configure(text='  RPF9 · CARVED  ', bg=C['accent'])
            edits_lbl.configure(text='')
            show_cb.configure(values=('All',) + tuple(ITEM_KINDS.values()))
            show_var.set('All')
            refill()
            txt.configure(state='normal')
            txt.delete('1.0', 'end')
            if a.items:
                kinds = __import__('collections').Counter(ITEM_KINDS[i.kind] for i in a.items)
                status.configure(text=f'{len(a.items)} files recovered from {os.path.basename(fn)}: '
                                      + ', '.join(f'{v} {k}' for k, v in kinds.most_common()))
                kids = tree.get_children()
                if not select_row(select):
                    tree.selection_set(kids[0])
                    tree.see(kids[0])
            else:
                txt.insert('1.0', 'Nothing recoverable in this archive.\n\nIts data is either all zeros '
                                  'or stored in a form this tool cannot read without the encryption keys.\n\n', 'com')
                txt.insert('end', a.header)
                status.configure(text=f'{os.path.basename(fn)}: nothing recoverable')
            txt.configure(state='disabled')
        if items is not None:  # already scanned by the library index: open instantly
            return run_bg(f'Opening {os.path.basename(fn)}…', lambda pr: Archive.from_items(fn, items), done_arc)
        run_bg(f'Scanning {os.path.basename(fn)}…  (big archives take a few minutes)', lambda pr: Archive(fn, pr), done_arc)

    def open_file(fn=None, select=None, items=None):
        fn = fn or filedialog.askopenfilename(title='Open texture pack or archive',
                                              filetypes=[('Packs, archives, videos', '*.ttpf *.rpf *.bk2 *.bik'), ('All files', '*.*')])
        if not fn:
            return
        if st['pack'] and st['pack'].edits and not messagebox.askyesno('Unsaved changes', 'Discard unsaved changes?'):
            return
        if fn.lower().endswith(('.bk2', '.bik')):
            choice = messagebox.askyesnocancel('Bink video', f'{os.path.basename(fn)}\n\nConvert it to MP4?\n\n'
                                               'Yes = convert to MP4    No = just play it')
            if choice is None:
                return
            return convert_video(fn, os.path.basename(fn)) if choice else play_video(fn)
        if fn.lower().endswith('.rpf'):
            return open_archive(fn, items, select)

        def done(p):
            close_current()
            set_mode('tex')
            st.update(pack=p, tex=None, img=None)
            st['gen'] += 1
            views = ('One per texture', 'All blobs') if p.carved else ('Has data', 'Full res only', 'All')
            show_cb.configure(values=views)
            if show_var.get() not in views:
                show_var.set(views[0])
            if p.carved and p.grouped != (show_var.get() == 'One per texture'):
                p.set_grouping(not p.grouped)
            thumbs.clear()
            thumb_req.clear()
            gstate['sel'] = None
            remember(fn)
            chip.configure(text='  HEADERLESS · CARVED  ' if p.carved else '  INDEXED  ',
                           bg=C['warn'] if p.carved else C['ok'])
            edits_lbl.configure(text='')
            refill()
            redraw()
            full = sum(t.status == 'full' for t in p.textures)
            status.configure(text=(f'{len(p.textures)} textures ({len(p.entries)} blobs) recovered from a file with no header. Formats are guessed per texture.'
                                   if p.carved else f'{len(p.textures)} textures · {full} with full-resolution data in this file.'))
            kids = tree.get_children()
            if kids and not select_row(select):
                tree.selection_set(kids[0])
                tree.see(kids[0])
        run_bg(f'Reading {os.path.basename(fn)}…', lambda pr: Pack(fn, pr), done)

    def info_win(title, text):
        w = tk.Toplevel(root, bg=C['panel'])
        w.title(title)
        w.geometry('620x360')
        tk.Label(w, text=title, bg=C['panel'], fg=C['text'], font=('Segoe UI Semibold', 13)).pack(anchor='w', padx=18, pady=(16, 6))
        tx = tk.Text(w, bg=C['panel2'], fg=C['text'], font=('Consolas', 10), relief='flat', padx=12, pady=10, wrap='word')
        tx.insert('1.0', text)
        tx.configure(state='disabled')
        tx.pack(fill='both', expand=True, padx=18, pady=(0, 18))

    # ---------- export / import
    def cur():
        t = st['tex']
        if not t or st['img'] is None:
            messagebox.showinfo('Nothing selected', 'Select a texture with data first.')
            return None
        return t

    def export_png():
        t = cur()
        if not t:
            return
        fn = filedialog.asksaveasfilename(defaultextension='.png', initialfile=f'{t.name}.png', filetypes=[('PNG', '*.png')])
        if fn:
            st['img'].save(fn)
            status.configure(text=f'Saved {fn}')

    def export_dds():
        t = cur()
        if not t:
            return
        fn = filedialog.asksaveasfilename(defaultextension='.dds', initialfile=f'{t.name}.dds', filetypes=[('DDS', '*.dds')])
        if fn:
            lin, w, h = st['pack'].linear(t)
            open(fn, 'wb').write(tc.dds_bytes(t.fmt, w, h, lin, t.dxgi if t.dxgi in tc.DXGI else None))
            status.configure(text=f'Saved {fn}  ({t.fmt}, {w}×{h})')

    def export_all():
        p = st['pack']
        if not p:
            return
        picked = [p.textures[i] for i in selected()] if len(tree.selection()) > 1 else p.textures
        todo = [t for t in picked if t.status not in ('missing', 'unsupported')]
        d = filedialog.askdirectory(title=f'Export {len(todo)} textures to…')
        if not d:
            return

        def work(pr):
            ok = bad = 0
            for i, t in enumerate(todo):
                pr(i / max(1, len(todo)))
                try:
                    img, _ = p.image(t)
                    if img is not None:
                        img.save(os.path.join(d, f'{t.name}_{t.fmt}_{img.width}x{img.height}.png'))
                        ok += 1
                except Exception:
                    bad += 1
            return ok, bad

        def done(res):
            refill()
            status.configure(text=f'Exported {res[0]} textures to {d}' + (f' ({res[1]} failed)' if res[1] else ''))
        run_bg(f'Exporting {len(todo)} textures…', work, done)

    def save_items(items, d, pr=lambda f: None):
        a = st['arc']
        for i, it in enumerate(items):
            pr(i / max(1, len(items)))
            sub = os.path.join(d, ITEM_KINDS[it.kind])
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, it.name), 'wb') as f:
                f.write(a.data(it))
        return len(items)

    def arc_save():
        it = st['item']
        if not it:
            return
        if len(tree.selection()) > 1:
            items = [st['arc'].items[i] for i in selected()]
            d = filedialog.askdirectory(title=f'Save {len(items)} files to…')
            if d:
                run_bg(f'Saving {len(items)} files…', lambda pr: save_items(items, d, pr),
                       lambda n: status.configure(text=f'Saved {n} files to {d}'))
            return
        fn = filedialog.asksaveasfilename(initialfile=it.name, defaultextension=os.path.splitext(it.name)[1])
        if fn:
            with open(fn, 'wb') as f:
                f.write(st['arc'].data(it))
            status.configure(text=f'Saved {fn}')

    def arc_save_many(same_type):
        a, it = st['arc'], st['item']
        if not a or (same_type and not it):
            return
        items = [i for i in a.items if i.kind == it.kind] if same_type else a.items
        d = filedialog.askdirectory(title=f'Save {len(items)} files to…')
        if d:
            run_bg(f'Saving {len(items)} files…', lambda pr: save_items(items, d, pr),
                   lambda n: status.configure(text=f'Saved {n} files to {d}'))

    def play_video(path):
        import subprocess
        exe = find_bink_player(settings.get('bink_player'))
        if not exe:
            if not messagebox.askyesno('Play video', 'Bink videos play in RAD Video Tools (free, from RAD Game Tools). '
                                       'VLC and ffmpeg cannot play Bink 2.\n\nIt was not found in the usual folders. '
                                       'If it is installed, locate its program now?'):
                return
            exe = filedialog.askopenfilename(title='Locate RAD Video Tools (radvideo64.exe or binkpl64.exe)',
                                             filetypes=[('Programs', '*.exe')])
            if not exe:
                return
        settings['bink_player'] = exe
        save_settings(settings)
        args = [exe, os.path.normpath(path)]
        try:
            frames_, vw_, vh_, _ = bink_info(path)
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            x_, y_, w_, h_ = fit_rect(vw_, vh_, int(sw * 0.8), int(sh * 0.8))
            args += [f'/W{w_}', f'/H{h_}', f'/X{(sw - w_) // 2}', f'/Y{(sh - h_) // 2}']  # never bigger than the screen
        except (OSError, ValueError):
            pass
        try:
            subprocess.Popen(args)
            status.configure(text=f'Playing {os.path.basename(path)} in {os.path.basename(exe)}')
        except OSError as ex_:
            settings.pop('bink_player', None)
            save_settings(settings)
            if getattr(ex_, 'winerror', None) == 740:  # needs admin rights: almost always the installer, not the player
                messagebox.showerror('Play video', f'{os.path.basename(exe)} looks like the RAD Video Tools installer.\n\n'
                                     'Run it yourself to install RAD Video Tools, then press Play video again '
                                     '(or pick radvideo64.exe from the folder it installed to).')
            else:
                messagebox.showerror('Play video', f'Could not start {exe}:\n{ex_}')

    # ---------- embedded video player
    st.update(player=None, vid=None, vid_info=None, seek_drag=None, fullscreen=False, pending_seek=None)

    def video_src(it):
        os.makedirs(VIDEO_CACHE, exist_ok=True)
        path = os.path.join(VIDEO_CACHE, it.name)
        if not os.path.exists(path) or os.path.getsize(path) != it.clen:
            with open(path, 'wb') as f:
                f.write(st['arc'].data(it))
        return path

    def stop_player():
        if st['player']:
            st['player'].stop()
            st['player'] = None
        vid_clip.place_forget()
        vb_play.configure(text='▶')

    def draw_seek(frac=None):
        seek_cv.delete('all')
        w = max(1, seek_cv.winfo_width())
        info = st['vid_info']
        if frac is None:
            frac = st['player'].position() / st['player'].duration if st['player'] else 0.0
        hover = st['seek_drag'] is not None
        y, th = 10, (5 if hover else 3)
        seek_cv.create_rectangle(0, y - th // 2, w, y + th // 2 + 1, fill='#3a3d45', outline='')
        seek_cv.create_rectangle(0, y - th // 2, int(w * frac), y + th // 2 + 1, fill=YT_RED, outline='')
        r = 7 if hover else 6
        x = int(w * frac)
        seek_cv.create_oval(x - r, y - r, x + r, y + r, fill=YT_RED, outline='')
        if info:
            dur = info[0] / info[3]
            vb_time.configure(text=f'{fmt_time(frac * dur)} / {fmt_time(dur)}')

    def place_player(_=None):
        pl, info = st['player'], st['vid_info']
        if pl and info:
            x, y, w, h = fit_rect(info[1], info[2], vid_host.winfo_width(), vid_host.winfo_height())
            vid_clip.place(x=x, y=y, width=w, height=h)
            # overscan: RAD draws a frame and a seek line at the picture's edges; push them just outside the clip
            my = 16
            mx = round(my * info[1] / info[2])
            pl.place(-mx, -my, w + 2 * mx, h + 2 * my)

    def show_video_panel(it):
        stop_player()
        st['vid'] = it
        try:
            st['vid_info'] = bink_info(video_src(it))
        except (OSError, ValueError):
            st['vid_info'] = None
        txt_frame.pack_forget()
        vid_frame.pack(fill='both', expand=True)
        vid_msg.place(relx=0.5, rely=0.5, anchor='center')
        vid_msg.configure(text=f'▶\n\n{it.info}\nClick or press Space to play')
        root.after(30, draw_seek)

    def hide_video_panel():
        stop_player()
        st['vid'] = None
        vid_frame.pack_forget()

    def start_player():
        it, info = st['vid'], st['vid_info']
        if not it or not info:
            return
        exe = find_bink_player(settings.get('bink_player'))
        if not exe:
            return play_video(video_src(it))  # asks where RAD Video Tools is, plays in its own window
        vid_host.update_idletasks()
        x, y, w, h = fit_rect(info[1], info[2], vid_host.winfo_width(), vid_host.winfo_height())
        pl = BinkEmbed(exe, video_src(it), info[0], info[3], info[1], info[2],
                       (vid_host.winfo_rootx() + x, vid_host.winfo_rooty() + y, w, h))
        st['player'] = pl
        vid_msg.configure(text='Loading…')

        def adopt(tries=0):
            if st['player'] is not pl:
                return
            if pl.attach(vid_clip.winfo_id()):
                vid_msg.place_forget()
                place_player()
                vb_play.configure(text='❚❚')
                if st['pending_seek'] is not None:
                    pl.seek(st['pending_seek'])
                    st['pending_seek'] = None
            elif tries < 60 and pl.alive():
                root.after(100, adopt, tries + 1)
            else:
                stop_player()
                vid_msg.configure(text='The player did not start.')
        adopt()

    def toggle_play(_=None):
        if st['mode'] != 'arc' or not st['vid']:
            return
        pl = st['player']
        if not pl:
            return start_player()
        pl.toggle_pause()
        vb_play.configure(text='▶' if pl.paused else '❚❚')

    def seek_frac(frac):
        if st['player']:
            st['player'].seek(frac)
        elif st['vid']:
            st['pending_seek'] = frac
            start_player()
        draw_seek(frac)

    def skip(sec):
        pl, info = st['player'], st['vid_info']
        if pl and info:
            seek_frac((pl.position() + sec) / pl.duration)

    def seek_event(e, release=False):
        frac = min(1.0, max(0.0, e.x / max(1, seek_cv.winfo_width())))
        st['seek_drag'] = None if release else frac
        if release:
            seek_frac(frac)
        else:
            draw_seek(frac)

    def player_tick():
        pl = st['player']
        if pl and st['seek_drag'] is None:
            if pl.hwnd and not pl.alive():  # reached the end (the player closes itself)
                st['player'] = None
                vid_clip.place_forget()
                vb_play.configure(text='↻')
                vid_msg.place(relx=0.5, rely=0.5, anchor='center')
                vid_msg.configure(text='↻\n\nReplay')
                draw_seek(1.0)
            else:
                draw_seek()
        root.after(250, player_tick)
    player_tick()

    def set_fullscreen(on=None):
        on = (not st['fullscreen']) if on is None else on
        if on == st['fullscreen'] or (on and st['mode'] != 'arc'):
            return
        st['fullscreen'] = on
        if on:
            for w_ in (head, sb):
                w_.pack_forget()
            left.grid_remove()
            right_arc.grid_remove()
        else:
            head.pack(fill='x', before=body)
            sb.pack(fill='x')
            left.grid()
            right_arc.grid()
        root.attributes('-fullscreen', on)
        root.after(150, place_player)

    def video_key(fn):
        def handler(e):
            fw = root.focus_get()
            if st['mode'] != 'arc' or not st['vid'] or (fw is not None and fw.winfo_class() in ('TEntry', 'Entry', 'Text')):
                return None
            fn()
            return 'break'
        return handler

    vid_host.bind('<Configure>', place_player)
    for w_ in (vid_host, vid_msg, vid_clip):
        w_.bind('<Button-1>', toggle_play)
        w_.bind('<Double-Button-1>', lambda e: set_fullscreen())
    vb_play.bind('<Button-1>', toggle_play)
    vb_back.bind('<Button-1>', lambda e: skip(-10))
    vb_fwd.bind('<Button-1>', lambda e: skip(10))
    vb_full.bind('<Button-1>', lambda e: set_fullscreen())
    vb_ext.bind('<Button-1>', lambda e: st['vid'] and (stop_player(), play_video(video_src(st['vid']))))
    vb_mp4.bind('<Button-1>', lambda e: arc_convert())
    seek_cv.bind('<Button-1>', seek_event)
    seek_cv.bind('<B1-Motion>', seek_event)
    seek_cv.bind('<ButtonRelease-1>', lambda e: seek_event(e, release=True))
    seek_cv.bind('<Configure>', lambda e: draw_seek())
    for key_, fn_ in (('<space>', toggle_play), ('k', toggle_play), ('j', lambda: skip(-10)), ('l', lambda: skip(10)),
                      ('f', set_fullscreen)):
        root.bind(key_, video_key(fn_ if key_ not in ('<space>', 'k') else (lambda: toggle_play())))
    root.bind('<Escape>', lambda e: set_fullscreen(False))

    def arc_play():
        it = st['item']
        if not it or it.kind != 'video':
            return
        if st['vid'] is not it:
            show_video_panel(it)
        toggle_play()

    def locate(key, finder, title, why, pattern):
        exe = finder()
        if exe:
            return exe
        if not messagebox.askyesno('Convert to MP4', why + '\n\nIf it is installed, locate it now?'):
            return None
        exe = filedialog.askopenfilename(title=title, filetypes=[(pattern, pattern), ('Programs', '*.exe')])
        if exe:
            settings[key] = exe
            save_settings(settings)
        return exe or None

    def video_tools():
        rad = locate('rad_converter', lambda: find_rad_converter(settings.get('rad_converter'), find_bink_player(settings.get('bink_player'))),
                     'Locate radvideo64.exe', 'Converting Bink 2 needs RAD Video Tools (free, from RAD Game Tools). '
                     'Its converter, radvideo64.exe, was not found.', 'radvideo64.exe')
        ff = rad and locate('ffmpeg', lambda: find_ffmpeg(settings.get('ffmpeg')), 'Locate ffmpeg.exe',
                            'Making the MP4 needs ffmpeg (free, from ffmpeg.org). It was not found.', 'ffmpeg.exe')
        return (rad, ff) if rad and ff else None

    def convert_video(src, default_name, cleanup=None):
        tools = video_tools()
        if not tools:
            return
        rad, ff = tools
        try:
            frames, w, h, fps = bink_info(src)
        except (OSError, ValueError) as ex_:
            return messagebox.showerror('Convert to MP4', str(ex_))
        start_dir = settings.get('mp4_dir') or os.path.join(os.path.expanduser('~'), 'Downloads')
        dst = filedialog.asksaveasfilename(title='Save MP4 as', initialdir=start_dir, defaultextension='.mp4',
                                           initialfile=os.path.splitext(default_name)[0] + '.mp4', filetypes=[('MP4 video', '*.mp4')])
        if not dst:
            return
        settings['mp4_dir'] = os.path.dirname(dst)
        save_settings(settings)

        def work(pr):
            try:
                return convert_bink_to_mp4(src, dst, rad, ff, lambda f, m: (pr(f), jobs.put(('msg', m))))
            finally:
                if cleanup:
                    cleanup()

        def done(path):
            status.configure(text=f'Saved {path}')
            if messagebox.askyesno('Convert to MP4', f'Done: {os.path.basename(path)}\n\nOpen the folder?'):
                os.startfile(os.path.dirname(path))
        run_bg(f'Converting {default_name}  ({frames} frames, {w}×{h}, {frames / fps:.0f} s)…', work, done)

    def convert_many(items):
        tools = video_tools()
        if not tools:
            return
        rad, ff = tools
        d = filedialog.askdirectory(title=f'Save {len(items)} MP4 videos to…',
                                    initialdir=settings.get('mp4_dir') or os.path.join(os.path.expanduser('~'), 'Downloads'))
        if not d:
            return
        settings['mp4_dir'] = d
        save_settings(settings)
        a = st['arc']

        def work(pr):
            ok, skipped, failed = 0, 0, []
            os.makedirs(VIDEO_CACHE, exist_ok=True)
            for k, it in enumerate(items):
                dst = os.path.join(d, os.path.splitext(it.name)[0] + '.mp4')
                if os.path.exists(dst):
                    skipped += 1  # resume-friendly: keep videos converted earlier
                    continue
                src = os.path.join(VIDEO_CACHE, 'src_' + it.name)
                try:
                    with open(src, 'wb') as f:
                        f.write(a.data(it))
                    convert_bink_to_mp4(src, dst, rad, ff, lambda f, m, k=k: (
                        pr((k + f) / len(items)), jobs.put(('msg', f'Video {k + 1}/{len(items)}: {m}'))))
                    ok += 1
                except Exception as ex_:
                    failed.append(f'{it.name}: {ex_}')
                    if os.path.exists(dst):
                        os.remove(dst)  # never leave a half-written MP4 behind
                finally:
                    if os.path.exists(src):
                        os.remove(src)
            return ok, skipped, failed

        def done(res):
            ok, skipped, failed = res
            msg = f'Converted {ok} videos' + (f', {skipped} already there' if skipped else '') + (f', {len(failed)} failed' if failed else '')
            status.configure(text=f'{msg}  ·  {d}')
            if failed:
                log_error('batch convert:\n' + '\n'.join(failed) + '\n')
            if messagebox.askyesno('Convert to MP4', msg + ('\n\nFirst problem: ' + failed[0] if failed else '') + '\n\nOpen the folder?'):
                os.startfile(d)
        run_bg(f'Converting {len(items)} videos…', work, done)

    def arc_convert():
        vids = [st['arc'].items[i] for i in selected() if st['arc'].items[i].kind == 'video']
        if len(vids) > 1:
            return convert_many(vids)
        it = st['item']
        if not it or it.kind != 'video':
            return
        os.makedirs(VIDEO_CACHE, exist_ok=True)
        src = os.path.join(VIDEO_CACHE, 'src_' + it.name)
        with open(src, 'wb') as f:
            f.write(st['arc'].data(it))
        convert_video(src, it.name, cleanup=lambda: os.path.exists(src) and os.remove(src))

    a_play.configure(command=arc_play)
    a_conv.configure(command=arc_convert)
    a_save.configure(command=arc_save)
    a_models.configure(command=lambda: open_models())  # defined further down
    a_type.configure(command=lambda: arc_save_many(True))
    a_all.configure(command=lambda: arc_save_many(False))
    a_head.configure(command=lambda: st['arc'] and info_win('Archive header · ' + os.path.basename(st['arc'].path), st['arc'].header))

    def open_models():
        """Every named 3D model (drawables / fragments) in the open archive: list, turntable preview, OBJ export."""
        import hashlib
        import models
        a = st['arc']
        if not a:
            return
        if st.get('modwin') and st['modwin'].winfo_exists():
            return st['modwin'].lift()
        w = tk.Toplevel(root, bg=C['bg'])
        st['modwin'] = w
        w.title(f'3D models · {os.path.basename(a.path)}')
        w.geometry('1280x800')
        w.minsize(900, 560)
        md = dict(all=[], shown=[], model=None, yaw=0.8, pitch=0.45, zoom=1.0, drag=None, photo=None, after=None,
                  streams=models.stream_offsets(a.items), cancel=threading.Event(),
                  paint=tuple(settings.get('paint') or models.PAINT))
        # textures live in .ttpf packs under the game folder (data/ttdb/...): find that folder above the archive
        game = os.path.dirname(os.path.abspath(a.path))
        for _ in range(4):
            if os.path.isdir(os.path.join(game, 'data', 'ttdb')) or os.path.dirname(game) == game:
                break
            game = os.path.dirname(game)
        if not os.path.isdir(os.path.join(game, 'data', 'ttdb')):
            game = st['folder'] or os.path.dirname(os.path.abspath(a.path))
        texidx = st.setdefault('texidx', {})
        if game not in texidx:
            texidx[game] = models.texture_index(game)
        md['tex'] = models.Textures(texidx[game])
        colours = tk.BooleanVar(value=True)
        key = hashlib.sha1(f'2|{os.path.abspath(a.path)}|{a.size}'.encode()).hexdigest()[:16]  # bump the 2 when find_models changes
        cache = os.path.join(APP_DIR, 'models', key + '.json')

        top = ttk.Frame(w, style='Head.TFrame', padding=(16, 12))
        top.pack(fill='x')
        ttk.Label(top, text='3D models', style='Head.TLabel').pack(side='left')
        msg = ttk.Label(top, text='', style='HeadDim.TLabel')
        msg.pack(side='left', padx=14)
        bar = ttk.Progressbar(top, length=240, mode='determinate')
        bar.pack(side='right')
        main = ttk.Frame(w, style='Bg.TFrame', padding=(12, 0, 12, 12))
        main.pack(fill='both', expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        lf = ttk.Frame(main, padding=10)
        lf.grid(row=0, column=0, sticky='nsw', padx=(0, 10))
        lf.rowconfigure(1, weight=1)
        q = tk.StringVar()
        ent = ttk.Entry(lf, textvariable=q, width=46)
        ent.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        lst = ttk.Treeview(lf, columns=('name',), show='headings', selectmode='extended')
        lst.heading('name', text='Name')
        lst.column('name', width=380)
        lst.grid(row=1, column=0, sticky='ns')
        sb_ = ttk.Scrollbar(lf, orient='vertical', command=lst.yview)
        sb_.grid(row=1, column=1, sticky='ns')
        lst.configure(yscrollcommand=sb_.set)
        btns = ttk.Frame(lf)
        btns.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        b_one = ttk.Button(btns, text='Export OBJ…', style='Accent.TButton')
        b_one.pack(side='left', fill='x', expand=True)
        b_many = ttk.Button(btns, text='Export listed to folder…')
        b_many.pack(side='left', fill='x', expand=True, padx=(6, 0))
        opts = ttk.Frame(lf)
        opts.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        ttk.Checkbutton(opts, text='Colours and textures', variable=colours,
                        command=lambda: draw()).pack(side='left')
        b_paint = ttk.Button(opts, text='Paint colour…')
        b_paint.pack(side='right')

        view = tk.Canvas(main, bg='#181a1f', highlightthickness=0, cursor='fleur')
        view.grid(row=0, column=1, sticky='nsew')
        info = ttk.Label(main, text='Drag to turn · wheel to zoom · double-click to reset', style='Dim.TLabel')
        info.grid(row=1, column=1, sticky='w', pady=(6, 0))

        def draw(fast=False):
            md['after'] = None
            view.delete('all')
            mdl = md['model']
            cw, ch = max(view.winfo_width(), 50), max(view.winfo_height(), 50)
            if not mdl:
                view.create_text(cw // 2, ch // 2, text='Pick a model on the left', fill=C['dim'], font=('Segoe UI', 12))
                return
            side = min(cw, ch)
            n = side // 2 if fast and len(mdl['tris']) > 20000 else side
            shown = mdl if colours.get() else dict(mdl, colours=None)
            img = models.render(shown, n, md['yaw'], md['pitch'], md['zoom'], bg=(24, 26, 31))
            if n != side:
                img = img.resize((side, side))
            md['photo'] = ImageTk.PhotoImage(img)
            view.create_image(cw // 2, ch // 2, image=md['photo'])

        def redraw(fast=False):
            if md['after']:
                w.after_cancel(md['after'])
            md['after'] = w.after(15 if fast else 60, lambda: draw(fast))

        def fill():
            ql = q.get().strip().lower()
            md['shown'] = [r for r in md['all'] if ql in r[1].lower()] if ql else list(md['all'])
            lst.delete(*lst.get_children())
            for k, (_, n) in enumerate(md['shown'][:20000]):
                lst.insert('', 'end', iid=str(k), values=(n,))
            msg.configure(text=f'{len(md["shown"]):,} of {len(md["all"]):,} models'
                          + ('  (first 20,000 listed, search to narrow)' if len(md['shown']) > 20000 else ''))

        def pick(_=None):
            sel = lst.selection()
            if not sel:
                return
            i, n = md['shown'][int(sel[0])]
            try:
                mdl = models.decode_model(a.m, md['streams'], i)
            except Exception:
                mdl = None
            md['model'] = mdl
            if mdl:
                import numpy as np
                found = models.colourise(mdl, md['tex'], md['paint'])
                ext = np.ptp(np.asarray(mdl['verts']), 0)
                info.configure(text=f'{n}  ·  {len(mdl["verts"]):,} vertices  ·  {len(mdl["tris"]):,} triangles  ·  '
                                    f'{ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} m  ·  '
                                    f'{found} of {len(mdl["materials"])} materials textured  ·  drag to turn, wheel to zoom')
            else:
                info.configure(text=f'{n}: no mesh in this one (collision bounds, or a layout not decoded yet)')
            draw()

        def safe(n):
            return ''.join(c if c.isalnum() or c in '_-' else '_' for c in n.replace('.#', '_'))

        def export_one():
            if not md['model']:
                return
            name = safe(md['model']['name'] or 'model')
            fn = filedialog.asksaveasfilename(parent=w, defaultextension='.obj', initialfile=name + '.obj',
                                              filetypes=[('Wavefront OBJ', '*.obj')])
            if fn:
                models.write_obj(md['model'], fn, md['tex'], md['paint'])
                msg.configure(text=f'Saved {fn} with its materials and textures')

        def export_many():
            todo = list(md['shown'])
            if not todo:
                return
            d_ = filedialog.askdirectory(parent=w, title=f'Export {len(todo):,} models to…')
            if not d_:
                return

            def work(pr):
                ok, tex = 0, models.Textures(md['tex'].index)  # own copy: the preview keeps using md['tex']
                for k, (i, n) in enumerate(todo):
                    if k % 20 == 0:
                        pr(k / len(todo))
                    try:
                        mdl = models.decode_model(a.m, md['streams'], i)
                    except Exception:
                        continue
                    if mdl:
                        models.write_obj(mdl, os.path.join(d_, f'{safe(n)}_{i}.obj'), tex, md['paint'])
                        ok += 1
                tex.close()
                return ok
            run_bg(f'Exporting {len(todo):,} models…', work, lambda ok: status.configure(text=f'Exported {ok:,} models to {d_}'))

        def press(e):
            md['drag'] = (e.x, e.y, md['yaw'], md['pitch'])

        def move(e):
            if md['drag'] and md['model']:
                x0, y0, yw, pt = md['drag']
                md['yaw'] = yw + (e.x - x0) * 0.01
                md['pitch'] = max(-1.5, min(1.5, pt + (e.y - y0) * 0.01))
                redraw(fast=True)

        def wheel(e):
            md['zoom'] = max(0.2, min(12.0, md['zoom'] * (1.15 if e.delta > 0 else 1 / 1.15)))
            redraw(fast=True)

        def reset(_=None):
            md.update(yaw=0.8, pitch=0.45, zoom=1.0)
            draw()

        view.bind('<ButtonPress-1>', press)
        view.bind('<B1-Motion>', move)
        view.bind('<ButtonRelease-1>', lambda e: (md.update(drag=None), redraw()))
        view.bind('<MouseWheel>', wheel)
        view.bind('<Double-Button-1>', reset)
        view.bind('<Configure>', lambda e: redraw())
        lst.bind('<<TreeviewSelect>>', pick)
        q.trace_add('write', lambda *_: fill())
        b_one.configure(command=export_one)
        b_many.configure(command=export_many)

        def pick_paint():
            from tkinter import colorchooser
            c = colorchooser.askcolor('#%02x%02x%02x' % md['paint'], parent=w, title='Vehicle paint colour')[0]
            if c:
                md['paint'] = tuple(int(x) for x in c)
                settings['paint'] = list(md['paint'])
                save_settings(settings)
                if md['model']:
                    models.colourise(md['model'], md['tex'], md['paint'])
                    draw()
        b_paint.configure(command=pick_paint)
        w.protocol('WM_DELETE_WINDOW', lambda: (md['cancel'].set(), md['tex'].close(), w.destroy()))

        def loaded(found):
            bar.pack_forget()
            md['all'] = found
            fill()

        try:
            with open(cache, encoding='utf-8') as f:
                return loaded([tuple(r) for r in json.load(f)])
        except (OSError, ValueError):
            pass
        msg.configure(text='Looking for models (once per archive, then remembered)…')
        out_q = queue.Queue()

        def scan():
            found = models.find_models(a.m, md['streams'], lambda f: out_q.put(f), md['cancel'])
            if not md['cancel'].is_set():
                try:
                    os.makedirs(os.path.dirname(cache), exist_ok=True)
                    with open(cache, 'w', encoding='utf-8') as f:
                        json.dump(found, f)
                except OSError:
                    pass
            out_q.put(found)

        def poll():
            if not w.winfo_exists():
                return
            try:
                while True:
                    r_ = out_q.get_nowait()
                    if isinstance(r_, list):
                        return loaded(r_)
                    bar['value'] = r_ * 100
            except queue.Empty:
                w.after(100, poll)
        threading.Thread(target=scan, daemon=True).start()
        poll()

    def open_library():
        """Browse everything in the folder by category, search it, preview it, open or export it."""
        import library
        folder = st['folder'] or settings.get('folder')
        if not folder or not os.path.isdir(folder):
            folder = filedialog.askdirectory(title='Library: choose the game folder')
            if not folder:
                return
            open_folder(folder)
        if st.get('libwin') and st['libwin'].winfo_exists():
            return st['libwin'].lift()
        db = library.db_path(folder)
        con = library.connect(db)
        w = tk.Toplevel(root, bg=C['bg'])
        st['libwin'] = w
        w.title(f'Library · {os.path.basename(folder)}')
        w.geometry('1360x820')
        w.minsize(1000, 600)
        lib = dict(rows=[], job=None, packs={}, mm={}, photo=None, gen=0)

        top = ttk.Frame(w, style='Head.TFrame', padding=(16, 12))
        top.pack(fill='x')
        ttk.Label(top, text='Library', style='Head.TLabel').pack(side='left')
        ttk.Label(top, text=folder, style='HeadDim.TLabel').pack(side='left', padx=14)
        idx_btn = ttk.Button(top, text='Update index', style='Accent.TButton')
        idx_btn.pack(side='right')
        stop_btn = ttk.Button(top, text='Stop')
        stop_btn.pack(side='right', padx=6)
        stop_btn.state(['disabled'])
        lbar = ttk.Progressbar(top, length=260, mode='determinate')
        lbar.pack(side='right', padx=10)
        lmsg = ttk.Label(top, text='', style='HeadDim.TLabel')
        lmsg.pack(side='right')

        main = ttk.Frame(w, style='Bg.TFrame', padding=(12, 0, 12, 12))
        main.pack(fill='both', expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # categories
        cf = ttk.Frame(main, padding=10)
        cf.grid(row=0, column=0, sticky='nsw', padx=(0, 10))
        ttk.Label(cf, text='CATEGORIES', style='Sec.TLabel').pack(anchor='w', pady=(0, 6))
        cats = ttk.Treeview(cf, show='tree', selectmode='browse', height=30)
        cats.column('#0', width=300)
        cvs = ttk.Scrollbar(cf, command=cats.yview)
        cats.configure(yscrollcommand=cvs.set)
        cats.pack(side='left', fill='both', expand=True)
        cvs.pack(side='right', fill='y')

        # results
        rf = ttk.Frame(main, padding=10)
        rf.grid(row=0, column=1, sticky='nsew')
        sr = ttk.Frame(rf)
        sr.pack(fill='x', pady=(0, 8))
        lq = tk.StringVar()
        lq_entry = ttk.Entry(sr, textvariable=lq)
        lq_entry.pack(side='left', fill='x', expand=True)
        lcount = ttk.Label(sr, text='', style='Dim.TLabel')
        lcount.pack(side='left', padx=10)
        cols_ = ('name', 'cat', 'sub', 'size', 'pack')
        res = ttk.Treeview(rf, columns=cols_, show='headings', selectmode='extended')
        for c_, t_, wd, an in (('name', 'Name', 260, 'w'), ('cat', 'Category', 170, 'w'), ('sub', 'Detail', 220, 'w'),
                               ('size', 'Size', 90, 'e'), ('pack', 'Pack', 220, 'w')):
            res.heading(c_, text=t_)
            res.column(c_, width=wd, anchor=an, stretch=c_ in ('name', 'sub', 'pack'))
        rvs = ttk.Scrollbar(rf, command=res.yview)
        res.configure(yscrollcommand=rvs.set)
        res.pack(side='left', fill='both', expand=True)
        rvs.pack(side='right', fill='y')

        # preview
        pf = ttk.Frame(main, padding=14, width=330)
        pf.grid(row=0, column=2, sticky='nse', padx=(10, 0))
        pf.grid_propagate(False)
        ttk.Label(pf, text='PREVIEW', style='Sec.TLabel').pack(anchor='w')
        pimg = tk.Label(pf, bg=C['panel'], height=18)
        pimg.pack(fill='x', pady=(8, 8))
        pinfo = ttk.Label(pf, text='Select something', style='Dim.TLabel', wraplength=300, justify='left')
        pinfo.pack(anchor='w')
        ptxt = tk.Text(pf, height=12, bg=C['panel2'], fg=C['text'], font=('Consolas', 9), relief='flat', wrap='none')
        ptxt.pack(fill='both', expand=True, pady=(8, 8))
        open_b = ttk.Button(pf, text='Open in main window', style='Accent.TButton')
        open_b.pack(fill='x')
        exp_b = ttk.Button(pf, text='Export selected…')
        exp_b.pack(fill='x', pady=(6, 0))

        def fill_categories():
            cats.delete(*cats.get_children())
            total = con.execute('SELECT COUNT(*) FROM items').fetchone()[0]
            cats.insert('', 'end', iid='*', text=f'Everything   {total:,}', open=True)
            for cat, n, subs in library.categories(con):
                node = cats.insert('', 'end', iid=f'c|{cat}', text=f'{cat}   {n:,}')
                for sub, k in subs[:200]:
                    cats.insert(node, 'end', iid=f's|{cat}|{sub}', text=f'{sub}   {k:,}')
            if not total:
                lcount.configure(text='Nothing indexed yet: press "Update index" (the first run takes a while).')

        def run_query(*_):
            sel = cats.selection()
            key = sel[0] if sel else '*'
            cat = sub = None
            if key.startswith('c|'):
                cat = key[2:]
            elif key.startswith('s|'):
                cat, sub = key[2:].split('|', 1)
            lib['rows'] = library.query(con, cat, sub, lq.get().strip() or None)
            res.delete(*res.get_children())
            for i, r_ in enumerate(lib['rows']):
                _, pack, kind, cat_, sub_, name, size, w_, h_, fmt, ref, parts, info = r_
                res.insert('', 'end', iid=str(i), values=(name, cat_, f'{sub_}{"  " + fmt if fmt else ""}', human(size or 0),
                                                          os.path.basename(pack)))
            lcount.configure(text=f'{len(lib["rows"]):,} shown' + ('  (first 5,000)' if len(lib['rows']) >= 5000 else ''))

        def debounced(*_):
            if lib.get('after'):
                w.after_cancel(lib['after'])
            lib['after'] = w.after(300, run_query)
        lq.trace_add('write', debounced)
        cats.bind('<<TreeviewSelect>>', run_query)

        # ---- preview (decoding runs on a thread; results come back through a queue)
        pq = queue.Queue()

        def item_bytes(r_):
            _, pack, kind, *_rest = r_
            parts, ref = r_[11], r_[10]
            if parts:
                if pack not in lib['mm']:
                    f_ = open(pack, 'rb')
                    lib['mm'][pack] = (f_, mmap.mmap(f_.fileno(), 0, access=mmap.ACCESS_READ))
                m_ = lib['mm'][pack][1]
                ps = json.loads(parts)
                if kind == 'video':
                    return m_[ps[0][0]:ps[0][0] + 4096]
                return b''.join((inflate_at(m_, off) or (b'',))[0] for off, _ in ps)
            return None

        def texture_image(r_):
            pack, ref = r_[1], r_[10]
            if pack not in lib['packs']:
                if len(lib['packs']) >= 3:
                    old = next(iter(lib['packs']))
                    lib['packs'].pop(old).close()
                lib['packs'][pack] = Pack(pack)
            pk = lib['packs'][pack]
            return pk.image(pk.textures[ref])[0]

        def preview_worker(r_, gen):
            try:
                if r_[2] == 'texture':
                    img = texture_image(r_)
                    if img is not None:
                        img = img.copy()
                        img.thumbnail((300, 300))
                    pq.put((gen, 'img', img))
                else:
                    pq.put((gen, 'data', item_bytes(r_)))
            except Exception as ex_:
                pq.put((gen, 'err', str(ex_)))

        def pump_preview():
            try:
                while True:
                    gen, kind, val = pq.get_nowait()
                    if gen != lib['gen']:
                        continue
                    ptxt.delete('1.0', 'end')
                    if kind == 'img':
                        if val is not None:
                            bg = checker_bg(*val.size).copy()
                            bg.paste(val, (0, 0), val)
                            lib['photo'] = ImageTk.PhotoImage(bg)
                            pimg.configure(image=lib['photo'], height=val.height)
                        else:
                            pimg.configure(image='')
                            ptxt.insert('1.0', 'No preview (no data in this file).')
                    elif kind == 'data':
                        pimg.configure(image='', height=1)
                        r_ = lib['rows'][int(res.focus() or 0)]
                        d_ = val or b''
                        if r_[2] in ('xml', 'text'):
                            ptxt.insert('1.0', d_[:6000].decode('utf-8', 'replace'))
                        elif r_[2] == 'skel':
                            import re
                            i_ = d_.find(b'SKEL_')
                            ptxt.insert('1.0', '\n'.join(dict.fromkeys(b.decode() for b in re.findall(rb'[A-Za-z][A-Za-z0-9_]{3,40}', d_[i_:])))[:6000])
                        else:
                            ptxt.insert('1.0', hexdump(d_, 1024))
                    else:
                        ptxt.insert('1.0', 'Preview failed: ' + str(val))
            except queue.Empty:
                pass
            if w.winfo_exists():
                w.after(80, pump_preview)
        pump_preview()

        def on_pick(_=None):
            foc = res.focus()
            if not foc:
                return
            r_ = lib['rows'][int(foc)]
            _, pack, kind, cat_, sub_, name, size, w_, h_, fmt, ref, parts, info = r_
            pinfo.configure(text=f'{name}\n{cat_}  ·  {sub_}  ·  {human(size or 0)}' + (f'\n{fmt}' if fmt else '')
                                 + (f'\n{info}' if info and kind != 'texture' else '') + f'\n{os.path.basename(pack)}')
            lib['gen'] += 1
            threading.Thread(target=preview_worker, args=(r_, lib['gen']), daemon=True).start()
        res.bind('<<TreeviewSelect>>', on_pick)

        def open_in_main(_=None):
            foc = res.focus()
            if not foc:
                return
            r_ = lib['rows'][int(foc)]
            pack, kind, ref = r_[1], r_[2], r_[10]
            items = library.archive_items(con, pack) if pack.lower().endswith('.rpf') else None
            root.lift()
            open_file(pack, select=ref, items=items)
        res.bind('<Double-Button-1>', open_in_main)
        open_b.configure(command=open_in_main)

        def export_selected():
            picked = [lib['rows'][int(i)] for i in res.selection()] or lib['rows']
            if not picked:
                return
            d_ = filedialog.askdirectory(title=f'Export {len(picked)} items to…', parent=w)
            if not d_:
                return

            def work(pr):
                done_n = 0
                for k, r_ in enumerate(picked):
                    pr(k / len(picked))
                    sub_dir = os.path.join(d_, r_[3])
                    os.makedirs(sub_dir, exist_ok=True)
                    try:
                        if r_[2] == 'texture':
                            img = texture_image(r_)
                            if img is not None:
                                img.save(os.path.join(sub_dir, f'{os.path.splitext(os.path.basename(r_[1]))[0]}_{r_[5]}.png'))
                                done_n += 1
                        else:
                            ps = json.loads(r_[11])
                            m_ = item_bytes(r_)  # also opens the archive's file mapping used for videos below
                            with open(os.path.join(sub_dir, r_[5]), 'wb') as f_:
                                if r_[2] == 'video':
                                    mm = lib['mm'][r_[1]][1]
                                    f_.write(mm[ps[0][0]:ps[0][0] + ps[0][1]])
                                else:
                                    f_.write(m_)
                            done_n += 1
                    except Exception as ex_:
                        log_error(f'library export {r_[5]}: {ex_}\n')
                return done_n
            run_bg(f'Exporting {len(picked)} items…', work, lambda n: status.configure(text=f'Exported {n} items to {d_}'))
        exp_b.configure(command=export_selected)

        # ---- indexing on a thread
        cancel = threading.Event()
        iq = queue.Queue()

        def poll_index():
            try:
                while True:
                    kind, a_, b_ = iq.get_nowait()
                    if kind == 'p':
                        lmsg.configure(text=a_)
                        lbar['value'] = b_ * 100
                        continue
                    idx_btn.state(['!disabled'])
                    stop_btn.state(['disabled'])
                    lmsg.configure(text=(f'{a_["items"]:,} items indexed' if kind == 'ok' else f'Index failed: {a_}'))
                    fill_categories()
                    run_query()
                    return
            except queue.Empty:
                pass
            if w.winfo_exists():
                w.after(150, poll_index)

        def start_index():
            cancel.clear()
            idx_btn.state(['disabled'])
            stop_btn.state(['!disabled'])

            def work():
                try:
                    iq.put(('ok', library.index_folder(folder, db, lambda m_, f_: iq.put(('p', m_, f_)), cancel), None))
                except Exception as ex_:
                    log_error(traceback.format_exc())
                    iq.put(('err', ex_, None))
            threading.Thread(target=work, daemon=True).start()
            poll_index()
        idx_btn.configure(command=start_index)
        stop_btn.configure(command=lambda: (cancel.set(), lmsg.configure(text='Stopping after the current pack…')))

        def on_close_lib():
            cancel.set()
            for pk in lib['packs'].values():
                pk.close()
            for f_, m_ in lib['mm'].values():
                m_.close()
                f_.close()
            w.destroy()
        w.protocol('WM_DELETE_WINDOW', on_close_lib)
        fill_categories()
        cats.selection_set('*')
        if not con.execute('SELECT COUNT(*) FROM packs').fetchone()[0]:
            start_index()  # first visit: build the index straight away

    def open_dumper():
        """Window that dumps every texture of every pack in a folder (see dump_folder)."""
        if st.get('dumper') and st['dumper'].winfo_exists():
            return st['dumper'].lift()
        src = st['folder'] or settings.get('folder')
        w = tk.Toplevel(root, bg=C['panel'])
        st['dumper'] = w
        w.title('Export everything')
        w.geometry('680x520')
        w.transient(root)
        fr = ttk.Frame(w, padding=18)
        fr.pack(fill='both', expand=True)
        fr.columnconfigure(1, weight=1)
        ttk.Label(fr, text='Export every texture in a folder of packs', font=('Segoe UI Semibold', 13)).grid(
            row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(fr, text='PNGs are sorted into <pack>/<Colour | Normal map | Packed data | Greyscale>, with an index.csv '
                  'catalogue of every texture (format, size, offset, hash). Duplicates and blank textures are skipped. '
                  'Stop any time: running again resumes.', style='Dim.TLabel', wraplength=610, justify='left').grid(
            row=1, column=0, columnspan=3, sticky='w', pady=(4, 14))
        src_v = tk.StringVar(value=src or '')
        out_v = tk.StringVar(value=settings.get('dump_out') or (src.rstrip('\\/') + '_export' if src else ''))
        for r_, (lab, var, title) in enumerate((('Packs folder', src_v, 'Folder with .ttpf files'),
                                                 ('Save to', out_v, 'Output folder')), start=2):
            ttk.Label(fr, text=lab, style='Dim.TLabel').grid(row=r_, column=0, sticky='w', pady=3)
            ttk.Entry(fr, textvariable=var).grid(row=r_, column=1, sticky='ew', padx=8, pady=3)
            ttk.Button(fr, text='Browse…', command=lambda v=var, t_=title: v.set(
                filedialog.askdirectory(title=t_, parent=w) or v.get())).grid(row=r_, column=2, pady=3)
        hl_v, bl_v, rp_v = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, text='Include packs without a header (format is guessed)', variable=hl_v).grid(
            row=4, column=0, columnspan=3, sticky='w', pady=(12, 2))
        ttk.Checkbutton(fr, text='Skip blank (single-colour) textures', variable=bl_v).grid(row=5, column=0, columnspan=3, sticky='w')
        ttk.Checkbutton(fr, text='Also carve RPF archives: XML data, videos, scripts (big archives add a lot of time)',
                        variable=rp_v).grid(row=6, column=0, columnspan=3, sticky='w')
        bar = ttk.Progressbar(fr, mode='determinate')
        bar.grid(row=7, column=0, columnspan=3, sticky='ew', pady=(18, 6))
        msg = ttk.Label(fr, text='Ready. A full dump of a whole game folder can take 30+ minutes and ~10 GB.', style='Dim.TLabel')
        msg.grid(row=8, column=0, columnspan=3, sticky='w')
        res = ttk.Label(fr, text='', style='Val.TLabel', justify='left')
        res.grid(row=9, column=0, columnspan=3, sticky='nw', pady=(10, 0))
        fr.rowconfigure(9, weight=1)
        btns = ttk.Frame(fr)
        btns.grid(row=10, column=0, columnspan=3, sticky='e', pady=(16, 0))
        start_b = ttk.Button(btns, text='Start', style='Accent.TButton')
        stop_b = ttk.Button(btns, text='Stop')
        open_b = ttk.Button(btns, text='Open output folder', command=lambda: os.path.isdir(out_v.get()) and os.startfile(out_v.get()))
        for b in (open_b, stop_b, start_b):
            b.pack(side='left', padx=(6, 0))
        stop_b.state(['disabled'])
        q = queue.Queue()
        cancel = threading.Event()

        def poll():
            try:
                while True:
                    kind, a, b = q.get_nowait()
                    if kind == 'p':
                        msg.configure(text=a)
                        bar['value'] = b * 100
                        continue
                    start_b.state(['!disabled'])
                    stop_b.state(['disabled'])
                    if kind == 'ok':
                        res.configure(text='\n'.join(f'{k:<30}{v}' for k, v in sorted(a.items())))
                        status.configure(text=f'Export finished: {a.get("exported", 0)} textures in {out_v.get()}')
                    else:
                        msg.configure(text=f'Failed: {a}')
                    return
            except queue.Empty:
                pass
            if w.winfo_exists():
                w.after(150, poll)

        def start():
            sv, ov = src_v.get().strip(), out_v.get().strip()
            if not os.path.isdir(sv) or not ov:
                return messagebox.showwarning('Export everything', 'Pick a packs folder and an output folder.', parent=w)
            if os.path.abspath(ov).startswith(os.path.abspath(sv) + os.sep):
                return messagebox.showwarning('Export everything', 'Save somewhere outside the packs folder.', parent=w)
            settings['dump_out'] = ov
            save_settings(settings)
            cancel.clear()
            start_b.state(['disabled'])
            stop_b.state(['!disabled'])
            res.configure(text='')

            def work():
                try:
                    r_ = dump_folder(sv, ov, headerless=hl_v.get(), skip_blank=bl_v.get(), rpf=rp_v.get(),
                                     progress=lambda m, f: q.put(('p', m, f)), cancel=cancel)
                    q.put(('ok', r_, None))
                except Exception as ex_:
                    log_error(traceback.format_exc())
                    q.put(('err', ex_, None))
            threading.Thread(target=work, daemon=True).start()
            poll()

        def stop():
            cancel.set()
            msg.configure(text='Stopping after the current textures…')
        start_b.configure(command=start)
        stop_b.configure(command=stop)
        w.protocol('WM_DELETE_WINDOW', lambda: (cancel.set(), w.destroy()))

    def apply_levels(t, fmt, w, h, levels):
        n = st['pack'].replace(t, fmt, w, h, levels)
        load_texture(t, refit=False)
        status.configure(text=f'Replaced {t.name}: {n} mip level{"s" * (n != 1)} updated. Save the pack to write it.')

    def import_png():
        t = cur()
        if not t:
            return
        fn = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.tga *.bmp')])
        if not fn:
            return
        try:
            img = Image.open(fn)
            _, w, h, _ = t.best()
            if img.size != (w, h):
                raise ValueError(f'Image is {img.width}×{img.height}, texture needs {w}×{h}.')
            lin = tc.image_to_lin(t.fmt, img)
            levels = [lin]
            for k in range(1, 12):  # rebuild lower mips that have their own blob
                mw, mh = max(1, w >> k), max(1, h >> k)
                if mw < 4 or mh < 4:
                    break
                levels.append(tc.image_to_lin(t.fmt, img.resize((mw, mh), Image.LANCZOS)))
            apply_levels(t, t.fmt, w, h, levels)
        except Exception as ex_:
            messagebox.showerror('Import PNG', str(ex_))

    def import_dds():
        t = cur()
        if not t:
            return
        fn = filedialog.askopenfilename(filetypes=[('DDS', '*.dds')])
        if not fn:
            return
        try:
            fmt, w, h, levels = tc.dds_read(open(fn, 'rb').read())
            apply_levels(t, fmt, w, h, levels)
        except Exception as ex_:
            messagebox.showerror('Import DDS', str(ex_))

    def revert():
        t = st['tex']
        if t:
            st['pack'].revert(t)
            load_texture(t, refit=False)

    def save():
        p = st['pack']
        if not p or not p.edits:
            return messagebox.showinfo('Nothing to save', 'Replace a texture first.')
        base, ext = os.path.splitext(os.path.basename(p.path))
        fn = filedialog.asksaveasfilename(defaultextension=ext, initialfile=f'{base}_mod{ext}',
                                          initialdir=os.path.dirname(p.path))
        if not fn:
            return

        def done(_):
            status.configure(text=f'Saved {fn}')
            messagebox.showinfo('Saved', f'Modded pack written to\n{fn}\n\nThe original file was not changed.')
        run_bg('Saving (copies the whole pack first)…', lambda pr: p.save_as(fn, pr), done)

    for b, fn_ in ((open_btn, open_file), (folder_btn, open_folder), (mine_btn, open_dumper), (lib_btn, open_library), (save_btn, save), (btn_png, export_png), (btn_dds, export_dds),
                   (btn_all, export_all), (btn_ipng, import_png), (btn_idds, import_dds), (btn_rev, revert)):
        b.configure(command=fn_)
    root.bind('<Control-o>', lambda _: open_file())
    root.bind('<Control-O>', lambda _: open_folder())  # Ctrl+Shift+O
    root.bind('<Control-s>', lambda _: save())
    root.bind('<Control-e>', lambda _: export_png())
    def copy_image(_=None):
        if root.focus_get() is not None and root.focus_get().winfo_class() in ('Text', 'TEntry', 'Entry'):
            return  # let text fields copy their own selection
        if st['mode'] != 'tex' or st['img'] is None:
            return
        try:
            shown = disp() if st['chan'] == 'Split' else channel_view(st['img'])
            copy_image_to_clipboard(shown)
            status.configure(text=f'Copied {st["tex"].name} ({shown.width}×{shown.height}) to the clipboard')
        except Exception as ex_:
            messagebox.showerror('Copy', str(ex_))

    def cycle_bg():
        order = ('checker', 'black', 'white')
        settings['viewer_bg'] = order[(order.index(settings.get('viewer_bg', 'checker')) + 1) % 3]
        bg_btn.configure(text={'checker': 'BG: grid', 'black': 'BG: black', 'white': 'BG: white'}[settings['viewer_bg']])
        redraw()

    def focus_search(_=None):
        q_entry.focus_set()
        q_entry.select_range(0, 'end')
        return 'break'

    btn_copy = ttk.Button(ex, text='Copy', command=copy_image)
    btn_copy.pack(side='left', fill='x', expand=True, padx=(6, 0))
    bg_btn.configure(command=cycle_bg, text={'checker': 'BG: grid', 'black': 'BG: black', 'white': 'BG: white'}[settings.get('viewer_bg', 'checker')])
    help_btn.configure(command=lambda: info_win('Keyboard shortcuts', SHORTCUTS))
    root.bind('<F1>', lambda _: info_win('Keyboard shortcuts', SHORTCUTS))
    root.bind('<Control-c>', copy_image)
    root.bind('<Control-f>', focus_search)
    root.bind('<Control-g>', lambda _: set_view('viewer' if st['view'] == 'grid' else 'grid'))

    def on_close():
        if st['pack'] and st['pack'].edits and not messagebox.askyesno('Unsaved changes', 'Quit without saving?'):
            return
        settings['zoomed'] = root.state() == 'zoomed'
        if not settings['zoomed']:
            settings['geometry'] = root.geometry()
        save_settings(settings)
        stop_player()
        shutil.rmtree(VIDEO_CACHE, ignore_errors=True)  # temp copies made for playback
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', on_close)
    set_view(settings.get('view', 'viewer'))
    if path:
        if os.path.isdir(path):
            root.after(100, lambda: open_folder(path))
        else:
            root.after(100, lambda: open_file(path))
    elif settings.get('folder') and os.path.isdir(settings['folder']):
        root.after(100, lambda: open_folder(settings['folder'], then=settings.get('file')))
    redraw()
    root.mainloop()


# ====================================================================== self-test
def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix='ttpf_selftest_')
    try:
        _selftest(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)  # also when an assert fails halfway
    print('selftest OK')


def _selftest(d):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    # round trip of the swizzle for every element size and tile size
    for fmt, w, h in (('RGBA8', 64, 64), ('RGBA8', 256, 128), ('BC1', 512, 256), ('BC7', 256, 256), ('BC7', 64, 32)):
        for tiling in tc.TILINGS:
            bpe, ew, eh = tc.dims(fmt, w, h)
            lin = rng.integers(0, 256, (eh, ew, bpe), dtype=np.uint8)
            sw = tc.swizzle(fmt, lin, w, h, b'', tiling)
            assert (tc.unswizzle(fmt, sw, w, h, tiling) == lin).all(), f'swizzle round trip {fmt} {w}x{h} {tiling}'
    assert tc.tiling_from_desc(0x313 | 0x0800 << 10) == 'S' and tc.tiling_from_desc(0x313 | 0x8800 << 10) == 'R4'

    # synthetic pack: texture 0 = RGBA8 64x64 (one blob), texture 1 = BC1 256x256 with a tail + mip1 + mip0 chain
    img0 = Image.fromarray(rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), 'RGBA')
    tex0 = tc.swizzle('RGBA8', tc.image_to_lin('RGBA8', img0), 64, 64, b'')
    grad = Image.linear_gradient('L').resize((256, 256)).convert('RGBA')
    m0 = tc.swizzle('BC1', tc.image_to_lin('BC1', grad), 256, 256, b'')
    m1 = tc.swizzle('BC1', tc.image_to_lin('BC1', grad.resize((128, 128))), 128, 128, b'')
    tail = bytes(4096)
    blobs = [(0x10, tex0), (0x40, tail), (0x41, m1), (0x41 + len(m1) // PAGE, m0)]
    body = bytearray(0x18 + 8 * len(blobs))
    etpf_at = len(body)
    recs = [(63 | 63 << 14 | 1 << 28, 28 << 3, 0x10, 0), (255 | 255 << 14 | 3 << 28, 71 << 3 | 1, 0x40, 0)]
    body += b'ETPF' + struct.pack('<III', 0, len(recs), 16 * len(recs))
    for rec in recs:
        body += struct.pack('<IIII', *rec)
    raws = []
    for page, blob in blobs:
        body += bytes(-len(body) % ALIGN)
        off = len(body)
        body += zlib.compress(blob, 9)
        raws.append((0xABCD << 48) | (((page << 6) | 32) << 24) | (off // ALIGN))
    body[:0x18] = b'RDS5' + struct.pack('<IIIII', len(raws), 0, 0x16000000, 0, 0)
    for i, raw in enumerate(raws):
        struct.pack_into('<Q', body, 0x18 + 8 * i, raw)
    assert etpf_at > 0
    src = os.path.join(d, 'a.ttpf')
    open(src, 'wb').write(bytes(body) + bytes(ALIGN))

    p = Pack(src)
    t0, t1 = p.textures
    assert t0.status == 'full' and t1.status == 'full' and len(t1.chain) == 3, (t0.status, t1.status, len(t1.chain))
    assert np.array_equal(np.asarray(p.image(t0)[0]), np.asarray(img0)), 'RGBA8 decode'
    got = np.asarray(p.image(t1)[0].convert('L'), np.int16)
    assert np.abs(got - np.asarray(grad.convert('L'), np.int16)).max() < 12, 'BC1 decode'

    # DDS round trip
    lin, w, h = p.linear(t1)
    fmt, w2, h2, lv = tc.dds_read(tc.dds_bytes('BC1', w, h, lin))
    assert (fmt, w2, h2) == ('BC1', 256, 256) and lv[0] == lin.tobytes()

    # replace both textures (incompressible noise forces append + repoint), save, reload
    noise = Image.fromarray(rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), 'RGBA')
    assert p.replace(t0, 'RGBA8', 64, 64, [tc.image_to_lin('RGBA8', noise)]) == 1
    flipped = grad.transpose(Image.FLIP_LEFT_RIGHT)
    lv = [tc.image_to_lin('BC1', flipped), tc.image_to_lin('BC1', flipped.resize((128, 128)))]
    assert p.replace(t1, 'BC1', 256, 256, lv) == 2, 'mip0 + mip1 written, tail left alone'
    out = os.path.join(d, 'b.ttpf')
    p.save_as(out)
    p.close()
    q = Pack(out)
    assert np.array_equal(np.asarray(q.image(q.textures[0])[0]), np.asarray(noise)), 'edited RGBA8 after save'
    got = np.asarray(q.image(q.textures[1])[0].convert('L'), np.int16)
    assert np.abs(got - np.asarray(flipped.convert('L'), np.int16)).max() < 12, 'edited BC1 after save'
    assert all(e.raw >> 48 == 0xABCD for e in q.entries), 'unknown entry bits preserved'
    q.close()

    # bulk dump + resume
    dump_dir = os.path.join(d, 'dump')
    stats = dump_folder([out], dump_dir, workers=2)
    assert stats['exported'] == 2 and stats['packs done'] == 1, stats
    with open(os.path.join(dump_dir, 'index.csv'), encoding='utf-8') as f:
        assert len(f.read().strip().splitlines()) == 3, 'header + 2 catalogue rows'
    assert dump_folder([out], dump_dir)['packs skipped (already done)'] == 1, 'resume skips finished packs'

    # RPF carving: a 2-chunk XML file, a small XML file and a Bink 2 video, each 16-byte aligned
    big = b'<?xml version="1.0"?>\n<CTestData>' + b''.join(b'<Item>%06d</Item>' % i for i in range(9000)) + b'</CTestData>'
    small = b'<?xml version="1.0"?>\n<Small>ok</Small>'
    video = b'KB2n' + struct.pack('<III', 200 - 8, 3, 0) + struct.pack('<III', 0, 640, 360) + bytes(200 - 28)
    arc = bytearray(b'9FPR' + bytes(0x3C))
    for chunk in (big[:65536], big[65536:], small):
        arc += zlib.compress(chunk, 9)
        arc += bytes(-len(arc) % 16)
    arc += video + bytes(-len(arc) % 16)
    rp = os.path.join(d, 'x.rpf')
    open(rp, 'wb').write(bytes(arc))
    a = Archive(rp)
    assert [i.kind for i in a.items] == ['xml', 'xml', 'video'], [i.kind for i in a.items]
    assert a.data(a.items[0]) == big and len(a.items[0].parts) == 2, '64 KB chunks stitched back into one file'
    assert a.items[0].name.endswith('CTestData.xml') and a.data(a.items[1]) == small
    assert a.data(a.items[2]) == video and a.items[2].info.startswith('640')
    assert a.search('item>008999') == [0] and a.search('OK') == [1] and a.search('nowhere') == [], 'search inside files'
    a.close()
    res = dump_archive(rp, os.path.join(d, 'xarc'))
    assert res == {'XML': 2, 'Video': 1}, res
    vp = os.path.join(d, 'v.bk2')
    open(vp, 'wb').write(video)
    assert bink_info(vp) == (3, 640, 360, 30.0), bink_info(vp)  # zero fps field falls back to 30

    # video player lookup: a remembered path wins, else the standard install folder
    fake = os.path.join(d, 'RADVideo', 'radvideo64.exe')
    os.makedirs(os.path.dirname(fake))
    open(fake, 'w').close()
    old_pf = os.environ.get('ProgramFiles(x86)')
    os.environ['ProgramFiles(x86)'] = d
    try:
        assert find_bink_player() == fake and find_bink_player(fake) == fake
        assert find_rad_converter(player=fake) == fake, 'converter found next to the player'
    finally:
        if old_pf is None:
            os.environ.pop('ProgramFiles(x86)')
        else:
            os.environ['ProgramFiles(x86)'] = old_pf

    # carved mode: zero the header, blobs must still be found and guessable
    blank = bytearray(open(src, 'rb').read())
    blank[:etpf_at + 16 + 32] = bytes(etpf_at + 16 + 32)
    open(src, 'wb').write(blank)
    c = Pack(src)
    assert c.carved and [len(t.chain) for t in c.textures] == [1, 3], 'mip chains grouped'
    assert c.image(c.textures[1])[0].size == (256, 256), 'grouped chain shows the full-size mip'
    c.set_grouping(False)
    assert len(c.textures) == 4
    c.close()

    # library index: catalogue the test folder, query it, reopen the archive from the index
    import library
    db = os.path.join(d, 'index.sqlite')
    first = library.index_folder(d.replace('\\', '/'), db)  # forward slashes, like a command-line run
    assert first['packs indexed'] == 3 and library.index_folder(d, db)['packs unchanged'] == 3, first
    import threading as _th
    stopped = _th.Event()
    stopped.set()
    library.index_folder(d, db, cancel=stopped)  # a stopped run must never forget anything
    assert library.index_folder(d, db)['items'] == first['items'], 'same packs under another path spelling were kept'
    con = library.connect(db)
    cats = {c: n for c, n, _ in library.categories(con)}
    assert cats.get('Game data (XML)') == 2 and cats.get('Videos') == 1 and cats.get('Misc textures', 0) == 0, cats
    assert [r[5] for r in library.query(con, text='CTestData')] == ['00000_CTestData.xml']
    rp_items = library.archive_items(con, rp)
    a2 = Archive.from_items(rp, rp_items)
    assert a2.data(a2.items[0]) == big and a2.data(a2.items[2]) == video, 'archive reopened from the index'
    a2.close()
    con.close()

    # 3D models: a hand-built resource (one virtual + one physical chunk) holding a single triangle
    import models
    v = bytearray(0x480)
    for at, val in ((8, 0x100), (0x50, 0x120), (0x120, 0x140), (0x140, 0x160), (0x168, 0x180), (0x180, 0x200),
                    (0x240, 0x300), (0x300, 0x320)):
        struct.pack_into('<Q', v, at, 0x50000000 + val)
    v[0x108:0x10a] = bytes((1, 1))
    struct.pack_into('<H', v, 0x128, 1)
    struct.pack_into('<H', v, 0x170, 1)
    struct.pack_into('<IIQQ', v, 0x258, (0x80 + 24) // 4, 0, models.GEOM_MARK, 0x60000000)
    struct.pack_into('<Q', v, 0x338, 0x60000100)
    v[0x400:0x40c] = b'test_box.#dr'
    p = bytearray(0x130)
    struct.pack_into('<I', p, 0, (0x80 >> 1) << 8 | 0x42)
    struct.pack_into('<3fI3f', p, 0x60, 1, 2, 3, 2, 0.01, 0.01, 0.01)
    for k, (a_, b_, c_) in enumerate(((0, 0, 0), (100, 0, 0), (0, 200, 5))):
        o = 0x80 + k * 8
        p[o + 1:o + 4], p[o + 6:o + 8] = a_.to_bytes(3, 'little'), c_.to_bytes(2, 'little')
        p[o], p[o + 4:o + 6] = b_ & 0xFF, (b_ >> 8).to_bytes(2, 'little')
    struct.pack_into('<III', p, 0x100, 1, 1, 3)
    struct.pack_into('<IHH', p, 0x10c, 0, 0, 0)
    struct.pack_into('<IHHH', p, 0x11c, 0 | 1 << 7 | 2 << 14, 0, 1, 2)
    z1, z2 = zlib.compress(bytes(v), 9), zlib.compress(bytes(p), 9)
    blob = z1 + z2
    assert models.find_models(blob, [0, len(z1)]) == [(0, 'test_box.#dr')]
    mdl = models.decode_model(blob, [0, len(z1)], 0)
    assert mdl['tris'] == [(0, 1, 2)], mdl['tris']
    want = [(1, 2, 3), (2, 2, 3), (1, 4, 3.05)]
    assert all(abs(x - y) < 1e-4 for got, exp in zip(mdl['verts'], want) for x, y in zip(got, exp)), mdl['verts']
    models.write_obj(mdl, os.path.join(d, 'box.obj'))
    assert open(os.path.join(d, 'box.obj')).read().count('\nf ') == 1
    assert models.render(mdl, 64).size == (64, 64)
    assert mdl['tri_mat'] == [0] and mdl['materials'][0]['tex'] is None
    assert models.colourise(mdl, models.Textures(models.texture_index(d))) == 0 and len(mdl['colours']) == 1
    assert models.render(mdl, 64).size == (64, 64)
    models.write_obj(mdl, os.path.join(d, 'box2.obj'), models.Textures({}))
    assert 'newmtl m0' in open(os.path.join(d, 'box2.mtl')).read()
    assert 'usemtl m0' in open(os.path.join(d, 'box2.obj')).read()


if __name__ == '__main__':
    if sys.argv[1:2] == ['--selftest']:
        selftest()
    elif sys.argv[1:2] == ['--rpf']:
        print(rpf_info(sys.argv[2]))
    elif sys.argv[1:2] == ['--install']:
        install()
    elif sys.argv[1:2] == ['--dump'] and len(sys.argv) >= 4:
        # python ttpf_tool.py --dump <folder> <output folder> [--no-headerless]
        res = dump_folder(sys.argv[2], sys.argv[3], headerless='--no-headerless' not in sys.argv, rpf='--rpf' in sys.argv,
                          progress=lambda m, f: print(f'{f * 100:5.1f}%  {m}', flush=True))
        print(dict(res))
    else:
        try:
            run_gui(sys.argv[1] if len(sys.argv) > 1 else None)
        except Exception:
            log_error(traceback.format_exc())
            try:  # pythonw has no console, so say it in a dialog
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f'TTPF Studio could not start.\n\nDetails: {LOG}', 'TTPF Studio', 0x10)
            except Exception:
                raise
