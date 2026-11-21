"""Library index: one SQLite catalogue of everything in a game folder (every texture of every .ttpf pack and
every file recovered from every .rpf archive), sorted into categories so it can be browsed and searched.

Indexing is incremental: a pack is re-read only when its size or modification time changed, so the first run
is slow (big archives take minutes) and later runs take seconds. Stop any time; finished packs are kept.
"""
import hashlib
import json
import os
import sqlite3
import time

# texture pack name -> library category (first match wins)
TEXTURE_CATEGORIES = [
    ('decorations', 'Tattoos & decals'), ('peds', 'Peds & clothing'), ('vehicle', 'Vehicles'), ('weapon', 'Weapons'),
    ('octa_imposters', 'Props'), ('props', 'Props'), ('interiors', 'Interiors'), ('terrain', 'Terrain'),
    ('maps', 'Maps & buildings'), ('platform', 'UI'), ('ui', 'UI'), ('graphics', 'Graphics'), ('vfx', 'Effects'),
    ('lighting', 'Lighting'), ('textures_loose', 'Misc textures'),
]
ARCHIVE_CATEGORIES = {'xml': 'Game data (XML)', 'text': 'Text', 'video': 'Videos', 'psin': 'Scripts & resources (PSIN)',
                      'skel': 'Skeletons & rigged models', 'bin': 'Other binary'}
SCHEMA = """
CREATE TABLE IF NOT EXISTS packs (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, kind TEXT, items INTEGER, seconds REAL);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY, pack TEXT, kind TEXT, category TEXT, sub TEXT, name TEXT,
    size INTEGER, w INTEGER, h INTEGER, fmt TEXT, ref INTEGER, parts TEXT, info TEXT);
CREATE INDEX IF NOT EXISTS items_cat ON items(category, sub);
CREATE INDEX IF NOT EXISTS items_pack ON items(pack);
"""


def texture_category(pack_path):
    name = os.path.basename(pack_path).lower()
    return next((cat for key, cat in TEXTURE_CATEGORIES if key in name), 'Other textures')


def db_path(folder, base=None):
    from ttpf_tool import APP_DIR
    key = hashlib.sha1(os.path.normcase(os.path.abspath(folder)).encode()).hexdigest()[:16]
    d = os.path.join(base or APP_DIR, 'index')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + '.sqlite')


def norm(path):
    """One spelling per file: C:/x\\y and C:\\x\\y must be the same pack."""
    return os.path.normpath(os.path.abspath(path))


def connect(path):
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript(SCHEMA)
    with con:  # older indexes may hold mixed path spellings; bring them to one form so nothing is re-read
        for (p,) in con.execute('SELECT path FROM packs').fetchall():
            if norm(p) != p:
                con.execute('DELETE FROM packs WHERE path = ?', (norm(p),))
                con.execute('UPDATE packs SET path = ? WHERE path = ?', (norm(p), p))
                con.execute('UPDATE items SET pack = ? WHERE pack = ?', (norm(p), p))
    return con


def _texture_rows(path, progress):
    import ttpf_tool as tt
    pk = tt.Pack(path, progress)
    cat = texture_category(path)
    rows = []
    for t in pk.textures:
        if t.status in ('missing', 'unsupported'):
            continue
        size = sum(e.length or 0 for e in t.chain)
        sub = f'{t.w}×{t.h}' if t.w else 'size unknown (no header)'
        rows.append((path, 'texture', cat, sub, t.name, size, t.w or 0, t.h or 0, t.fmt or '', t.n, None, t.status))
    pk.close()
    return rows


def _archive_rows(path, progress, cancel):
    import ttpf_tool as tt
    a = tt.Archive(path, progress)
    rows = []
    for it in a.items:
        if cancel and cancel.is_set():
            break
        cat = ARCHIVE_CATEGORIES[it.kind]
        if it.kind == 'xml':
            sub = it.name.split('_', 1)[1].rsplit('.', 1)[0]  # XML root element, e.g. CVehicleModelInfo
        elif it.kind == 'video':
            sub = it.info.split(',')[0]
        else:
            sub = os.path.basename(path)
        rows.append((path, it.kind, cat, sub, it.name, it.length, 0, 0, '', it.n, json.dumps(it.parts), it.info))
    a.close()
    return rows


def index_folder(folder, db, progress=lambda msg, frac: None, cancel=None):
    """Bring the catalogue in `db` up to date with `folder`. Returns a dict of counts."""
    import ttpf_tool as tt
    con = connect(db)
    packs = [(norm(p), k, n) for p, k, n in tt.scan_folder(folder) if k in ('indexed', 'headerless', 'rpf', 'rpf_noheader')]
    known = {p: (s, m) for p, s, m in con.execute('SELECT path, size, mtime FROM packs')}
    stats = {'packs indexed': 0, 'packs unchanged': 0, 'packs failed': 0}
    for i, (path, kind, size) in enumerate(packs):
        if cancel and cancel.is_set():
            break
        mtime = os.path.getmtime(path)
        if known.get(path) == (size, mtime):
            stats['packs unchanged'] += 1
            continue
        name = os.path.basename(path)
        started = time.time()

        def sub_progress(f, i=i, name=name):
            progress(f'Indexing {name}  ({i + 1}/{len(packs)})', (i + max(0.0, min(1.0, f))) / len(packs))
        sub_progress(0)
        try:
            rows = (_archive_rows(path, sub_progress, cancel) if path.lower().endswith('.rpf')
                    else _texture_rows(path, sub_progress))
        except Exception as ex:
            stats['packs failed'] += 1
            tt.log_error(f'index: {path}: {ex}\n')
            continue
        if cancel and cancel.is_set():
            break  # never store a half-read pack
        with con:
            con.execute('DELETE FROM items WHERE pack = ?', (path,))
            con.executemany('INSERT INTO items (pack, kind, category, sub, name, size, w, h, fmt, ref, parts, info) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
            con.execute('INSERT OR REPLACE INTO packs VALUES (?, ?, ?, ?, ?, ?)',
                        (path, size, mtime, kind, len(rows), time.time() - started))
        stats['packs indexed'] += 1
    finished = not (cancel and cancel.is_set())
    with con:  # forget packs that are no longer in the folder (only after a complete run: a stopped run proves nothing)
        present = {os.path.normcase(p) for p, *_ in packs}
        root_ = os.path.normcase(norm(folder)) + os.sep
        for (p,) in con.execute('SELECT path FROM packs').fetchall() if finished else []:
            if os.path.normcase(p) not in present and os.path.normcase(p).startswith(root_):
                con.execute('DELETE FROM items WHERE pack = ?', (p,))
                con.execute('DELETE FROM packs WHERE path = ?', (p,))
    stats['items'] = con.execute('SELECT COUNT(*) FROM items').fetchone()[0]
    con.close()
    progress('Index up to date' if not (cancel and cancel.is_set()) else 'Stopped', 1.0)
    return stats


def categories(con):
    """[(category, total, [(sub, count), ...]), ...] sorted for display."""
    out = {}
    for cat, sub, n in con.execute('SELECT category, sub, COUNT(*) FROM items GROUP BY category, sub'):
        out.setdefault(cat, []).append((sub, n))
    order = [c for _, c in TEXTURE_CATEGORIES] + list(ARCHIVE_CATEGORIES.values())
    res = []
    for cat in sorted(out, key=lambda c: (order.index(c) if c in order else len(order), c)):
        subs = sorted(out[cat], key=lambda sn: (-sn[1], str(sn[0])))
        res.append((cat, sum(n for _, n in subs), subs))
    return res


def query(con, category=None, sub=None, text=None, limit=5000):
    sql, args = 'SELECT id, pack, kind, category, sub, name, size, w, h, fmt, ref, parts, info FROM items WHERE 1=1', []
    if category:
        sql += ' AND category = ?'
        args.append(category)
    if sub:
        sql += ' AND sub = ?'
        args.append(sub)
    if text:
        sql += ' AND (name LIKE ? OR sub LIKE ? OR pack LIKE ? OR info LIKE ?)'
        args += [f'%{text}%'] * 4
    sql += ' ORDER BY category, sub, pack, ref LIMIT ?'
    args.append(limit)
    return con.execute(sql, args).fetchall()


def archive_items(con, pack):
    """Rebuild an archive's item list from the index (for fast opening)."""
    import ttpf_tool as tt
    items = []
    for kind, name, size, parts, info in con.execute(
            'SELECT kind, name, size, parts, info FROM items WHERE pack = ? ORDER BY ref', (pack,)):
        ps = [tuple(x) for x in json.loads(parts)]
        it = tt.Item(kind, ps[0][0], sum(c for _, c in ps), size, name, info or '')
        it.parts = ps
        items.append(it)
    return items
