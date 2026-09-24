import json, subprocess, os
G = '/tmp/kgallery'; OUT = '/tmp/kgraphs'
data = json.load(open(f'{OUT}/data.json'))
compile_ = {tuple(k.split('|')): v for k, v in data['compile'].items()}
render = {tuple(k.split('|')): v for k, v in data['render'].items()}
summary = {(e['model'], e['mode']): e for e in json.load(open(f'{G}/summary.json'))}
OPS = {'hi': 48, 'quarter': 11, 'tanglecube': 21, 'bear': 543, 'colonnade': 682, 'prospero': 7211}
MODES = ['interpreted', 'compiled', 'leaves']
LABEL = {'interpreted': 'Interpreter', 'compiled': 'Whole-shape kernel', 'leaves': 'Leaf kernels'}
COLOR = {'interpreted': '#4c72b0', 'compiled': '#dd8452', 'leaves': '#55a868'}
T = 400  # tile size

def run(*a):
    a = list(a)
    # Use a clean sans-serif font for every text operation
    subprocess.run(['magick', '-font', 'Noto-Sans-Regular', *a], check=True)

def tile(model, mode, dim):
    src = f'{G}/{model}_{mode}_{dim}.ppm'
    dst = f'{OUT}/t_{model}_{mode}_{dim}.png'
    s = summary.get((model, mode), {})
    c = compile_.get((model, dim, mode)); r = render.get((model, dim, mode))
    if s.get('unsupported') or not os.path.exists(src):
        run('-size', f'{T}x{T}', 'xc:#2a2a30', '-fill', '#ff8080', '-gravity', 'center', '-pointsize', '20',
            '-annotate', '+0-20', 'not supported by the\nWGSL interpreter',
            '-fill', '#bbbbbb', '-pointsize', '15', '-annotate', '+0+45',
            'tape spills registers (423 slots):\nrenders NaN, hangs at >= 255 px', dst)
        return dst
    timing = f"{r:.2f} ms/frame" if r is not None else ""
    comp = f"compile {c:.2f} s" if c is not None and c < 10 else (f"compile {c:.0f} s" if c else "")
    if dim == '2d':
        tot = s['fills'] + s['points']
        extra = f"{100 * s['fills'] / tot:.0f}% of pixels proven by tile fills"
    else:
        d = s.get('normals_differ', 0)
        extra = "reference normals" if mode == 'interpreted' else (
            f"{d} normals differ (red)" if d else "normals identical to interpreter")
        if model == 'prospero':
            extra = "no interpreter reference"
    run(src, '-resize', f'{T}x{T}',
        '-fill', '#000000a0', '-draw', f'rectangle 0,{T-58} {T},{T}',
        '-fill', 'white', '-pointsize', '17', '-gravity', 'southwest', '-annotate', '+8+32', f"{timing}   {comp}",
        '-fill', '#dddddd', '-pointsize', '14', '-annotate', '+8+10', extra, dst)
    return dst

def legend():
    dst = f'{OUT}/legend.png'
    items = [('#28406e', 'inside (tile fill)'), ('#ececec', 'outside (tile fill)'),
             ('#466ebe', 'inside (point sample)'), ('#ffd696', 'outside (point sample)'),
             ('#ff1e1e', '3D: normal differs from interpreter')]
    args = ['-size', f'{3*T+40}x34', 'xc:white', '-pointsize', '15']
    x = 12
    for col, txt in items:
        args += ['-fill', col, '-stroke', '#555', '-draw', f'rectangle {x},9 {x+16},25', '-stroke', 'none',
                 '-fill', 'black', '-annotate', f'+{x+22}+23', txt]
        x += 22 + len(txt) * 8 + 18
    run(*args, dst)
    return dst

leg = legend()
for model in OPS:
    rows = []
    for dim in ['2d', '3d']:
        tiles = [tile(model, m, dim) for m in MODES]
        row = f'{OUT}/row_{model}_{dim}.png'
        run(*tiles, '-bordercolor', 'white', '-border', '6', '+append', row)
        # row label
        run(row, '-gravity', 'west', '-background', 'white', '-splice', '44x0', '-fill', '#333', '-pointsize', '22',
            '-annotate', '+6+0' if False else '+8+0', '2D' if dim == '2d' else '3D', row)
        rows.append(row)
    header = f'{OUT}/hdr_{model}.png'
    args = ['-size', f'{3*(T+12)+44}x70', 'xc:white', '-fill', 'black', '-pointsize', '26', '-gravity', 'northwest',
            '-annotate', '+12+6', f'{model}.vm — {OPS[model]} ops']
    for i, m in enumerate(MODES):
        args += ['-fill', COLOR[m], '-pointsize', '19', '-annotate', f'+{44 + i*(T+12) + 10}+44', LABEL[m]]
    run(*args, header)
    run(header, *rows, leg, '-background', 'white', '-gravity', 'west', '-append', f'{OUT}/gallery_{model}.png')
print('ok')
