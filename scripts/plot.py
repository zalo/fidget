import re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = open('/tmp/kbench.log').read() + open('/tmp/kbench_prospero.log').read()
# Cold compile report rows
compile_ = {}
report_render = {}
for m in re.finditer(r'^\| (\w+) \| (\d+) \| (\d+) \| (\w+) \| ([\d.]+) s \| ([\d.]+) ms \| ([\d.]+) s \| ([\d.]+) ms \|', log, re.M):
    model, ops, k, mode = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    compile_[(model, '2d', mode)] = float(m.group(5))
    compile_[(model, '3d', mode)] = float(m.group(7))
    report_render[(model, '2d', mode)] = float(m.group(6))
    report_render[(model, '3d', mode)] = float(m.group(8))
# Criterion warm medians
render = {}
for m in re.finditer(r'^kernels-(2d|3d)-(\w+)/(\w+)\s*\n?\s*time:\s+\[([\d.]+) (\S+) ([\d.]+) (\S+) ([\d.]+) (\S+)\]', log, re.M):
    dim, model, mode = m.group(1), m.group(2), m.group(3)
    v, u = float(m.group(6)), m.group(7)
    ms = v / 1000 if u in ('µs', 'us') else (v * 1000 if u == 's' else v)
    render[(model, dim, mode)] = ms
for k, v in report_render.items():
    render.setdefault(k, v)  # prospero compiled: report numbers

OPS = {'hi': 48, 'quarter': 11, 'tanglecube': 21, 'bear': 543, 'colonnade': 682, 'prospero': 7211}
MODELS = list(OPS)
MODES = ['interpreted', 'compiled', 'leaves']
COLORS = {'interpreted': '#4c72b0', 'compiled': '#dd8452', 'leaves': '#55a868'}
LABELS = {'interpreted': 'interpreter', 'compiled': 'whole-shape kernel', 'leaves': 'leaf kernels'}
json.dump({'compile': {'|'.join(k): v for k, v in compile_.items()},
           'render': {'|'.join(k): v for k, v in render.items()}},
          open('/tmp/kgraphs/data.json', 'w'), indent=1)

def time_to_n(dim, fname, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    N = np.logspace(0, 5, 300)
    pending = None
    for ax, model in zip(axes.flat, MODELS):
        base = None
        for mode in MODES:
            c = compile_.get((model, dim, mode)); r = render.get((model, dim, mode))
            if c is None or r is None:
                continue
            t = c + N * r / 1000
            ax.plot(N, t, color=COLORS[mode], lw=2, label=f"{LABELS[mode]}: {c:.2g} s + {r:.3g} ms/frame")
            if mode == 'interpreted' or (base is None and mode == 'leaves'):
                base = (c, r)
            if mode == 'compiled' and base is None:
                pending = (c, r)
            elif mode == 'leaves' and model == 'prospero' and 'pending' in dir() and pending:
                # No interpreter: compare the whole-shape kernel against leaves
                (cc, rc) = pending
                n = (cc - c) / ((r - rc) / 1000)
                y = cc + n * rc / 1000
                ax.plot([n], [y], 'o', color=COLORS['compiled'], ms=7, mec='k')
                ax.annotate(f"whole-shape kernel\novertakes leaves\nafter {n:,.0f} frames", (n, y),
                            textcoords='offset points', xytext=(-60, 20), fontsize=8, color=COLORS['compiled'])
                pending = None
            elif base and mode != 'interpreted' and r < base[1] and c > base[0]:
                n = (c - base[0]) / ((base[1] - r) / 1000)
                if 1 <= n <= 1e5:
                    y = base[0] + n * base[1] / 1000
                    ax.plot([n], [y], 'o', color=COLORS[mode], ms=7, mec='k')
                    ax.annotate(f"break-even\n{n:,.0f} frames", (n, y), textcoords='offset points',
                                xytext=(-10 if mode == 'compiled' else 10, 12 if mode == 'compiled' else -30),
                                fontsize=8, color=COLORS[mode], ha='center')
        if model == 'prospero':
            ax.text(0.97, 0.03, "interpreter: unsupported\n(tape spills registers → NaN)", transform=ax.transAxes,
                    va='bottom', ha='right', fontsize=9, color=COLORS['interpreted'],
                    bbox=dict(fc='white', ec=COLORS['interpreted'], alpha=0.8))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f"{model} ({OPS[model]} ops)")
        ax.set_xlabel("frames rendered"); ax.set_ylabel("total time (s)")
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.005, "Dots mark break-even vs. the interpreter (frames needed to repay the extra pipeline compile time). "
             "Interpreter pipelines are shared by every shape, so in practice their cost is paid once per process.",
             ha='center', fontsize=9, color='dimgray')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(fname, dpi=110)

time_to_n('3d', '/tmp/kgraphs/time_to_frames_3d.png',
          "3D (512³): total time = cold pipeline compile + N × warm render   [RTX 4090, Vulkan]")
time_to_n('2d', '/tmp/kgraphs/time_to_frames_2d.png',
          "2D (1024²): total time = cold pipeline compile + N × warm render   [RTX 4090, Vulkan]")

# Compile vs render scatter
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, dim, size in zip(axes, ['2d', '3d'], ['1024²', '512³']):
    for mode in MODES:
        xs, ys, names = [], [], []
        for model in [m for m in MODELS if m != 'prospero']:
            c = compile_.get((model, dim, mode)); r = render.get((model, dim, mode))
            if c is None or r is None:
                continue
            xs.append(c); ys.append(r); names.append(model)
        ax.scatter(xs, ys, s=70, color=COLORS[mode], label=LABELS[mode], edgecolor='k', zorder=3)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8, color=COLORS[mode])
    # connect modes per model
    for model in [m for m in MODELS if m != 'prospero']:
        pts = [(compile_.get((model, dim, m)), render.get((model, dim, m))) for m in ('interpreted', 'leaves', 'compiled')]
        pts = [p for p in pts if None not in p]
        if len(pts) > 1:
            ax.plot(*zip(*pts), color='gray', lw=0.8, alpha=0.5, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10, 20]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("cold pipeline compile time (s)  →  worse")
    ax.set_ylabel("warm render time (ms)  →  worse")
    ax.set_title(f"{dim.upper()} ({size}): compile cost vs. runtime (prospero omitted)")
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    ax.text(0.98, 0.02, "lower-left is better; gray lines join each model:\ninterpreter → leaf kernels → whole-shape kernel",
            ha='right',
            transform=ax.transAxes, fontsize=8, color='gray')
fig.tight_layout()
fig.savefig('/tmp/kgraphs/compile_vs_render.png', dpi=110)

# Speedup bars
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, dim in zip(axes, ['2d', '3d']):
    x = np.arange(len(MODELS)); w = 0.38
    for i, mode in enumerate(['compiled', 'leaves']):
        vals = []
        for model in MODELS:
            b = render.get((model, dim, 'interpreted')); r = render.get((model, dim, mode))
            vals.append(b / r if b and r else 0)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COLORS[mode], label=LABELS[mode], edgecolor='k')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0.05) * 1.05, f"{v:.1f}×" if v else "n/a",
                    ha='center', fontsize=8)
    ax.axhline(1, color='k', lw=0.8)
    ax.set_xticks(x, [f"{m}\n({OPS[m]} ops)" for m in MODELS])
    ax.set_yscale('log'); ax.set_ylabel("warm render speedup vs. interpreter")
    ax.set_title(f"{dim.upper()}: runtime speedup (prospero: interpreter unsupported)")
    ax.legend(); ax.grid(True, axis='y', which='both', alpha=0.25)
fig.tight_layout()
fig.savefig('/tmp/kgraphs/speedup.png', dpi=110)
print("ok")
