//! Benchmarks for compiled kernels (see `fidget_wgpu::kernel`)
//!
//! This compares three ways of rendering each model in `models/`:
//!
//! - `interpreted`: the default tape interpreter
//! - `compiled`: the whole shape as a single kernel
//!   ([`RenderShape::compiled`])
//! - `leaves`: CSG structure interpreted, with choice-free subexpressions
//!   compiled ([`Kernels::compile_leaves`])
//!
//! Before the Criterion benchmarks (warm render times), it prints a table of
//! pipeline compilation times: the time for the first render of each shape in
//! a fresh context, minus the warm render time.  Driver-level shader caches
//! make repeated runs faster; to measure cold compiles on NVIDIA, run with
//! `__GL_SHADER_DISK_CACHE=0`.
//!
//! Set `FIDGET_KERNEL_BENCH_MODELS` to a comma-separated list of model names
//! to only benchmark those models.
//!
//! Models whose tapes spill registers (e.g. `prospero`) can't be rendered by
//! the WGSL interpreter, and fully compiling very large models takes the
//! driver a long time, so those combinations are skipped (set
//! `FIDGET_KERNEL_BENCH_SLOW=1` to compile them anyway).
use criterion::{BenchmarkId, Criterion};
use fidget_core::{context::Context, context::Tree, vm::VmShape};
use fidget_wgpu::{Gpu, RenderShape, kernel::Kernels, pixel, voxel};
use std::time::{Duration, Instant};

const MODELS: &[(&str, &str)] = &[
    ("hi", include_str!("../../models/hi.vm")),
    ("quarter", include_str!("../../models/quarter.vm")),
    ("tanglecube", include_str!("../../models/tanglecube.vm")),
    ("bear", include_str!("../../models/bear.vm")),
    ("colonnade", include_str!("../../models/colonnade.vm")),
    ("prospero", include_str!("../../models/prospero.vm")),
];

const PIXEL_SIZE: u32 = 1024;
const VOXEL_SIZE: u32 = 512;

/// Operation count below which choice-free subexpressions stay interpreted
const MIN_KERNEL_OPS: usize = 4;

/// Returns the models to benchmark (see `FIDGET_KERNEL_BENCH_MODELS`)
fn models() -> impl Iterator<Item = &'static (&'static str, &'static str)> {
    let only = std::env::var("FIDGET_KERNEL_BENCH_MODELS").ok();
    MODELS.iter().filter(move |(name, _)| {
        only.as_ref()
            .is_none_or(|o| o.split(',').any(|m| m == *name))
    })
}

fn load(text: &str) -> Tree {
    let (ctx, root) = Context::from_text(text.as_bytes()).unwrap();
    ctx.export(root).unwrap()
}

/// Returns `(mode, shape)` pairs for a model, skipping unsupported or slow
/// combinations
fn variants(name: &str, tree: &Tree) -> Vec<(&'static str, RenderShape)> {
    let slow = std::env::var("FIDGET_KERNEL_BENCH_SLOW").is_ok();
    let shape = VmShape::from(tree.clone());
    let mut out = vec![];
    let spills = {
        fidget_bytecode::Bytecode::new(shape.inner().data())
            .unwrap()
            .mem_count()
            > 0
    };
    if !spills {
        out.push(("interpreted", RenderShape::new(&shape).unwrap()));
    }
    if name != "prospero" || slow {
        out.push(("compiled", RenderShape::compiled(&shape).unwrap()));
    }
    let mut kernels = Kernels::new();
    let outer = kernels.compile_leaves(tree, MIN_KERNEL_OPS);
    out.push((
        "leaves",
        RenderShape::with_kernels(&VmShape::from(outer), &kernels).unwrap(),
    ));
    out
}

fn voxel_config() -> fidget_raster::voxel::RenderConfig {
    fidget_raster::voxel::RenderConfig::from_size(
        fidget_raster::voxel::RenderSize::new(
            VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE,
        ),
    )
}

fn pixel_config() -> fidget_raster::pixel::RenderConfig {
    fidget_raster::pixel::RenderConfig::from_size(
        fidget_raster::pixel::RenderSize::new(PIXEL_SIZE, PIXEL_SIZE),
    )
}

/// Renders in 2D, returning the elapsed time
fn render_2d(
    gpu: &Gpu,
    ctx: &pixel::Context,
    ws: &mut pixel::Workspace,
    shape: &RenderShape,
) -> Duration {
    let mut out = gpu.read_buffer_for(ws.output());
    let t = Instant::now();
    ctx.run(shape, ws, &mut out, pixel_config()).unwrap();
    t.elapsed()
}

/// Renders in 3D, returning the elapsed time
fn render_3d(
    gpu: &Gpu,
    ctx: &voxel::Context,
    ws: &mut voxel::Workspace,
    shape: &RenderShape,
) -> Duration {
    let mut out = gpu.read_buffer_for(ws.output());
    let t = Instant::now();
    ctx.run(shape, ws, &mut out, voxel_config()).unwrap();
    t.elapsed()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Prints pipeline compilation times and kernel statistics
fn compile_report(gpu: &Gpu) {
    println!(
        "\n| model | ops | kernels | mode | 2D compile | 2D render \
         | 3D compile | 3D render |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for (name, text) in models() {
        let tree = load(text);
        let ops = {
            let s = VmShape::from(tree.clone());
            fidget_bytecode::Bytecode::new(s.inner().data())
                .unwrap()
                .len()
                / 2
        };
        let mut kernels = Kernels::new();
        kernels.compile_leaves(&tree, MIN_KERNEL_OPS);
        for (mode, shape) in variants(name, &tree) {
            // Fresh contexts, so that pipelines are built from scratch
            let pctx = pixel::Context::new(gpu);
            let mut pws = pctx.workspace();
            let first = render_2d(gpu, &pctx, &mut pws, &shape);
            let warm = median(
                (0..5)
                    .map(|_| render_2d(gpu, &pctx, &mut pws, &shape))
                    .collect(),
            );
            let vctx = voxel::Context::new(gpu);
            let mut vws = vctx.workspace();
            let first3 = render_3d(gpu, &vctx, &mut vws, &shape);
            let warm3 = median(
                (0..5)
                    .map(|_| render_3d(gpu, &vctx, &mut vws, &shape))
                    .collect(),
            );
            let k = match mode {
                "interpreted" => 0,
                "compiled" => 1,
                _ => kernels.len(),
            };
            println!(
                "| {name} | {ops} | {k} | {mode} | {:.2} s | {:.2} ms \
                 | {:.2} s | {:.2} ms |",
                first.saturating_sub(warm).as_secs_f64(),
                warm.as_secs_f64() * 1e3,
                first3.saturating_sub(warm3).as_secs_f64(),
                warm3.as_secs_f64() * 1e3,
            );
        }
    }
    println!();
}

fn render_benches(c: &mut Criterion, gpu: &Gpu) {
    let pctx = pixel::Context::new(gpu);
    let mut pws = pctx.workspace();
    let vctx = voxel::Context::new(gpu);
    let mut vws = vctx.workspace();
    for (name, text) in models() {
        let tree = load(text);
        let shapes = variants(name, &tree);

        let mut group = c.benchmark_group(format!("kernels-2d-{name}"));
        for (mode, shape) in &shapes {
            render_2d(gpu, &pctx, &mut pws, shape); // build pipelines
            group.bench_function(BenchmarkId::from_parameter(mode), |b| {
                b.iter(|| render_2d(gpu, &pctx, &mut pws, shape))
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("kernels-3d-{name}"));
        for (mode, shape) in &shapes {
            render_3d(gpu, &vctx, &mut vws, shape);
            group.bench_function(BenchmarkId::from_parameter(mode), |b| {
                b.iter(|| render_3d(gpu, &vctx, &mut vws, shape))
            });
        }
        group.finish();
    }
}

fn main() {
    let gpu = pollster::block_on(Gpu::init()).unwrap();
    if std::env::var("FIDGET_KERNEL_BENCH_NO_REPORT").is_err() {
        compile_report(&gpu);
    }
    let mut c = Criterion::default().configure_from_args();
    render_benches(&mut c, &gpu);
    c.final_summary();
}
