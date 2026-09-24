//! Renders every model in `models/` with each evaluation mode, writing PPM
//! images and a JSON summary to the directory given as the first argument.
//!
//! 2D images show how each pixel was decided: tile fills (proven by interval
//! arithmetic) vs. individual point samples.  3D images are shaded normals;
//! pixels whose normal differs from the interpreter's are marked in red.
use fidget_core::{context::Context, context::Tree, vm::VmShape};
use fidget_raster::pixel::DistancePixel;
use fidget_wgpu::{Gpu, RenderShape, kernel::Kernels, pixel, voxel};
use std::io::Write;

const MODELS: &[&str] =
    &["hi", "quarter", "tanglecube", "bear", "colonnade", "prospero"];
const SIZE: u32 = 512;

fn write_ppm(path: &str, w: u32, h: u32, rgb: &[[u8; 3]]) {
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    for p in rgb {
        f.write_all(p).unwrap();
    }
}

fn camera() -> nalgebra::Matrix4<f32> {
    // Same view as `fidget/benches/voxel.rs`
    let s = 1.0 / 0.7;
    let scale = nalgebra::Scale3::new(s, s, s);
    let pitch = nalgebra::Rotation3::new(
        nalgebra::Vector3::x() * 60.0 * std::f32::consts::PI / 180.0,
    );
    let roll = nalgebra::Rotation3::new(
        nalgebra::Vector3::z() * 30.0 * std::f32::consts::PI / 180.0,
    );
    let mut cam = nalgebra::Transform3::identity();
    *cam.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    roll.to_homogeneous()
        * pitch.to_homogeneous()
        * scale.to_homogeneous()
        * cam.to_homogeneous()
}

fn main() {
    let out = std::env::args().nth(1).expect("output directory");
    std::fs::create_dir_all(&out).unwrap();
    let only = std::env::var("MODELS").ok();
    let gpu = pollster::block_on(Gpu::init()).unwrap();
    let pctx = pixel::Context::new(&gpu);
    let vctx = voxel::Context::new(&gpu);
    let mut pws = pctx.workspace();
    let mut vws = vctx.workspace();
    let mut summary = vec![];

    for name in MODELS {
        if only.as_ref().is_some_and(|o| !o.split(',').any(|m| m == *name)) {
            continue;
        }
        let text =
            std::fs::read_to_string(format!("models/{name}.vm")).unwrap();
        let (ctx, root) = Context::from_text(text.as_bytes()).unwrap();
        let tree: Tree = ctx.export(root).unwrap();
        let shape = VmShape::from(tree.clone());
        let spills = {
            use fidget_core::eval::Function;
            fidget_bytecode::Bytecode::new(shape.inner().data())
                .unwrap()
                .mem_count()
                > 0
        };
        let mut kernels = Kernels::new();
        let outer = kernels.compile_leaves(&tree, 4);
        let modes: Vec<(&str, RenderShape)> = vec![
            ("interpreted", RenderShape::new(&shape).unwrap()),
            ("compiled", RenderShape::compiled(&shape).unwrap()),
            (
                "leaves",
                RenderShape::with_kernels(&VmShape::from(outer), &kernels)
                    .unwrap(),
            ),
        ];

        let mut reference_normals: Option<Vec<[f32; 3]>> = None;
        for (mode, rs) in &modes {
            // The interpreter can't evaluate spilling tapes (and hangs on
            // large `prospero` renders), so we draw a placeholder instead
            if spills && *mode == "interpreted" {
                summary.push(format!(
                    "{{\"model\":\"{name}\",\"mode\":\"{mode}\",\
                     \"unsupported\":true}}"
                ));
                continue;
            }

            // 2D
            let cfg = fidget_raster::pixel::RenderConfig::from_size(
                fidget_raster::pixel::RenderSize::new(SIZE, SIZE),
            );
            let t = std::time::Instant::now();
            let mut buf = gpu.read_buffer_for(pws.output());
            let im = pctx.run(rs, &mut pws, &mut buf, cfg).unwrap();
            let first_2d = t.elapsed().as_secs_f64();
            let (mut fills, mut points) = (0usize, 0usize);
            let rgb: Vec<[u8; 3]> = im
                .iter()
                .map(|p| match p.unpack() {
                    DistancePixel::Fill { inside, .. } => {
                        fills += 1;
                        if inside { [40, 60, 110] } else { [236, 236, 236] }
                    }
                    DistancePixel::Value(v) => {
                        points += 1;
                        if v.is_nan() {
                            [255, 0, 255]
                        } else if v < 0.0 {
                            [70, 110, 190]
                        } else {
                            [255, 214, 150]
                        }
                    }
                })
                .collect();
            write_ppm(&format!("{out}/{name}_{mode}_2d.ppm"), SIZE, SIZE, &rgb);

            // 3D
            let cfg = fidget_raster::voxel::RenderConfig {
                world_to_model: camera(),
                ..fidget_raster::voxel::RenderConfig::from_size(
                    fidget_raster::voxel::RenderSize::new(SIZE, SIZE, SIZE),
                )
            };
            let t = std::time::Instant::now();
            let mut buf = gpu.read_buffer_for(vws.output());
            let im = vctx.run(rs, &mut vws, &mut buf, cfg).unwrap();
            let first_3d = t.elapsed().as_secs_f64();
            let normals: Vec<[f32; 3]> = im.iter().map(|p| p.normal).collect();
            let mut differ = 0usize;
            let rgb: Vec<[u8; 3]> = im
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if p.depth == 0 {
                        return [30, 30, 36];
                    }
                    let [x, y, z] = p.normal;
                    let n = (x * x + y * y + z * z).sqrt().max(1e-12);
                    if let Some(r) = &reference_normals {
                        let q = r[i];
                        let d = (0..3)
                            .map(|k| (q[k] - p.normal[k]).abs())
                            .fold(0.0, f32::max);
                        if d > 1e-4 * n.max(1.0) {
                            differ += 1;
                            return [255, 30, 30];
                        }
                    }
                    let l = [0.4f32, 0.5, 0.77];
                    let shade =
                        ((x * l[0] + y * l[1] + z * l[2]) / n).max(0.0);
                    let depth = p.depth as f32 / SIZE as f32;
                    let v = 0.25 + 0.75 * shade;
                    [
                        (v * (180.0 + 60.0 * depth)) as u8,
                        (v * (190.0 + 40.0 * depth)) as u8,
                        (v * 230.0) as u8,
                    ]
                })
                .collect();
            write_ppm(&format!("{out}/{name}_{mode}_3d.ppm"), SIZE, SIZE, &rgb);
            if reference_normals.is_none() && *mode == "interpreted" {
                reference_normals = Some(normals);
            }
            summary.push(format!(
                "{{\"model\":\"{name}\",\"mode\":\"{mode}\",\
                 \"fills\":{fills},\"points\":{points},\
                 \"normals_differ\":{differ},\
                 \"first_2d\":{first_2d},\"first_3d\":{first_3d}}}"
            ));
            eprintln!("{name} {mode}: 2D {first_2d:.2}s, 3D {first_3d:.2}s");
        }
    }
    std::fs::write(
        format!("{out}/summary.json"),
        format!("[{}]", summary.join(",\n")),
    )
    .unwrap();
}
