# Media for the `wgsl-kernels` PR

Benchmark graphs and annotated renders for compiled WGSL kernels in
`fidget-wgpu`, measured on an RTX 4090 (NVIDIA 610.57, Vulkan, Linux).

- `images/`: graphs and per-model galleries
- `scripts/kernel_gallery.rs`: renders every model in each mode. It's an
  example program for `fidget-wgpu`; run it from the repository root with
  `cargo run --release -p fidget-wgpu --example kernel_gallery <out dir>`.
- `scripts/plot.py`, `scripts/annotate.py`: build the graphs from the
  `benches/kernels.rs` output, and the annotated galleries from the gallery
  renders (matplotlib, ImageMagick)
- `scripts/data.json`, `scripts/gallery_summary.json`: the measurements
