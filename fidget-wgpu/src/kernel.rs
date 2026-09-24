//! Compiled kernels: tape subexpressions compiled to straight-line WGSL
//!
//! By default, every GPU stage evaluates shapes with a tape interpreter: a
//! loop around a `switch` on the opcode, with values stored in a register
//! array.  The interpreter is flexible (one pipeline serves every shape, and
//! tapes can be simplified per tile), but each operation pays for dispatch,
//! and the register array usually lives in (slow) thread-local memory.
//!
//! A **kernel** is a subexpression which is instead compiled into a WGSL
//! function, with one `let` binding per operation.  The shader compiler can
//! then keep values in hardware registers and schedule the whole expression.
//!
//! # Semantics
//! Kernels are declared by wrapping a [`Tree`] with [`Kernels::compile`],
//! which returns a *placeholder* tree to use in its place (in the spirit of
//! Numba's `@jit` decorator: you mark the hot numeric code, and the
//! surrounding program stays interpreted):
//!
//! ```
//! # use fidget_core::context::Tree;
//! # use fidget_core::vm::VmShape;
//! # use fidget_wgpu::{RenderShape, kernel::Kernels};
//! let (x, y, z) = Tree::axes();
//! let gyroid = x.clone().sin() * y.clone().cos()
//!     + y.clone().sin() * z.clone().cos()
//!     + z.clone().sin() * x.clone().cos();
//! let sphere = (x.square() + y.square() + z.square()).sqrt() - 5.0;
//!
//! let mut kernels = Kernels::new();
//! let g = kernels.compile(gyroid.abs() - 0.2); // compiled
//! let shape = g.max(sphere); // interpreted
//! let shape = RenderShape::with_kernels(&VmShape::from(shape), &kernels)?;
//! # Ok::<(), fidget_wgpu::RenderShapeError>(())
//! ```
//!
//! - A kernel is a pure function of the evaluation point `(x, y, z)`.  It's
//!   evaluated with exactly the same operation definitions as the
//!   interpreter, in every mode (float, interval, and forward-mode gradients),
//!   so compiled and interpreted evaluation agree.
//! - In the enclosing tape, a kernel behaves like an input variable.  It's
//!   **opaque to tape simplification**: interval evaluation doesn't record
//!   `min` / `max` choices inside the kernel, so it's never partially
//!   simplified.  The enclosing tape's choices still apply around it, so a
//!   kernel which is pruned by an outer `min` / `max` isn't evaluated at all.
//! - A kernel is evaluated at the same point as the enclosing tape.  Remapping
//!   a placeholder tree (e.g. with [`Tree::remap_xyz`]) has no effect on the
//!   kernel's inputs: apply transforms *inside* the kernel's tree instead.
//! - Kernels may only use the `X`, `Y`, and `Z` axes (not other variables).
//! - Every distinct set of kernels needs its own compute pipelines (one per
//!   GPU stage), which are compiled by the driver on first use and then
//!   cached by content (see below).
//!
//! # Performance trade-offs
//! Compiled code trades pipeline compilation time (and tape simplification)
//! for faster evaluation:
//!
//! - **Faster evaluation**: there's no per-operation dispatch, and values
//!   live in hardware registers.  The benefit grows with tape length; small
//!   shapes are dominated by fixed per-render costs.
//! - **Pipeline compilation**: interpreter pipelines are shared by every
//!   shape, but each distinct set of kernels needs its own pipelines: one per
//!   GPU stage (3 for 2D rendering, 5 for 3D), each containing the kernels in
//!   a different value type (interval, float, or gradient).  The driver's
//!   compile time depends on the size of the kernels and on how many values
//!   are live at once, and ranges from well under a second (dozens of
//!   operations) to seconds (hundreds) to minutes (thousands).  Pipelines are
//!   built on first use and cached by the generated source (keeping the most
//!   recently used [`MAX_KERNEL_PROGRAMS`](crate::MAX_KERNEL_PROGRAMS) kernel
//!   sets per stage), so rendering the same kernels again is free.
//! - **Tape simplification**: the interpreter evaluates ever-shorter tapes as
//!   it subdivides space, but a kernel always runs in full.  Keeping CSG
//!   operations interpreted ([`Kernels::compile_leaves`]) preserves
//!   simplification around the kernels, and splits one huge function into
//!   many smaller ones, which can bound compile time for large shapes.
//!   For moderately sized shapes, compiling the whole shape
//!   ([`RenderShape::compiled`](crate::RenderShape::compiled)) is usually
//!   faster to evaluate despite losing simplification.
//! - **Register spills**: the tape interpreter doesn't support tapes which
//!   spill registers (more than 255 live values), but kernels resolve spills
//!   at compile time, so large shapes can be rendered with kernels.
use crate::RenderShapeError;
use fidget_bytecode::{Bytecode, BytecodeOp};
use fidget_core::{
    context::Tree,
    eval::Function,
    var::{Var, VarMap},
    vm::VmShape,
};
use std::sync::Arc;

/// A set of compiled kernels, used to build a [`RenderShape`]
///
/// See the [module-level documentation](self) for semantics.
///
/// [`RenderShape`]: crate::RenderShape
#[derive(Clone, Default)]
pub struct Kernels {
    kernels: Vec<(Var, Tree)>,
}

impl Kernels {
    /// Builds an empty kernel set
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `tree` to be compiled, returning a placeholder tree
    ///
    /// The placeholder should be used in place of `tree` when building the
    /// enclosing shape, which is then passed (along with this kernel set) to
    /// [`RenderShape::with_kernels`](crate::RenderShape::with_kernels).
    pub fn compile(&mut self, tree: Tree) -> Tree {
        let v = Var::new();
        self.kernels.push((v, tree));
        Tree::from(v)
    }

    /// Returns the number of kernels in the set
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Checks whether the set is empty
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Iterates over `(placeholder, tree)` pairs
    pub fn iter(&self) -> impl Iterator<Item = (Var, &Tree)> {
        self.kernels.iter().map(|(v, t)| (*v, t))
    }

    /// Compiles the choice-free leaves of a tree, returning the new tree
    ///
    /// This is a simple policy for splitting a shape into interpreted and
    /// compiled parts: CSG structure (`min`, `max`, `and`, `or`) stays in the
    /// interpreted tape, where it benefits from per-tile tape simplification,
    /// while each maximal subexpression without those operations (e.g. a
    /// primitive, transform chain, or noise function) with at least
    /// `min_ops` operations becomes a kernel.  Identical subexpressions share
    /// a kernel.
    ///
    /// Smaller choice-free subexpressions are left interpreted, since calling
    /// a kernel isn't free: it's evaluated in full whenever the tape reaches
    /// it.
    pub fn compile_leaves(&mut self, tree: &Tree, min_ops: usize) -> Tree {
        use fidget_core::context::{BinaryOpcode, Context, Node, Op};
        use std::collections::HashMap;

        let mut ctx = Context::new();
        let root = ctx.import(tree);

        // Post-order traversal (iterative, since trees may be deep)
        let mut order = vec![];
        let mut seen = std::collections::HashSet::new();
        let mut todo = vec![(root, false)];
        while let Some((n, expanded)) = todo.pop() {
            if expanded {
                order.push(n);
                continue;
            }
            if !seen.insert(n) {
                continue;
            }
            todo.push((n, true));
            match ctx.get_op(n).unwrap() {
                Op::Binary(_, a, b) => {
                    todo.push((*b, false));
                    todo.push((*a, false));
                }
                Op::Unary(_, a) => todo.push((*a, false)),
                Op::Input(..) | Op::Const(..) => (),
            }
        }

        // Per-node properties: compilable (choice-free, and only using
        // axes), size (with shared nodes counted once per use, which is fine
        // for a heuristic), and whether it uses any axis
        let mut info: HashMap<Node, (bool, usize, bool)> = HashMap::new();
        for &n in &order {
            let i = match *ctx.get_op(n).unwrap() {
                Op::Input(v) => {
                    let axis = matches!(v, Var::X | Var::Y | Var::Z);
                    (axis, 1, axis)
                }
                Op::Const(..) => (true, 0, false),
                Op::Unary(_, a) => {
                    let (c, s, x) = info[&a];
                    (c, s + 1, x)
                }
                Op::Binary(op, a, b) => {
                    let (ca, sa, xa) = info[&a];
                    let (cb, sb, xb) = info[&b];
                    let choice = matches!(
                        op,
                        BinaryOpcode::Min
                            | BinaryOpcode::Max
                            | BinaryOpcode::And
                            | BinaryOpcode::Or
                    );
                    (ca && cb && !choice, sa + sb + 1, xa || xb)
                }
            };
            info.insert(n, i);
        }
        let big = |n: Node| {
            let (c, s, x) = info[&n];
            c && x && s >= min_ops
        };

        // A kernel root is a large compilable node which is the root, or has
        // a parent which isn't itself compiled
        let mut roots: std::collections::HashSet<Node> = Default::default();
        if big(root) {
            roots.insert(root);
        }
        for &n in &order {
            if big(n) {
                continue;
            }
            match *ctx.get_op(n).unwrap() {
                Op::Binary(_, a, b) => {
                    roots.extend([a, b].into_iter().filter(|c| big(*c)));
                }
                Op::Unary(_, a) if big(a) => {
                    roots.insert(a);
                }
                _ => (),
            }
        }

        let mut placeholders: HashMap<Node, Tree> = HashMap::new();
        let mut rebuilt: HashMap<Node, Node> = HashMap::new();
        let mut out = Context::new();
        for &n in &order {
            let new = if roots.contains(&n) {
                let p = placeholders
                    .entry(n)
                    .or_insert_with(|| self.compile(ctx.export(n).unwrap()));
                out.import(p)
            } else {
                rebuild_node(&mut out, ctx.get_op(n).unwrap(), &rebuilt)
            };
            rebuilt.insert(n, new);
        }
        out.export(rebuilt[&root]).unwrap()
    }

    /// Converts each kernel's tree into a shape
    pub(crate) fn shapes(&self) -> Vec<(Var, VmShape)> {
        self.kernels
            .iter()
            .map(|(v, t)| (*v, VmShape::from(t.clone())))
            .collect()
    }
}

/// Rebuilds a single node in a new context, given mapped children
fn rebuild_node(
    out: &mut fidget_core::context::Context,
    op: &fidget_core::context::Op,
    map: &std::collections::HashMap<
        fidget_core::context::Node,
        fidget_core::context::Node,
    >,
) -> fidget_core::context::Node {
    use fidget_core::context::{BinaryOpcode as B, Op, UnaryOpcode as U};
    let r = match *op {
        Op::Input(v) => Ok(out.var(v)),
        Op::Const(c) => Ok(out.constant(c.0)),
        Op::Unary(op, a) => {
            let a = map[&a];
            match op {
                U::Neg => out.neg(a),
                U::Abs => out.abs(a),
                U::Recip => out.recip(a),
                U::Sqrt => out.sqrt(a),
                U::Square => out.square(a),
                U::Floor => out.floor(a),
                U::Ceil => out.ceil(a),
                U::Round => out.round(a),
                U::Sin => out.sin(a),
                U::Cos => out.cos(a),
                U::Tan => out.tan(a),
                U::Asin => out.asin(a),
                U::Acos => out.acos(a),
                U::Atan => out.atan(a),
                U::Exp => out.exp(a),
                U::Ln => out.ln(a),
                U::Not => out.not(a),
                U::Rand => out.rand(a),
            }
        }
        Op::Binary(op, a, b) => {
            let (a, b) = (map[&a], map[&b]);
            match op {
                B::Add => out.add(a, b),
                B::Sub => out.sub(a, b),
                B::Mul => out.mul(a, b),
                B::Div => out.div(a, b),
                B::Atan => out.atan2(a, b),
                B::Min => out.min(a, b),
                B::Max => out.max(a, b),
                B::Compare => out.compare(a, b),
                B::Mod => out.modulo(a, b),
                B::And => out.and(a, b),
                B::Or => out.or(a, b),
                B::Mix => out.mix(a, b),
            }
        }
    };
    r.expect("nodes are valid in the output context")
}

/// Error when compiling a kernel
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    /// Kernels may only use the `X`, `Y`, and `Z` axes
    #[error("kernels may only use the X, Y, and Z axes (found {0})")]
    UnsupportedVar(Var),
    /// The kernel's bytecode contains an unknown opcode
    #[error("unknown opcode {0}")]
    UnknownOpcode(u8),
    /// The kernel's bytecode is malformed
    #[error("malformed bytecode: {0}")]
    Malformed(&'static str),
    /// The kernel's tape uses the reserved register
    #[error(transparent)]
    RegisterError(#[from] fidget_bytecode::ReservedRegister),
}

/// Compiled kernel code for a particular [`RenderShape`](crate::RenderShape)
///
/// The WGSL source defines `kernel_input`, which the tape interpreter calls
/// for input variables which aren't axes.  Pipelines are cached by `key`, a
/// hash of the source.
pub(crate) struct KernelProgram {
    pub key: u64,
    pub source: Arc<str>,
    /// Indices (in the shape's variable map) of kernel placeholders
    pub inputs: Vec<u32>,
}

/// Default `kernel_input` for shaders without kernels
pub(crate) const KERNEL_STUB: &str = "
fn kernel_input(
    index: u32, xyz: array<Value, 3>, out: ptr<function, Value>,
) -> bool {
    return false;
}
";

/// Returns the `kernel_input` source for a (possibly absent) program
pub(crate) fn kernel_source(program: Option<&KernelProgram>) -> &str {
    program.map(|p| &*p.source).unwrap_or(KERNEL_STUB)
}

impl KernelProgram {
    /// Builds kernel code for a shape with the given variable map
    ///
    /// Kernels whose placeholders don't appear in `vars` are skipped.  Returns
    /// `None` if no kernels are used.
    pub(crate) fn build(
        vars: &VarMap,
        kernels: &[(Var, VmShape)],
    ) -> Result<Option<Self>, RenderShapeError> {
        let mut source = String::new();
        let mut cases = String::new();
        let mut inputs = vec![];
        for (i, (v, shape)) in kernels.iter().enumerate() {
            let Some(index) = vars.get(v) else {
                continue;
            };
            let bytecode = Bytecode::new(shape.inner().data())?;
            let name = format!("kernel_{i}");
            source +=
                &compile_function(&bytecode, shape.inner().vars(), &name)?;
            cases += &format!(
                "        case {index}u: {{ *out = {name}(xyz); return true; }}\n"
            );
            inputs.push(index as u32);
        }
        if inputs.is_empty() {
            return Ok(None);
        }
        source += &format!(
            "
fn kernel_input(
    index: u32, xyz: array<Value, 3>, out: ptr<function, Value>,
) -> bool {{
    switch index {{
{cases}        default: {{ return false; }}
    }}
}}
"
        );
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            h.finish()
        };
        Ok(Some(Self {
            key,
            source: source.into(),
            inputs,
        }))
    }
}

/// Compiles a bytecode tape into a WGSL function
///
/// The function has the signature `fn name(xyz: array<Value, 3>) -> Value`.
/// It's written in terms of the operation functions used by the tape
/// interpreter (`op_add`, `build_imm`, etc.), so the same source can be used
/// with any `Value` type (float, interval, or gradient).  It declares a local
/// `Stack` for `min` / `max` choices, which are discarded.
///
/// `vars` is the variable map of the tape; only the `X`, `Y`, and `Z` axes
/// are allowed.
///
/// The function is straight-line code, with one `let` binding per operation;
/// register spills (`Load` / `Store`) are resolved at compile time.
pub fn compile_function(
    bytecode: &Bytecode,
    vars: &VarMap,
    name: &str,
) -> Result<String, KernelError> {
    // Map from input index to axis
    let mut axes = vec![];
    for (v, i) in vars.iter() {
        let axis = match v {
            Var::X => 0,
            Var::Y => 1,
            Var::Z => 2,
            v => return Err(KernelError::UnsupportedVar(v)),
        };
        axes.push((i as u32, axis));
    }

    let mut body = String::new();
    // Name of the SSA value currently held in each register / memory slot
    let mut regs: Vec<Option<String>> = vec![None; 256];
    let mut mem: std::collections::HashMap<u32, String> = Default::default();
    let mut out = None;
    let mut n = 0usize;

    let data = bytecode.data();
    let (words, rest) = data.as_chunks::<2>();
    if !rest.is_empty() {
        return Err(KernelError::Malformed("odd number of words"));
    }
    for &[word, imm] in words {
        let [opcode, dst, lhs, rhs] = word.to_le_bytes();
        if opcode == 0xFF {
            // Jumps: only the start and end markers are expected
            match imm {
                0 => continue,
                u32::MAX => break,
                _ => return Err(KernelError::Malformed("unexpected jump")),
            }
        }
        let op = BytecodeOp::from_repr(opcode)
            .ok_or(KernelError::UnknownOpcode(opcode))?;
        let arg = |r: u8| -> Result<String, KernelError> {
            if r == u8::MAX {
                Ok(format!("build_imm(bitcast<f32>({imm:#010x}u))"))
            } else {
                regs[r as usize]
                    .clone()
                    .ok_or(KernelError::Malformed("read of unset register"))
            }
        };
        let expr = match op {
            BytecodeOp::Output => {
                out = Some(arg(dst)?);
                continue;
            }
            BytecodeOp::Mem => {
                if dst == u8::MAX {
                    // Store: mem[imm] = reg[lhs]
                    mem.insert(imm, arg(lhs)?);
                } else {
                    // Load: reg[dst] = mem[imm]
                    let v = mem
                        .get(&imm)
                        .cloned()
                        .ok_or(KernelError::Malformed("load of unset slot"))?;
                    regs[dst as usize] = Some(v);
                }
                continue;
            }
            BytecodeOp::Copy => {
                // Register copies just rename the value; immediates get a
                // binding, so that the constant is only built once
                if lhs == u8::MAX {
                    arg(lhs)?
                } else {
                    regs[dst as usize] = Some(arg(lhs)?);
                    continue;
                }
            }
            BytecodeOp::Input => {
                let axis = axes
                    .iter()
                    .find(|(i, _)| *i == imm)
                    .map(|(_, a)| *a)
                    .ok_or(KernelError::Malformed("unknown input"))?;
                format!("xyz[{axis}]")
            }
            BytecodeOp::Neg
            | BytecodeOp::Abs
            | BytecodeOp::Recip
            | BytecodeOp::Sqrt
            | BytecodeOp::Square
            | BytecodeOp::Floor
            | BytecodeOp::Ceil
            | BytecodeOp::Round
            | BytecodeOp::Not
            | BytecodeOp::Rand
            | BytecodeOp::Sin
            | BytecodeOp::Cos
            | BytecodeOp::Tan
            | BytecodeOp::Asin
            | BytecodeOp::Acos
            | BytecodeOp::Atan
            | BytecodeOp::Exp
            | BytecodeOp::Ln => {
                format!("{}({})", op_function(op), arg(lhs)?)
            }
            BytecodeOp::Add
            | BytecodeOp::Sub
            | BytecodeOp::Mul
            | BytecodeOp::Div
            | BytecodeOp::Atan2
            | BytecodeOp::Compare
            | BytecodeOp::Mix
            | BytecodeOp::Mod => {
                format!("{}({}, {})", op_function(op), arg(lhs)?, arg(rhs)?)
            }
            BytecodeOp::Min
            | BytecodeOp::Max
            | BytecodeOp::And
            | BytecodeOp::Or => {
                format!(
                    "{}({}, {}, &stack)",
                    op_function(op),
                    arg(lhs)?,
                    arg(rhs)?
                )
            }
        };
        let v = format!("v{n}");
        n += 1;
        body += &format!("    let {v} = {expr};\n");
        regs[dst as usize] = Some(v);
    }
    let out = out.ok_or(KernelError::Malformed("no output"))?;
    Ok(format!(
        "fn {name}(xyz: array<Value, 3>) -> Value {{\n    \
            var stack = Stack();\n{body}    return {out};\n}}\n"
    ))
}

/// Returns the name of the WGSL function implementing an opcode
///
/// These match the function names used by `tape_interpreter.wgsl`.
fn op_function(op: BytecodeOp) -> &'static str {
    match op {
        BytecodeOp::Neg => "op_neg",
        BytecodeOp::Abs => "op_abs",
        BytecodeOp::Recip => "op_recip",
        BytecodeOp::Sqrt => "op_sqrt",
        BytecodeOp::Square => "op_square",
        BytecodeOp::Floor => "op_floor",
        BytecodeOp::Ceil => "op_ceil",
        BytecodeOp::Round => "op_round",
        BytecodeOp::Not => "op_not",
        BytecodeOp::Rand => "op_rand",
        BytecodeOp::Sin => "op_sin",
        BytecodeOp::Cos => "op_cos",
        BytecodeOp::Tan => "op_tan",
        BytecodeOp::Asin => "op_asin",
        BytecodeOp::Acos => "op_acos",
        BytecodeOp::Atan => "op_atan",
        BytecodeOp::Exp => "op_exp",
        BytecodeOp::Ln => "op_log",
        BytecodeOp::Add => "op_add",
        BytecodeOp::Sub => "op_sub",
        BytecodeOp::Mul => "op_mul",
        BytecodeOp::Div => "op_div",
        BytecodeOp::Atan2 => "op_atan2",
        BytecodeOp::Compare => "op_compare",
        BytecodeOp::Mix => "op_mix",
        BytecodeOp::Mod => "op_mod",
        BytecodeOp::Min => "op_min",
        BytecodeOp::Max => "op_max",
        BytecodeOp::And => "op_and",
        BytecodeOp::Or => "op_or",
        BytecodeOp::Output
        | BytecodeOp::Input
        | BytecodeOp::Copy
        | BytecodeOp::Mem => unreachable!("not a math operation"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Gpu, RenderShape};
    use fidget_core::context::Context;

    /// Models from the Fidget repository
    const MODELS: &[(&str, &str)] = &[
        ("hi", include_str!("../../models/hi.vm")),
        ("quarter", include_str!("../../models/quarter.vm")),
        ("tanglecube", include_str!("../../models/tanglecube.vm")),
        ("bear", include_str!("../../models/bear.vm")),
        ("colonnade", include_str!("../../models/colonnade.vm")),
        ("prospero", include_str!("../../models/prospero.vm")),
    ];

    fn load(text: &str) -> Tree {
        let (ctx, root) = Context::from_text(text.as_bytes()).unwrap();
        ctx.export(root).unwrap()
    }

    /// Builds a tree which uses every opcode
    fn every_op() -> Tree {
        let (x, y, z) = Tree::axes();
        let a = x.clone().sin() + y.clone().cos() * z.clone().tan();
        let b = (x.clone().square() + 1.0).sqrt().recip() - y.clone().abs();
        let c = x.clone().asin().atan2(y.clone().acos()) / z.clone().atan();
        let d = (x.clone().exp() + 2.0).ln().floor() + y.clone().ceil()
            - z.clone().round();
        let e = x.clone().modulo(0.5).compare(y.clone())
            + x.clone().mix(y.clone())
            + z.clone().rand()
            + x.clone().not();
        let f = a.min(b).max(c).and(d).or(e);
        -f
    }

    /// Returns the sources of every stage which evaluates tapes, including
    /// the given kernel source
    fn stage_shaders(k: &str) -> Vec<(&'static str, String)> {
        vec![
            ("pixel root", crate::pixel::interval_root_shader(64, k)),
            ("pixel tiles", crate::pixel::interval_tiles_shader(64, k)),
            ("pixel", crate::pixel::pixel_tiles_shader(64, k)),
            ("voxel root", crate::voxel::interval_root_shader(64, k)),
            ("voxel tiles", crate::voxel::interval_tiles_shader(64, k)),
            ("voxel", crate::voxel::voxel_tiles_shader(64, k)),
            ("normals", crate::voxel::normals_shader(64, k)),
        ]
    }

    fn check_stages(shape: &RenderShape) {
        let program = shape.kernels().expect("shape should have kernels");
        for (name, src) in stage_shaders(&program.source) {
            crate::compile_shader(&src, name);
        }
    }

    #[test]
    fn stub_shaders_compile() {
        for (name, src) in stage_shaders(KERNEL_STUB) {
            crate::compile_shader(&src, name);
        }
    }

    #[test]
    fn compile_every_op() {
        let shape = VmShape::from(every_op());
        let rs = RenderShape::compiled(&shape).unwrap();
        let src = &rs.kernels().unwrap().source;
        for (_, i) in fidget_bytecode::iter_ops() {
            let op = BytecodeOp::from_repr(i).unwrap();
            if matches!(
                op,
                BytecodeOp::Output
                    | BytecodeOp::Input
                    | BytecodeOp::Copy
                    | BytecodeOp::Mem
            ) {
                continue;
            }
            assert!(
                src.contains(op_function(op)),
                "kernel is missing {}",
                op_function(op)
            );
        }
        check_stages(&rs);
    }

    #[test]
    fn compile_spills() {
        // Many simultaneously-live values force register spills
        let (x, y, _z) = Tree::axes();
        let terms: Vec<Tree> = (0..400)
            .map(|i| (x.clone() * (i as f32 * 0.01)).sin() + y.clone())
            .collect();
        let mut t = terms[0].clone();
        for (i, v) in terms.iter().enumerate().skip(1) {
            t = if i % 2 == 0 {
                t.max(v.clone())
            } else {
                t.min(v.clone())
            };
        }
        for v in terms.iter().rev() {
            t += v.clone();
        }
        let shape = VmShape::from(t);
        let bytecode = Bytecode::new(shape.inner().data()).unwrap();
        assert!(bytecode.mem_count() > 0, "test tape should spill");
        let rs = RenderShape::compiled(&shape).unwrap();
        check_stages(&rs);
    }

    #[test]
    fn unsupported_var() {
        let mut kernels = Kernels::new();
        let v = Var::new();
        let k = kernels.compile(Tree::x() + Tree::from(v));
        let shape = VmShape::from(k.min(Tree::y()));
        assert!(matches!(
            RenderShape::with_kernels(&shape, &kernels),
            Err(RenderShapeError::KernelError(KernelError::UnsupportedVar(
                ..
            )))
        ));
    }

    #[test]
    fn unused_kernels() {
        let mut kernels = Kernels::new();
        let _unused = kernels.compile(Tree::x().sin());
        let shape = VmShape::from(Tree::y());
        let rs = RenderShape::with_kernels(&shape, &kernels).unwrap();
        assert!(!rs.has_kernels());
    }

    #[test]
    fn identical_kernels_share_source() {
        let t = Tree::x().sin() * Tree::y().cos() - 0.5;
        let a = RenderShape::compiled(&VmShape::from(t.clone())).unwrap();
        let b = RenderShape::compiled(&VmShape::from(t)).unwrap();
        assert_eq!(a.kernels().unwrap().key, b.kernels().unwrap().key);
    }

    #[test]
    fn models_compile() {
        for (name, text) in MODELS {
            let tree = load(text);
            let rs = RenderShape::compiled(&VmShape::from(tree.clone()))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            check_stages(&rs);

            let mut kernels = Kernels::new();
            let outer = kernels.compile_leaves(&tree, 4);
            let rs = RenderShape::with_kernels(&VmShape::from(outer), &kernels)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            if rs.has_kernels() {
                check_stages(&rs);
            }
        }
    }

    /// Splitting into kernels must not change the function
    #[test]
    fn compile_leaves_is_exact() {
        for (name, text) in MODELS {
            let tree = load(text);
            let mut kernels = Kernels::new();
            let outer = kernels.compile_leaves(&tree, 4);

            let mut ctx = Context::new();
            let orig = ctx.import(&tree);
            let outer = ctx.import(&outer);
            let ks: Vec<_> =
                kernels.iter().map(|(v, t)| (v, ctx.import(t))).collect();
            for i in 0..64 {
                let f = |k: u32| {
                    ((i * 7919 + k * 104729) % 1000) as f32 / 500.0 - 1.0
                };
                let (x, y, z) = (f(1), f(2), f(3));
                let mut vars: std::collections::HashMap<Var, f32> =
                    [(Var::X, x), (Var::Y, y), (Var::Z, z)].into();
                for (v, n) in &ks {
                    vars.insert(*v, ctx.eval(*n, &vars.clone()).unwrap());
                }
                let a = ctx.eval(orig, &vars).unwrap();
                let b = ctx.eval(outer, &vars).unwrap();
                assert!(
                    a == b || (a.is_nan() && b.is_nan()),
                    "{name}: mismatch at ({x}, {y}, {z}): {a} != {b}"
                );
            }
            if *name == "prospero" || *name == "colonnade" {
                assert!(!kernels.is_empty(), "{name} should have kernels");
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // GPU tests: compiled and interpreted renders must agree

    fn skip_gpu_test() -> bool {
        // As elsewhere, only run on CI if we're on MacOS (other runners don't
        // have GPUs)
        cfg!(not(target_os = "macos")) && std::env::var("CI").is_ok()
    }

    /// Builds a model in each mode: interpreted, fully compiled, and with
    /// compiled leaves
    ///
    /// The fully compiled version of `prospero` (a ~7800-operation kernel)
    /// is skipped, because the driver takes minutes to compile its pipelines.
    fn variants(name: &str, tree: &Tree) -> Vec<(&'static str, RenderShape)> {
        let shape = VmShape::from(tree.clone());
        let mut kernels = Kernels::new();
        let outer = kernels.compile_leaves(tree, 4);
        let mut out = vec![];
        // The WGSL interpreter doesn't implement register spills (`Mem`
        // operations), so it can't render large tapes
        let bytecode = Bytecode::new(shape.inner().data()).unwrap();
        if bytecode.mem_count() == 0 {
            out.push(("interpreted", RenderShape::new(&shape).unwrap()));
        }
        if name != "prospero" {
            out.push(("compiled", RenderShape::compiled(&shape).unwrap()));
        }
        out.push((
            "leaves",
            RenderShape::with_kernels(&VmShape::from(outer), &kernels).unwrap(),
        ));
        out
    }

    /// Image size for GPU tests
    ///
    /// `prospero` is rendered at a smaller size because larger interpreted
    /// renders currently hang on some GPUs (seen with NVIDIA / Vulkan), even
    /// without kernels.
    fn test_size(name: &str) -> u32 {
        if name == "prospero" { 192 } else { 256 }
    }

    /// Maximum difference between two floats in units of relative error
    fn close(a: f32, b: f32) -> bool {
        a == b
            || (a.is_nan() && b.is_nan())
            || (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn pixel_models_match() {
        if skip_gpu_test() {
            return;
        }
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let ctx = crate::pixel::Context::new(&gpu);
        let mut ws = ctx.workspace();
        for (name, text) in MODELS {
            let tree = load(text);
            for pixel_perfect in [false, true] {
                let cfg = fidget_raster::pixel::RenderConfig {
                    pixel_perfect,
                    ..fidget_raster::pixel::RenderConfig::from_size(
                        fidget_raster::pixel::RenderSize::new(
                            test_size(name),
                            test_size(name),
                        ),
                    )
                };
                let reference = cfg.run(
                    fidget_core::shape::BoundShape::try_from(VmShape::from(
                        tree.clone(),
                    ))
                    .unwrap(),
                );
                let mut images = vec![];
                for (mode, shape) in variants(name, &tree) {
                    let t = std::time::Instant::now();
                    let mut out = gpu.read_buffer_for(ws.output());
                    let im = ctx.run(&shape, &mut ws, &mut out, cfg).unwrap();
                    if std::env::var("KDEBUG").is_ok() {
                        eprintln!("{name} {mode}: {:?}", t.elapsed());
                    }
                    images.push((mode, im));
                }
                for (mode, im) in &images {
                    // Inside / outside must agree everywhere (except for
                    // point samples within rounding error of the surface),
                    // and point samples must agree in value.  One render may
                    // prove a tile empty or full (writing a fill) where the
                    // other evaluates individual pixels: both are correct.
                    let mut mismatched = 0;
                    for (a, b) in reference.iter().zip(im.iter()) {
                        let on_surface = [a, b].iter().any(|p| {
                            p.distance().is_some_and(|d| d.abs() < 1e-5)
                        });
                        let ok = (a.inside() == b.inside() || on_surface)
                            && match (a.distance(), b.distance()) {
                                (Some(a), Some(b)) => close(a, b),
                                _ => true,
                            };
                        if !ok
                            && mismatched < 5
                            && std::env::var("KDEBUG").is_ok()
                        {
                            eprintln!(
                                "{name} {mode}: {:?} vs {:?}",
                                a.unpack(),
                                b.unpack()
                            );
                        }
                        mismatched += usize::from(!ok);
                    }
                    assert_eq!(
                        mismatched, 0,
                        "{name} ({mode}, pixel_perfect = {pixel_perfect}): \
                         {mismatched} pixels differ from the CPU renderer"
                    );
                }
            }
        }
    }

    #[test]
    fn voxel_models_match() {
        if skip_gpu_test() {
            return;
        }
        let gpu = pollster::block_on(Gpu::init_basic()).unwrap();
        let ctx = crate::voxel::Context::new(&gpu);
        let mut ws = ctx.workspace();
        for (name, text) in MODELS {
            let tree = load(text);
            let cfg = fidget_raster::voxel::RenderConfig::from_size(
                fidget_raster::voxel::RenderSize::new(
                    test_size(name) / 2,
                    test_size(name) / 2,
                    test_size(name) / 2,
                ),
            );
            let mut images = vec![];
            for (mode, shape) in variants(name, &tree) {
                let mut out = gpu.read_buffer_for(ws.output());
                let im = ctx.run(&shape, &mut ws, &mut out, cfg).unwrap();
                images.push((mode, im));
            }
            // The CPU and GPU voxel renderers use different conventions (e.g.
            // for surfaces clipped by the top of the view volume), so the
            // GPU interpreter is the reference here.  Models which it can't
            // render (because their tapes spill registers) are covered by
            // `pixel_models_match` instead.
            let Some(((_, reference), rest)) = images
                .split_first()
                .filter(|((m, _), _)| *m == "interpreted")
            else {
                continue;
            };
            for (mode, im) in rest {
                // Depths must match exactly.  Normals must match exactly in
                // the fully compiled mode (which evaluates the same tape).
                //
                // With compiled leaves, normals may differ at creases where a
                // `min` / `max` is exactly tied: splitting a tree rebuilds
                // it, which may swap the operands of commutative operations
                // and so pick the other (equally valid) branch's gradient.
                // Such points should be rare.
                let mut depth_mismatch = 0;
                let mut normal_mismatch = 0;
                let mut filled = 0;
                for (a, b) in reference.iter().zip(im.iter()) {
                    if a.depth != b.depth {
                        depth_mismatch += 1;
                        if std::env::var("KDEBUG").is_ok() {
                            eprintln!("{name} {mode}: {a:?} vs {b:?}");
                        }
                        continue;
                    } else if a.depth == 0 {
                        continue;
                    }
                    filled += 1;
                    let same = a
                        .normal
                        .iter()
                        .zip(&b.normal)
                        .all(|(a, b)| close(*a, *b));
                    normal_mismatch += usize::from(!same);
                }
                if std::env::var("KDEBUG").is_ok() {
                    eprintln!(
                        "{name} {mode}: {normal_mismatch} / {filled} normals \
                         differ"
                    );
                }
                assert_eq!(
                    depth_mismatch, 0,
                    "{name} ({mode}): {depth_mismatch} depths differ from \
                     the interpreter"
                );
                let allowed = if *mode == "compiled" { 0 } else { filled / 20 };
                assert!(
                    normal_mismatch <= allowed,
                    "{name} ({mode}): {normal_mismatch} / {filled} normals \
                     differ from the interpreter"
                );
            }
        }
    }
}
