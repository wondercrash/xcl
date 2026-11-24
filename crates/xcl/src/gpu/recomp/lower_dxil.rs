//! The direct lowering (`lower`) written into a DXIL module through `gpu::dxil`'s builder:
//! the per-lane form of a program as a signed container, with no HLSL and no DXC.
//!
//! `Ir` maps one to one onto the builder: blocks, phis (`phi_at` / `add_incoming`), integer
//! ops, compares, `select`, the thread / group / lane ids, `waveActiveBallot`, and raw buffer
//! loads and stores on the ABI's root UAVs (u0 `io`, u1 `res`, u2 `mem`). The builder state is
//! #1426's translator state (`gpu::dxil::lower::Lw`, built by `Lw::new` and finished by
//! `Lw::finish`), so its per-entry setup (`setup_lanes_compute`) and its instruction
//! lowering are reachable from here. A phi the SSA
//! construction finds trivial has its uses replaced (`Module::replace_uses`); the phi itself
//! stays in its block, unused. The replacements are collected and made in one pass in
//! `DxilIr::finish`: one pass over the module per trivial phi made the lowering quadratic in
//! the program's size, and nothing reads an instruction's operands back before then (the
//! SSA construction resolves a replaced phi itself).

use super::lower::{self, Bin, Cmp, Ir};
use super::ssa::{Sink, Var};
use crate::gpu::dxil::kernel::{Entry, KernelOptions};
use crate::gpu::dxil::lower::Lw;
use crate::gpu::dxil::module::{BinOp, Block, CastOp, Pred, Value};
use crate::gpu::dxil::shader::RawUav;
use crate::gpu::rdna2::Decoded;
use std::collections::{BTreeSet, HashMap};

struct DxilIr {
    lw: Lw,
    io: RawUav,
    res: Option<RawUav>,
    mem: RawUav,
    /// each `Ir` block: where it starts (its phis) and where it ends now (its branch; later than
    /// the start when #1426's lowering of an instruction in it made blocks of its own)
    blocks: Vec<Block>,
    tails: Vec<Block>,
    /// the `Ir` block being written
    cur: usize,
    /// `compute_ids` set up the lanes (`Lw::lane` is the console lane)
    lanes_set: bool,
    /// a split compute program's thread ids, set up with its lanes in the constructor: the
    /// lowering takes `lane()` before `compute_ids`, and in the split form the console lane
    /// is not the hardware lane (the upper half's are 32-63)
    ids: Option<(Value, [Value; 3])>,
    /// trivial phis → what replaces their uses, made in the module by `finish`
    replaced: HashMap<Value, Value>,
}

impl DxilIr {
    /// The builder state for `insts` at `entry` (a kernel or a compute program), with one memory
    /// window; a compute program's is coherent, as the HLSL path declares it.
    fn new(insts: &[Decoded], wave64: bool, entry: &Entry) -> DxilIr {
        Self::with(insts, KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 }, entry)
    }

    /// `new` in the split form (two hardware wave32s per console wave64): #1426's `Lw` numbers
    /// the threads as the hardware waves arrive and lays out the halves' exchange storage; its
    /// `ballot` / `read_lane_uniform` meet the other half (a group barrier for one console
    /// wave per group, the spin meet for more).
    fn split(insts: &[Decoded], entry: &Entry) -> DxilIr {
        Self::with(insts, KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 }, entry)
    }

    /// The builder state with the runtime's options `o` (memory windows, sampler registers, the
    /// split form).
    fn with(insts: &[Decoded], o: KernelOptions, entry: &Entry) -> DxilIr {
        // block leaders for the constant-sampler scan: the first instruction, branch targets
        // and the instruction after a branch
        let mut leaders: BTreeSet<u64> = insts.first().map(|d| d.pc).into_iter().collect();
        for (i, d) in insts.iter().enumerate() {
            if let Some(t) = crate::gpu::rdna2::branch_target(d) {
                leaders.insert(t);
                leaders.extend(insts.get(i + 1).map(|n| n.pc));
            }
        }
        let mut lw = Lw::new(insts, &leaders, o, entry);
        // the rounding modes the program sets, as `kernel::translate_on_this_thread` finds them:
        // #1426's `s_round_mode` / `s_setreg` refuse a mode they cannot honour from these
        let branchy = insts.iter().any(|d| crate::gpu::rdna2::branch_target(d).is_some());
        (lw.dyn_rm, lw.has_f64_round_op) = crate::gpu::dxil::lower::round_modes(insts, !branchy);
        // a kernel's lanes in the split form: the console lane each hardware lane stands for,
        // set up in the entry block (a compute program's come with `compute_ids`)
        let mut ids = None;
        let lanes_set = match entry {
            // (a kernel always: `Lw`'s lane bit, `lane31`, is what #1426's lowering of a carry-in or
            // a `v_cndmask` reads, and stays 0 without it)
            Entry::Kernel => {
                lw.setup_lanes();
                true
            }
            Entry::Compute(cs) if lw.split => {
                let lds = lw.lds.is_some();
                ids = Some(lw.setup_lanes_compute(cs.threads, lds));
                true
            }
            _ => false,
        };
        let io = lw.init.io.expect("the I/O buffer");
        let (res, mem) = (lw.res, lw.mems[0]);
        DxilIr { lw, io, res, mem, blocks: Vec::new(), tails: Vec::new(), cur: 0, lanes_set, ids, replaced: HashMap::new() }
    }

    fn ty(&mut self, var: Var) -> crate::gpu::dxil::module::Ty {
        match var {
            Var::M(_) | Var::Scc => self.lw.s.m.i1(),
            Var::S(_) | Var::V(_) => self.lw.s.m.i32(),
        }
    }

    /// The container: `Lw`'s entry block jumps to the first `Ir` block (as `kernel.rs` ends it),
    /// the lowering's blocks are sealed for `Lw`'s variables (every edge into them is known once
    /// the lowering is done), then `Lw::finish`.
    fn finish(mut self, numthreads: [u32; 3], hw: u32) -> Result<crate::gpu::dxil::kernel::Kernel, String> {
        // the trivial phis' uses first, as `replace` would have left them
        self.lw.s.m.replace_uses(&self.replaced);
        let first = *self.blocks.first().ok_or("no blocks")?;
        self.lw.s.m.set_block(Block(0));
        self.lw.br(first);
        for i in 0..self.blocks.len() {
            let b = self.blocks[i];
            if !self.lw.ssa.is_sealed(b) {
                self.lw.seal(b);
            }
        }
        self.lw.finish(numthreads, hw)
    }

    fn bytes(&mut self, dword: Value) -> Value {
        let four = self.lw.s.m.i32c(4);
        self.lw.s.m.bin(BinOp::Mul, dword, four)
    }
}

impl Sink for DxilIr {
    type V = Value;
    fn phi(&mut self, block: usize, var: Var) -> Value {
        let t = self.ty(var);
        self.lw.s.m.phi_at(self.blocks[block], t)
    }
    fn incoming(&mut self, phi: Value, value: Value, pred: usize) {
        self.lw.s.m.add_incoming(phi, value, self.tails[pred]);
    }
    fn replace(&mut self, old: Value, new: Value) {
        // every use of the trivial phi becomes `new` (in `finish`); the phi stays in its
        // block, unused
        self.replaced.insert(old, new);
    }
    fn undef(&mut self, var: Var) -> Value {
        let t = self.ty(var);
        self.lw.s.m.undef(t)
    }
}

impl Ir for DxilIr {
    fn new_block(&mut self) -> usize {
        // not `Block(0)`: `Lw` keeps writing there (static samplers, scratch) until `finish`
        let b = self.lw.s.m.new_block();
        self.blocks.push(b);
        self.tails.push(b);
        self.blocks.len() - 1
    }
    fn at(&mut self, block: usize) {
        self.cur = block;
        self.lw.s.m.set_block(self.tails[block]);
    }
    fn int(&mut self, v: u32) -> Value {
        self.lw.s.m.i32c(v)
    }
    fn boolean(&mut self, v: bool) -> Value {
        self.lw.s.m.i1c(v)
    }
    fn bin(&mut self, op: Bin, a: Value, b: Value) -> Value {
        let op = match op {
            Bin::Add => BinOp::Add,
            Bin::Sub => BinOp::Sub,
            Bin::Mul => BinOp::Mul,
            Bin::And => BinOp::And,
            Bin::Or => BinOp::Or,
            Bin::Xor => BinOp::Xor,
            Bin::Shl => BinOp::Shl,
            Bin::LShr => BinOp::LShr,
            Bin::AShr => BinOp::AShr,
        };
        self.lw.s.m.bin(op, a, b)
    }
    fn icmp(&mut self, c: Cmp, a: Value, b: Value) -> Value {
        let p = match c {
            Cmp::Eq => Pred::Eq,
            Cmp::Ne => Pred::Ne,
            Cmp::Ult => Pred::ULt,
            Cmp::Ule => Pred::ULe,
            Cmp::Ugt => Pred::UGt,
            Cmp::Uge => Pred::UGe,
            Cmp::Slt => Pred::SLt,
            Cmp::Sle => Pred::SLe,
            Cmp::Sgt => Pred::SGt,
            Cmp::Sge => Pred::SGe,
        };
        self.lw.s.m.cmp(p, a, b)
    }
    fn select(&mut self, c: Value, a: Value, b: Value) -> Value {
        self.lw.s.m.select(c, a, b)
    }
    fn zext(&mut self, b: Value) -> Value {
        let t = self.lw.s.m.i32();
        self.lw.s.m.cast(CastOp::ZExt, b, t)
    }
    fn lane(&mut self) -> Value {
        if self.lanes_set {
            self.lw.lane
        } else {
            self.lw.s.wave_lane_index()
        }
    }
    fn group_id(&mut self, k: u32) -> Value {
        self.lw.s.group_id(k)
    }
    fn compute_ids(&mut self, threads: [u32; 3]) -> (Value, [Value; 3]) {
        // #1426's compute entry: 2-D / 3-D groups numbered as their waves arrive, the counter
        // in the program's top LDS word when it has LDS
        if let Some(ids) = self.ids {
            return ids;
        }
        let lds = self.lw.lds.is_some();
        let ids = self.lw.setup_lanes_compute(threads, lds);
        self.lanes_set = true;
        ids
    }
    fn whole_needs_exchange(&self) -> bool {
        self.lw.split
    }
    fn ballot(&mut self, c: Value) -> [Value; 2] {
        if self.lw.split {
            // this half's word from the hardware, the other's through the exchange
            let (lo, hi) = self.lw.ballot(c);
            return [lo, hi];
        }
        let w = self.lw.s.wave_ballot(c);
        [w[0], w[1]]
    }
    // one hardware wave per console wave (native wave64, or wave32): the half-local forms are
    fn mem_windows(&self) -> u32 {
        self.lw.mems.len() as u32
    }
    // the same operations; in the split form only the reader's half is right (`Ir::ballot_half`)
    fn ballot_half(&mut self, c: Value) -> [Value; 2] {
        if self.lw.split {
            let w = self.lw.s.wave_ballot(c);
            return [w[0], w[0]];
        }
        self.ballot(c)
    }
    /// #1426's semantics of `d` (`Lw::inst_guarded`) in the current block: the inputs are written
    /// into `Lw`'s SSA there first, so it reads them and nothing from other blocks (which its SSA
    /// doesn't know). A register it touches that is neither an input nor an output refuses the
    /// instruction: `Lw` would read that register's value at the program's start. Blocks it makes
    /// (a guarded access, the memory window switch) end the `Ir` block in the last of them.
    fn foreign(&mut self, d: &Decoded, guard: Value, inputs: &[(Var, Value)], outputs: &[Var]) -> Result<Vec<Value>, String> {
        use crate::gpu::dxil::lower::{VAR_BUDGET, VAR_SCC, VAR_VGPR};
        let var = |v: Var| match v {
            Var::S(r) => r as u32,
            Var::V(n) => VAR_VGPR + n as u32,
            Var::Scc => VAR_SCC,
            Var::M(_) => unreachable!("a lane-bit view is not a register"),
        };
        for &(v, x) in inputs {
            self.lw.wr(var(v), x);
        }
        // the registers `Lw` touches for `d` alone
        let before = std::mem::take(&mut self.lw.init.touched);
        self.lw.pc = d.pc;
        let r = self.lw.inst_guarded(d, guard, None);
        let used = std::mem::replace(&mut self.lw.init.touched, before);
        self.lw.init.touched.extend(&used);
        r?;
        let known: BTreeSet<u32> = inputs.iter().map(|&(v, _)| var(v)).chain(outputs.iter().map(|&v| var(v))).collect();
        if let Some(r) = used.iter().find(|&&r| r < VAR_BUDGET && !known.contains(&r)) {
            return Err(format!("it reads register {r} the per-lane form did not pass"));
        }
        let tail = self.lw.cur();
        self.tails[self.cur] = tail;
        Ok(outputs.iter().map(|&v| self.lw.rd(var(v))).collect())
    }
    fn read_lane(&mut self, v: Value, lane: Value) -> Value {
        if self.lw.split {
            // the lowering's lane reads take the same lane in every lane (readlane,
            // readfirstlane, writelane's source)
            return self.lw.read_lane_uniform(v, lane);
        }
        self.lw.s.wave_read_lane_at(v, lane)
    }
    fn read_lane_half(&mut self, v: Value, lane: Value) -> Value {
        if self.lw.split {
            return self.lw.read_lane_half(v, lane);
        }
        self.read_lane(v, lane)
    }
    fn count_bits(&mut self, v: Value) -> Value {
        self.lw.s.bit_count(crate::gpu::dxil::ops::BitCount::Countbits, v)
    }
    fn io_load(&mut self, dword: Value) -> Value {
        let a = self.bytes(dword);
        let io = self.io;
        self.lw.s.raw_load(io, a, 1)[0]
    }
    fn res_store(&mut self, dword: Value, value: Value) {
        let a = self.bytes(dword);
        let res = self.res.expect("res is bound in kernel mode");
        self.lw.s.raw_store(res, a, &[value]);
    }
    fn mem_load(&mut self, addr: Value) -> Value {
        let mem = self.mem;
        self.lw.s.raw_load(mem, addr, 1)[0]
    }
    fn mem_store(&mut self, addr: Value, value: Value) {
        let mem = self.mem;
        self.lw.s.raw_store(mem, addr, &[value]);
    }
    // through `Lw`, which records the edges for its own variables (the split form's exchange
    // count and dead flag, its temporaries); the lowering's blocks are sealed in `finish`
    fn br(&mut self, target: usize) {
        let b = self.blocks[target];
        self.lw.br(b);
    }
    fn br_if(&mut self, c: Value, then: usize, otherwise: usize) {
        let (t, f) = (self.blocks[then], self.blocks[otherwise]);
        self.lw.cond_br(c, t, f);
    }
    fn ret(&mut self) {
        self.lw.s.m.ret();
    }
}

/// A kernel-mode program in its per-lane form, as a signed DXIL container (one wave of
/// `WaveSize` = the program's wave size), and the masks whose final value may differ from
/// the console's (`lower::lower_kernel`).
pub fn kernel(insts: &[Decoded], wave64: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, wave64, DxilIr::new(insts, wave64, &Entry::Kernel))?;
    let lanes = if wave64 { 64 } else { 32 };
    Ok((ir.finish([lanes, 1, 1], lanes)?.dxil, stale))
}

/// `kernel` in the split form: a wave64 program as two hardware wave32s (`DxilIr::split`).
/// EXEC at launch must be the whole console wave, as the CPU interpreter and the HLSL path
/// start it.
pub fn kernel_split(insts: &[Decoded]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, true, DxilIr::split(insts, &Entry::Kernel))?;
    Ok((ir.finish([64, 1, 1], 32)?.dxil, stale))
}

/// `compute` in the split form: each console wave64 as two hardware wave32s, threads numbered
/// by arrival (`Lw::setup_lanes_compute`).
pub fn compute_split(insts: &[Decoded], cs: &super::CsInterface) -> Result<Vec<u8>, String> {
    let ir = lower::lower_compute(insts, cs, true, DxilIr::split(insts, &Entry::Compute(cs.clone())))?;
    Ok(ir.finish(cs.threads, 32)?.dxil)
}

/// `compute` from the program's code (decoded, far jumps resolved).
pub fn compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).map_err(|e| format!("cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    compute(&insts, cs, wave64)
}

/// Survey: every reason the per-lane form refuses a compute program (`lower::survey_compute`).
pub fn survey_compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Vec<String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match crate::gpu::rdna2::decode(&code[off..], off as u64) {
            Ok(d) => {
                off += d.len;
                insts.push(d);
            }
            Err(e) => return vec![format!("cannot decode ({e:?})")],
        }
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    lower::survey_compute(&insts, cs, wave64, DxilIr::new(&insts, wave64, &Entry::Compute(cs.clone())))
}

/// A compute program in its per-lane form, as a signed DXIL container.
pub fn compute(insts: &[Decoded], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let ir = lower::lower_compute(insts, cs, wave64, DxilIr::new(insts, wave64, &Entry::Compute(cs.clone())))?;
    Ok(ir.finish(cs.threads, if wave64 { 64 } else { 32 })?.dxil)
}


/// `XCL_PER_LANE=1` (with `XCL_DIRECT_DXIL`): compute programs try the per-lane form first, then
/// #1426's per-instruction form, then the HLSL path.
pub fn enabled() -> bool {
    std::env::var("XCL_PER_LANE").is_ok_and(|v| v == "1")
}

/// A compute program in its per-lane form with the runtime's options `o` (`pso::direct_compute`):
/// the container and its static samplers.
pub fn compute_with(code: &[u8], cs: &super::CsInterface, o: KernelOptions) -> Result<crate::gpu::dxil::kernel::Kernel, String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).map_err(|e| format!("cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    // as `kernel::translate_entry`: a thread with a large stack, a panic is a refusal
    let cs = cs.clone();
    std::thread::Builder::new()
        .name("xcl-per-lane".into())
        .stack_size(256 << 20)
        .spawn(move || {
            let ir = lower::lower_compute(&insts, &cs, o.wave64, DxilIr::with(&insts, o, &Entry::Compute(cs.clone())))?;
            ir.finish(cs.threads, if o.wave64 && !o.split { 64 } else { 32 })
        })
        .map_err(|e| format!("per-lane thread: {e}"))?
        .join()
        .map_err(|e| {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            format!("the per-lane lowering panicked: {msg}")
        })?
}

#[cfg(test)]
mod tests {
    use super::super::lower::tests::*;
    use super::super::{dxc, runner, translate, Mode, Options};

    fn gpu() -> bool {
        if !gpu_waves() {
            runner::skip_test("no GPU with 32-lane waves");
            return false;
        }
        true
    }

    /// A rounding mode #1426's lowering cannot honour refuses the program, as #1426 and the HLSL
    /// path do, instead of the per-lane form computing with the default mode (Kalmarv, #1745):
    /// f32 toward zero by `s_setreg`, f64 by `s_round_mode` and by `s_setreg`.
    #[test]
    fn a_rounding_mode_the_lowering_cannot_honour_is_refused() {
        let add_f32: I = ("V_ADD_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 1)]);
        let add_f64: I = ("V_ADD_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 2)]);
        let programs: [(&str, Vec<I>); 3] = [
            ("f32 by s_setreg", vec![("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", 0x801), ("LITERAL", 3)]), add_f32, ENDPGM]),
            ("f64 by s_round_mode", vec![("S_ROUND_MODE", "ENC_SOPP", &[("SIMM16", 0xC)]), add_f64, ENDPGM]),
            ("f64 by s_setreg", vec![("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", 0x881), ("LITERAL", 3)]), add_f64, ENDPGM]),
        ];
        for (name, p) in programs {
            assert!(super::kernel(&decode(&p), false).is_err(), "{name}: taken");
        }
        // denormal bits only (MODE[7:4]): the default rounding, taken
        let denorm: Vec<I> = vec![("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", 0x1901), ("LITERAL", 0xF)]), add_f32, ENDPGM];
        super::kernel(&decode(&denorm), false).unwrap();
    }

    /// An instruction lowered by #1426 that writes to `Lw`'s entry block after the program's
    /// blocks are written (scratch) gives a container, and runs.
    #[test]
    fn a_bridged_instruction_may_write_to_the_entry_block() {
        // scratch through #1426's lowering: its array is allocated in `Lw`'s entry block once
        // the program's blocks are written (an "instruction after a terminator" before)
        let lines: &[I] = &[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)]),
            ("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 1), ("SADDR", 0x7F)]),
            ("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 3), ("ADDR", 2), ("SADDR", 0x7F)]),
            ENDPGM,
        ];
        let dxil = super::kernel(&decode(lines), false).map(|r| r.0);
        let dxil = dxil.unwrap_or_else(|e| panic!("{e}"));
        if !gpu() {
            return;
        }
        let (s, v) = state(32);
        let mut st = runner::KernelState::default();
        st.sgpr.copy_from_slice(&s);
        for (l, regs) in v.iter().enumerate() {
            st.vgpr[l][..regs.len()].copy_from_slice(regs);
        }
        let out = runner::run(&dxil, &st, &[]).unwrap();
        for l in 0..32 {
            assert_eq!(out.vgpr[l][3], l as u32, "lane {l}");
        }
    }

    /// The f16 and packed programs of `lower`'s tests, as `f32_programs_on_the_gpu`.
    #[test]
    fn f16_programs_on_the_gpu() {
        programs_on_the_gpu(&[("f16 arithmetic", F16_ARITHMETIC, check_f16_arithmetic, &[]), ("f16 packed", F16_PACKED, check_f16_packed, &[])]);
    }

    /// The per-lane form as real DXIL on the GPU agrees with the lowering run on the CPU (and so
    /// with the HLSL path, which `lower`'s tests compare it with): registers and memory.
    #[test]
    fn per_lane_kernels_run_on_the_gpu() {
        if !gpu() {
            return;
        }
        let run = coalesced_run(false);
        for (name, lines) in [
            ("straight", STRAIGHT),
            ("skipped if", SKIPPED_IF),
            ("leaving loop", LEAVING_LOOP),
            ("uniform loop", UNIFORM_LOOP),
            ("memory", MEMORY),
            ("coalesced run", &run[..]),
            ("coalesced breaks", COALESCED_BREAKS),
            ("coalesced in partial", COALESCED_IN_PARTIAL),
            ("coalesced in loop", COALESCED_IN_LOOP),
        ] {
            let insts = decode(lines);
            let (dxil, stale) = super::kernel(&insts, false).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (s, v) = state(32);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (cs, cv, cmem) = run_mem(lines, false, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let mut sregs: Vec<usize> = vec![106, 126];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = crate::gpu::rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(gpu.sgpr[r], cs[r], "{name}: s{r}");
            }
            for l in 0..32 {
                for &r in &vregs {
                    assert_eq!(gpu.vgpr[l][r], cv[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    /// The integer programs of `lower`'s tests, as `f32_programs_on_the_gpu`.
    #[test]
    fn integer_programs_on_the_gpu() {
        programs_on_the_gpu(&[("integer arithmetic", INT_ARITHMETIC, check_int_arithmetic, &[]), ("integer carries", INT_CARRIES, check_int_carries, &[])]);
    }

    /// The f32 programs of `lower`'s tests on the GPU: the HLSL path gives the expected
    /// registers (`check_f32_*`, wave32), and the per-lane DXIL gives the HLSL path's, register
    /// for register (rcp within `f32_close`: HLSL's `rcp` is a fast fdiv). While the per-lane
    /// form refuses a program, that half is SKIPPED with the refusal.
    #[test]
    fn f32_programs_on_the_gpu() {
        programs_on_the_gpu(&[
            ("f32 arithmetic", F32_ARITHMETIC, check_f32_arithmetic, &[]),
            ("f32 compares", F32_COMPARES, check_f32_compares, &[]),
            ("f32 conversions", F32_CONVERSIONS, check_f32_conversions, &[]),
            ("f32 rounding", F32_ROUNDING, check_f32_rounding, &[]),
            ("f32 transcendentals", F32_TRANSCENDENTALS, check_f32_transcendentals, &[3, 4]),
            ("f32 three-source", F32_THREE_SOURCE, check_f32_three_source, &[]),
        ]);
    }

    /// A program of `lower`'s tests, the check of its registers on the HLSL path (wave32), and
    /// the VGPRs the per-lane form need only match within `f32_close`.
    type Program = (&'static str, &'static [I], fn(&[u32], &[Vec<u32>]), &'static [usize]);

    /// `f32_programs_on_the_gpu`'s two halves, for any programs.
    fn programs_on_the_gpu(programs: &[Program]) {
        if !gpu() {
            return;
        }
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let wave64 = runner::wave64_supported() == Ok(true);
        for &(name, lines, check, approx) in programs {
            for w64 in [false, true] {
                let lanes = if w64 { 64 } else { 32 };
                if w64 && !wave64 {
                    runner::skip_test(&format!("{name}, wave64: no wave64"));
                    continue;
                }
                let (s, v) = state(lanes);
                let mut st = runner::KernelState::default();
                st.sgpr.copy_from_slice(&s);
                for (l, regs) in v.iter().enumerate() {
                    st.vgpr[l][..regs.len()].copy_from_slice(regs);
                }
                let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
                let t = translate(&bytes, &Options { wave64: w64, ..Default::default() }).unwrap();
                assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
                let hlsl = runner::run(&dxc::compile(&t.hlsl, "main", "cs_6_6", &["-Gis"]).unwrap(), &st, &[]).unwrap();
                if !w64 {
                    check(&hlsl.sgpr, &hlsl.vgpr[..32]);
                }
                let insts = decode(lines);
                let (dxil, stale) = match super::kernel(&insts, w64) {
                    Ok(k) => k,
                    Err(e) => {
                        runner::skip_test(&format!("{name}, wave{lanes}, per-lane form: refused ({e})"));
                        continue;
                    }
                };
                let direct = runner::run(&dxil, &st, &[]).unwrap();
                let mut sregs: Vec<usize> = vec![106, 126];
                let mut vregs: Vec<usize> = Vec::new();
                for d in &insts {
                    let (r, w) = crate::gpu::rdna2::sgpr_uses(d);
                    sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                    let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                    vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
                }
                if w64 {
                    sregs.extend([107, 127]);
                }
                let stale_at = |r: usize| stale.contains(&(r as u8)) || (w64 && r % 2 == 1 && stale.contains(&(r as u8 - 1)));
                for &r in sregs.iter().filter(|&&r| !stale_at(r)) {
                    assert_eq!(direct.sgpr[r], hlsl.sgpr[r], "{name}, wave{lanes}: s{r}");
                }
                for l in 0..lanes {
                    for &r in &vregs {
                        let (d, h) = (direct.vgpr[l][r], hlsl.vgpr[l][r]);
                        let same = if approx.contains(&r) { f32_close(d, h) } else { d == h };
                        assert!(same, "{name}, wave{lanes}: lane {l} v{r}: {d:#x} against {h:#x}");
                    }
                }
            }
        }
    }

    /// 2-D and 3-D groups in the per-lane form: the threads numbered as the waves arrive
    /// (`Lw::setup_lanes_compute`), each wave's lanes the console's consecutive threads, as the
    /// lowering run on the CPU has them.
    #[test]
    fn a_per_lane_2d_or_3d_group_runs_on_the_gpu() {
        if !gpu() {
            return;
        }
        for wave64 in [false, true] {
            if wave64 && runner::wave64_supported() != Ok(true) {
                runner::skip_test("no wave64");
                return;
            }
            for threads in ID_SHAPES {
                let lines = ids_program(threads, wave64);
                let cs = ids_cs(threads);
                let dxil = super::compute(&decode(&lines), &cs, wave64).unwrap();
                let mem = vec![0u32; 256];
                let gpu = runner::dispatch(&dxil, [2, 1, 1], &[100, 0], &mem).unwrap();
                let cpu = run_compute_waves(&lines, &cs, wave64, 2, &[100, 0], &mem);
                assert_eq!(&gpu[..128], &cpu[..128], "{threads:?} wave64 {wave64}");
            }
        }
    }

    /// A wave64 program in the split form through `compute_with` (the runtime's path on a GPU
    /// with 32-lane waves only): 32-lane hardware waves (`WaveSize(32)`, not 64), the console's
    /// wave64 results (MiguelMachado-dev found `compute_with` passing 64).
    #[test]
    fn the_runtime_path_takes_the_split_form() {
        if !gpu() {
            return;
        }
        for threads in ID_SHAPES {
            let lines = ids_program(threads, true);
            let cs = ids_cs(threads);
            let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
            let o = super::KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
            let k = super::compute_with(&bytes, &cs, o).unwrap_or_else(|e| panic!("{threads:?}: {e}"));
            let mem = vec![0u32; 256];
            let gpu = runner::dispatch(&k.dxil, [2, 1, 1], &[100, 0], &mem).unwrap_or_else(|e| panic!("{threads:?}: {e}"));
            let cpu = run_compute_waves(&lines, &cs, true, 2, &[100, 0], &mem);
            assert_eq!(&gpu[..128], &cpu[..128], "{threads:?}");
        }
    }

    /// The f64 program of `lower`'s tests, as `f32_programs_on_the_gpu`.
    #[test]
    fn f64_programs_on_the_gpu() {
        programs_on_the_gpu(&[("f64", F64_ARITHMETIC, check_f64, &[])]);
    }

    #[test]
    fn a_per_lane_compute_program_runs_on_the_gpu() {
        if !gpu() {
            return;
        }
        let cs = compute_cs();
        let dxil = super::compute(&decode(COMPUTE), &cs, false).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..]);
        // and the HLSL path, for good measure
        if dxc::location().is_ok() {
            let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
            let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
            let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            assert_eq!(runner::dispatch(&hlsl, [3, 1, 1], &[100, 0], &mem).unwrap(), gpu);
        }
    }

    /// With the runtime's four memory windows (`compute_with`), the memory instructions go
    /// through #1426's window switch (blocks of its own in the `Ir` block) and the program
    /// still gives the CPU lowering's memory.
    #[test]
    fn a_per_lane_program_with_four_memory_windows_runs_on_the_gpu() {
        let cs = compute_cs();
        let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
        let o = super::KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 };
        if !gpu() {
            return;
        }
        let k = super::compute_with(&bytes, &cs, o).unwrap();
        let mem = vec![0u32; 256];
        let out = runner::dispatch(&k.dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&out[..cpu.len()], &cpu[..]);
    }

    /// Coalesced guarded runs lowered through `Ir::foreign` (`Lw::inst_guarded` with `guard = true`
    /// inside the run's block): f32 arithmetic against HLSL, and 4-window memory load/store inside
    /// a coalesced run (wave32 and split wave64, `Lw` making window-switch blocks inside the run)
    /// against the CPU lowering.
    #[test]
    fn coalesced_foreign_runs_agree_on_the_gpu() {
        if !gpu() {
            return;
        }
        // 4-window compute: lanes below the cut (16 in wave32; 40 in split wave64, spanning both
        // wave32 halves) run a coalesced load + add + store via `foreign`
        let cs = compute_cs();
        let mem = memory_image();
        for (wave64, split) in [(false, false), (true, true)] {
            let run = coalesced_run(wave64);
            let bytes: Vec<u8> = run.iter().flat_map(encode_line).collect();
            let o = super::KernelOptions { wave64, split, mem_windows: 4, sampler_base: 0 };
            let k = super::compute_with(&bytes, &cs, o).unwrap();
            let out = runner::dispatch(&k.dxil, [1, 1, 1], &[0, 0], &mem).unwrap();
            let cpu = run_compute_waves(&run, &cs, wave64, 1, &[0, 0], &mem);
            assert_eq!(&out[..cpu.len()], &cpu[..], "4-window coalesced memory via foreign (wave64={wave64}, split={split})");
        }

        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let (s, v) = state(32);
        let mut st = runner::KernelState::default();
        st.sgpr.copy_from_slice(&s);
        for (l, regs) in v.iter().enumerate() {
            st.vgpr[l][..regs.len()].copy_from_slice(regs);
        }
        let bytes: Vec<u8> = COALESCED_F32_RUN.iter().flat_map(encode_line).collect();
        let t = translate(&bytes, &Options { wave64: false, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let hlsl = runner::run(&dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap(), &st, &[]).unwrap();
        let (dxil, _) = super::kernel(&decode(COALESCED_F32_RUN), false).unwrap();
        let direct = runner::run(&dxil, &st, &[]).unwrap();
        for l in 0..32 {
            for r in 0..=5 {
                assert_eq!(direct.vgpr[l][r], hlsl.vgpr[l][r], "lane {l} v{r}");
            }
        }
        assert_eq!((direct.sgpr[106], direct.sgpr[126]), (hlsl.sgpr[106], hlsl.sgpr[126]));
    }
}
