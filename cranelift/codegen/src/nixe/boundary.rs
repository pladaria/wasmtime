//! Allocation-visible boundary operands, independent of guest semantics.

use crate::ir::{self, Type};
use crate::machinst::{
    FrameLayout, InstOutput, Lower, MachBuffer, OperandVisitor, Reg, VCodeInst, ValueRegs,
};
use alloc::{boxed::Box, vec::Vec};
use regalloc2::{OperandConstraint, OperandKind, OperandPos};

/// Constraint on one simultaneous `nixe_entry` result. Unused results are
/// still reported as `Location::Unused`. Register numbers use hardware encodings.
/// Spill offsets are chosen by allocation and reported in the final entry map;
/// requesting a particular spill slot (or forcing a spill) is not supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EntryConstraint {
    /// Let allocation choose a register or frame slot.
    Any,
    /// Require this physical register at ingress.
    Register {
        /// Architectural register encoding.
        index: u8,
        /// True for the SIMD/FP bank; false for integer registers.
        vector: bool,
    },
}

impl core::fmt::Display for EntryConstraint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Any => f.write_str("any"),
            Self::Register { index, vector } => {
                write!(f, "{} {index}", if *vector { "vector" } else { "integer" })
            }
        }
    }
}

/// A final location at a Nixe boundary. Spill offsets include the transfer
/// partition and explicit stack slots; they are relative to NativeFrame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Location {
    /// An entry result eliminated by optimization/lowering; transfer nothing.
    /// Exit and observation operands never have this location.
    Unused,
    /// Exact low/high 64-bit words of a literal. No allocation is required.
    /// This is a use-only location; entry definitions cannot be constants.
    Constant([u64; 2]),
    /// Architectural register number and register bank.
    Register {
        /// Architectural register encoding.
        index: u8,
        /// True for the SIMD/FP register bank.
        vector: bool,
    },
    /// Absolute byte offset from the pinned NativeFrame pointer.
    Spill {
        /// Byte offset from NativeFrame, not from SP.
        offset: u32,
    },
}

/// A typed operand in the original caller-supplied order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocatedValue {
    /// Original CLIF operand type.
    pub ty: Type,
    /// Location after final allocation.
    pub location: Location,
}

/// A terminal's already-charged deadline path. Both patches share the same
/// physical operands. Resumption must use the hot patch, not repeat the charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PollCheckpoint {
    /// Offset of the deadline exit patch, with the enclosing map's patch width.
    pub offset: u32,
    /// Work subtracted once before either patch; zero for already charged work.
    pub completed: u16,
}

/// Establish subtraction flags at an exit, not an ambient flag-liveness claim.
/// Both arguments must have the same I32 or I64 type. Their registers are
/// allocation-visible uses of the terminal; neither is modified by CMP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExitCompare {
    /// Minuend boundary argument index.
    pub lhs: usize,
    /// Subtrahend boundary argument index.
    pub rhs: usize,
}

/// Final state at an exact native offset, after all preceding allocator edits.
/// A state marker describes only its own point. Maps in `nixe_faults` instead
/// describe the actual fault PC, preserving operands through compound ops.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateMap {
    /// Opaque caller-supplied boundary identity.
    pub id: u64,
    /// Byte offset from the start of the compiled unit.
    pub offset: u32,
    /// True when values are simultaneous fast-entry definitions, not uses.
    pub entry: bool,
    /// Zero for a marker; 8 (x86-64) or 4 (AArch64) for an exit/check patch.
    pub patch_bytes: u8,
    /// Exact native instruction length for a fault map; zero for other maps.
    pub fault_bytes: u8,
    /// Optional deadline patch after an inline budget subtraction/branch.
    /// Mapped SSA values survive; only `subtract_flags` guarantees host flags.
    pub poll: Option<PollCheckpoint>,
    /// The terminal establishes all host integer subtraction condition flags
    /// at both patches. x86 CF is borrow; AArch64 C is not-borrow. No claim is
    /// made about flags at entry, ordinary markers or fault instructions.
    pub subtract_flags: bool,
    /// Allocations in the same order as the CLIF boundary arguments.
    pub values: Vec<LocatedValue>,
}

impl StateMap {
    /// Install an in-range direct branch in caller-owned, unpublished bytes.
    /// `base` is the eventual executable address of `code[0]`. For live code,
    /// the owner must first establish its maintenance rendezvous and writable
    /// mapping; this function provides neither synchronization nor W^X changes.
    /// Out-of-range targets require an owner-provided local island.
    pub fn patch_exit(&self, code: &mut [u8], base: u64, target: u64) -> crate::CodegenResult<()> {
        self.patch_at(self.offset, code, base, target)
    }

    /// Install the cold deadline target in unpublished bytes. The target sees
    /// an already-charged counter and the same operands as the normal exit.
    pub fn patch_poll(&self, code: &mut [u8], base: u64, target: u64) -> crate::CodegenResult<()> {
        let poll = self.poll.ok_or_else(|| {
            crate::CodegenError::Unsupported("Nixe exit has no poll checkpoint".into())
        })?;
        self.patch_at(poll.offset, code, base, target)
    }

    fn patch_at(
        &self,
        offset: u32,
        code: &mut [u8],
        base: u64,
        target: u64,
    ) -> crate::CodegenResult<()> {
        let fail = |detail: &str| {
            crate::CodegenError::Unsupported(alloc::format!("Nixe exit patch: {detail}"))
        };
        let address = base
            .checked_add(u64::from(offset))
            .ok_or_else(|| fail("address overflow"))?;
        let start = offset as usize;
        let end = start
            .checked_add(usize::from(self.patch_bytes))
            .ok_or_else(|| fail("offset overflow"))?;
        let patch = code
            .get_mut(start..end)
            .ok_or_else(|| fail("patch outside code"))?;
        match self.patch_bytes {
            8 => {
                if address % 8 != 0 {
                    return Err(fail("unaligned x86-64 patch"));
                }
                let delta = i32::try_from(i128::from(target) - (i128::from(address) + 5))
                    .map_err(|_| fail("x86-64 target requires an island"))?;
                patch.copy_from_slice(&[0xe9, 0, 0, 0, 0, 0x90, 0x90, 0x90]);
                patch[1..5].copy_from_slice(&delta.to_le_bytes());
            }
            4 => {
                if address % 4 != 0 || target % 4 != 0 {
                    return Err(fail("unaligned AArch64 branch"));
                }
                let delta = i128::from(target) - i128::from(address);
                if !(-(1i128 << 27)..(1i128 << 27)).contains(&delta) {
                    return Err(fail("AArch64 target requires an island"));
                }
                let instruction = 0x14000000 | (((delta / 4) as u32) & 0x03ffffff);
                patch.copy_from_slice(&instruction.to_le_bytes());
            }
            _ => return Err(fail("not an exit patch")),
        }
        Ok(())
    }
}

/// Machine-level boundary operands. Construct through CLIF boundary operations.
#[derive(Clone, Debug)]
pub struct Boundary {
    pub(crate) id: u64,
    pub(crate) entry: bool,
    pub(crate) exit: bool,
    pub(crate) check: bool,
    pub(crate) poll_cost: Option<u16>,
    pub(crate) charge: Option<u16>,
    compare: Option<ExitCompare>,
    pub(crate) fault_pos: OperandPos,
    values: Vec<(BoundaryValue, Type)>,
    entry_constraints: Vec<EntryConstraint>,
}

#[derive(Clone, Copy, Debug)]
enum BoundaryValue {
    Unused,
    Register(Reg),
    Constant([u64; 2]),
}

impl Boundary {
    // Consume only literal definitions in the optimized backend input, never
    // predict an allocator location from SSA. Constants are carried unchanged
    // through allocation and emitted in the final map; other values remain
    // real allocator operands. Avoiding put_input_in_regs also permits a
    // boundary-only literal to disappear from the machine instruction stream.
    fn constant<I: VCodeInst>(ctx: &Lower<I>, inst: ir::Inst, index: usize) -> Option<[u64; 2]> {
        let value = ctx.input_as_value(inst, index);
        let definition = ctx.dfg().value_def(value).inst()?;
        match *ctx.data(definition) {
            ir::InstructionData::UnaryImm {
                opcode: ir::Opcode::Iconst,
                imm,
            } => {
                let bits = ctx.value_ty(value).bits();
                (bits <= 64).then(|| [(imm.bits() as u64) & (u64::MAX >> (64 - bits)), 0])
            }
            ir::InstructionData::UnaryIeee32 {
                opcode: ir::Opcode::F32const,
                imm,
            } => Some([u64::from(imm.bits()), 0]),
            ir::InstructionData::UnaryIeee64 {
                opcode: ir::Opcode::F64const,
                imm,
            } => Some([imm.bits(), 0]),
            ir::InstructionData::UnaryConst {
                opcode: ir::Opcode::Vconst,
                constant_handle,
            } => {
                let bytes = ctx.get_constant_data(constant_handle).as_slice();
                (bytes.len() == 16).then(|| {
                    [
                        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                    ]
                })
            }
            _ => None,
        }
    }

    pub(crate) fn lower<I: VCodeInst>(
        ctx: &mut Lower<I>,
        inst: ir::Inst,
    ) -> Option<(Box<Self>, InstOutput)> {
        if let ir::InstructionData::UnaryImm {
            opcode: ir::Opcode::NixeCharge,
            imm,
        } = *ctx.data(inst)
        {
            return Some((
                Box::new(Self {
                    id: 0,
                    entry: false,
                    exit: false,
                    check: false,
                    poll_cost: None,
                    charge: Some(imm.bits() as u16),
                    compare: None,
                    fault_pos: OperandPos::Early,
                    values: Vec::new(),
                    entry_constraints: Vec::new(),
                }),
                InstOutput::new(),
            ));
        }
        if let ir::InstructionData::NixeEntry { imm, .. } = *ctx.data(inst) {
            let mut outputs = InstOutput::new();
            let mut values = Vec::new();
            for index in 0..ctx.num_outputs(inst) {
                let ty = ctx.output_ty(inst, index);
                let reg = ctx.alloc_tmp(ty).only_reg().unwrap().to_reg();
                let value = if ctx.nixe_result_is_used(inst, index) {
                    BoundaryValue::Register(reg)
                } else {
                    BoundaryValue::Unused
                };
                values.push((value, ty));
                outputs.push(ValueRegs::one(reg));
            }
            return Some((
                Box::new(Self {
                    id: imm.bits() as u64,
                    entry: true,
                    exit: false,
                    check: false,
                    poll_cost: None,
                    charge: None,
                    compare: None,
                    fault_pos: OperandPos::Early,
                    values,
                    entry_constraints: ctx
                        .f
                        .nixe_entry_constraints
                        .get(&(imm.bits() as u64))
                        .cloned()
                        .unwrap_or_default(),
                }),
                outputs,
            ));
        }
        let ir::InstructionData::NixeBoundary { opcode, imm, .. } = *ctx.data(inst) else {
            return None;
        };
        let compare = ctx.f.nixe_exit_compares.get(&(imm.bits() as u64)).copied();
        let values = (0..ctx.num_inputs(inst))
            .map(|index| {
                let ty = ctx.input_ty(inst, index);
                let value = if compare.is_some_and(|c| index == c.lhs || index == c.rhs) {
                    BoundaryValue::Register(ctx.put_input_in_regs(inst, index).only_reg().unwrap())
                } else if let Some(bits) = Self::constant(ctx, inst, index) {
                    BoundaryValue::Constant(bits)
                } else {
                    BoundaryValue::Register(ctx.put_input_in_regs(inst, index).only_reg().unwrap())
                };
                (value, ty)
            })
            .collect();
        Some((
            Box::new(Self {
                id: imm.bits() as u64,
                entry: false,
                exit: opcode == ir::Opcode::NixeExit,
                check: opcode == ir::Opcode::NixeCheck,
                poll_cost: ctx.f.nixe_exit_costs.get(&(imm.bits() as u64)).copied(),
                charge: None,
                compare,
                fault_pos: OperandPos::Early,
                values,
                entry_constraints: Vec::new(),
            }),
            InstOutput::new(),
        ))
    }

    pub(crate) fn operands(&mut self, collector: &mut impl OperandVisitor) {
        // Give mandatory register uses first choice before single-pass
        // allocation places flexible map-only values in remaining registers.
        let required = |index: &usize| {
            self.compare
                .is_some_and(|c| *index == c.lhs || *index == c.rhs)
        };
        let indices = (0..self.values.len())
            .filter(required)
            .chain((0..self.values.len()).filter(|i| !required(i)));
        for index in indices {
            let (reg, _) = &mut self.values[index];
            let BoundaryValue::Register(reg) = reg else {
                continue;
            };
            let (kind, pos) = if self.entry {
                (OperandKind::Def, OperandPos::Late)
            } else {
                (OperandKind::Use, OperandPos::Early)
            };
            let constraint = match self
                .entry_constraints
                .get(index)
                .copied()
                .unwrap_or(EntryConstraint::Any)
            {
                EntryConstraint::Any
                    if self
                        .compare
                        .is_some_and(|c| index == c.lhs || index == c.rhs) =>
                {
                    OperandConstraint::Reg
                }
                EntryConstraint::Any => OperandConstraint::Any,
                EntryConstraint::Register { index, vector } => {
                    OperandConstraint::FixedReg(regalloc2::PReg::new(
                        usize::from(index),
                        if vector {
                            regalloc2::RegClass::Float
                        } else {
                            regalloc2::RegClass::Int
                        },
                    ))
                }
            };
            collector.add_operand(reg, constraint, kind, pos);
        }
    }

    pub(crate) fn fault_operands(&mut self, collector: &mut impl OperandVisitor) {
        for (reg, _) in &mut self.values {
            let reg = match reg {
                BoundaryValue::Register(reg) => reg,
                BoundaryValue::Constant(_) => continue,
                BoundaryValue::Unused => unreachable!("prefault values are always used"),
            };
            // A precise single memory instruction faults before its defs;
            // a compound operation may have already written intermediate
            // results, so it must preserve these values through every def.
            collector.add_operand(
                reg,
                OperandConstraint::Any,
                OperandKind::Use,
                self.fault_pos,
            );
        }
    }

    pub(crate) fn fault_map(&self, frame: &FrameLayout) -> StateMap {
        StateMap {
            id: self.id,
            offset: 0,
            entry: false,
            patch_bytes: 0,
            fault_bytes: 0,
            poll: None,
            subtract_flags: false,
            values: self.locations(frame),
        }
    }

    pub(crate) fn record<I: VCodeInst>(
        &self,
        sink: &mut MachBuffer<I>,
        frame: &FrameLayout,
        patch_bytes: u8,
    ) {
        sink.push_nixe_state(StateMap {
            id: self.id,
            offset: sink.cur_offset(),
            entry: self.entry,
            patch_bytes: if self.exit || self.check {
                patch_bytes
            } else {
                0
            },
            fault_bytes: 0,
            poll: self.poll_cost.map(|completed| PollCheckpoint {
                // Each path has its own CMP immediately before its patch.
                // CMP plus alignment occupies one additional patch width.
                offset: sink.cur_offset()
                    + u32::from(patch_bytes) * if self.compare.is_some() { 2 } else { 1 },
                completed,
            }),
            subtract_flags: self.compare.is_some(),
            values: self.locations(frame),
        });
    }

    /// Physical compare operands after allocation. Register constraints keep
    /// the fused terminal free of scratch registers and late spill reloads.
    pub(crate) fn comparison(&self) -> Option<(Reg, Reg, Type)> {
        let compare = self.compare?;
        let (BoundaryValue::Register(lhs), ty) = self.values[compare.lhs] else {
            unreachable!("comparison operand must be allocated")
        };
        let (BoundaryValue::Register(rhs), _) = self.values[compare.rhs] else {
            unreachable!("comparison operand must be allocated")
        };
        assert!(lhs.to_real_reg().is_some() && rhs.to_real_reg().is_some());
        Some((lhs, rhs, ty))
    }

    fn locations(&self, frame: &FrameLayout) -> Vec<LocatedValue> {
        self.values
            .iter()
            .map(|&(reg, ty)| {
                let reg = match reg {
                    BoundaryValue::Register(reg) => reg,
                    BoundaryValue::Unused => {
                        return LocatedValue {
                            ty,
                            location: Location::Unused,
                        };
                    }
                    BoundaryValue::Constant(bits) => {
                        return LocatedValue {
                            ty,
                            location: Location::Constant(bits),
                        };
                    }
                };
                let location = if let Some(slot) = reg.to_spillslot() {
                    let offset = super::TRANSFER_BYTES
                        + u32::try_from(frame.spillslot_offset(slot)).unwrap();
                    assert!(offset + ty.bytes() <= super::FRAME_BYTES);
                    Location::Spill { offset }
                } else {
                    let reg = reg.to_real_reg().expect("final boundary allocation");
                    Location::Register {
                        index: reg.hw_enc(),
                        vector: reg.class() == regalloc2::RegClass::Float,
                    }
                };
                LocatedValue { ty, location }
            })
            .collect()
    }
}
