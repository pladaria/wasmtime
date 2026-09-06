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
    /// Zero for a marker; 8 (x86-64) or 4 (AArch64) for an external exit.
    pub patch_bytes: u8,
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
        let fail = |detail: &str| {
            crate::CodegenError::Unsupported(alloc::format!("Nixe exit patch: {detail}"))
        };
        let address = base
            .checked_add(u64::from(self.offset))
            .ok_or_else(|| fail("address overflow"))?;
        let start = self.offset as usize;
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
    pub(crate) fault_pos: OperandPos,
    pub(crate) values: Vec<(Option<Reg>, Type)>,
    entry_constraints: Vec<EntryConstraint>,
}

impl Boundary {
    pub(crate) fn lower<I: VCodeInst>(
        ctx: &mut Lower<I>,
        inst: ir::Inst,
    ) -> Option<(Box<Self>, InstOutput)> {
        if let ir::InstructionData::NixeEntry { imm, .. } = *ctx.data(inst) {
            let mut outputs = InstOutput::new();
            let mut values = Vec::new();
            for index in 0..ctx.num_outputs(inst) {
                let ty = ctx.output_ty(inst, index);
                let reg = ctx.alloc_tmp(ty).only_reg().unwrap().to_reg();
                values.push((ctx.nixe_result_is_used(inst, index).then_some(reg), ty));
                outputs.push(ValueRegs::one(reg));
            }
            return Some((
                Box::new(Self {
                    id: imm.bits() as u64,
                    entry: true,
                    exit: false,
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
        let values = (0..ctx.num_inputs(inst))
            .map(|index| {
                let ty = ctx.input_ty(inst, index);
                let reg = ctx.put_input_in_regs(inst, index).only_reg().unwrap();
                (Some(reg), ty)
            })
            .collect();
        Some((
            Box::new(Self {
                id: imm.bits() as u64,
                entry: false,
                exit: opcode == ir::Opcode::NixeExit,
                fault_pos: OperandPos::Early,
                values,
                entry_constraints: Vec::new(),
            }),
            InstOutput::new(),
        ))
    }

    pub(crate) fn operands(&mut self, collector: &mut impl OperandVisitor) {
        for (index, (reg, _)) in self.values.iter_mut().enumerate() {
            let Some(reg) = reg else {
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
            let reg = reg.as_mut().expect("prefault values are always used");
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
            patch_bytes: if self.exit { patch_bytes } else { 0 },
            values: self.locations(frame),
        });
    }

    fn locations(&self, frame: &FrameLayout) -> Vec<LocatedValue> {
        self.values
            .iter()
            .map(|&(reg, ty)| {
                let Some(reg) = reg else {
                    return LocatedValue {
                        ty,
                        location: Location::Unused,
                    };
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
