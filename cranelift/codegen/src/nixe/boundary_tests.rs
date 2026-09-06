use super::tests::{compile, target};
use super::{FRAME_BYTES, Location, TRANSFER_BYTES};
use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{self, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind, types};
use alloc::vec::Vec;

const COUNT: usize = 40;

#[test]
#[should_panic(expected = "Nixe memory trap was not given allocation-visible prefault operands")]
fn nixe_fault_emission_rejects_missing_allocation_operands() {
    let mut buffer = crate::machinst::MachBuffer::<crate::isa::x64::Inst>::new();
    buffer.set_nixe_fault(None, true);
    buffer.add_trap(ir::TrapCode::HEAP_OUT_OF_BOUNDS);
}

fn fault_fragment(operation: u8) -> ir::Function {
    let mut f = ir::Function::new();
    let entry = f.dfg.make_block();
    f.layout.append_block(entry);
    let mut c = FuncCursor::new(&mut f).at_bottom(entry);
    let frame = c.ins().get_pinned_reg(types::I64);
    let mut state = Vec::new();
    for i in 0..COUNT {
        state.push(
            c.ins()
                .load(types::I64, MemFlagsData::trusted(), frame, (i * 8) as i32),
        );
        state.push(c.ins().load(
            types::I8X16,
            MemFlagsData::trusted(),
            frame,
            (512 + i * 16) as i32,
        ));
    }
    let address = c
        .ins()
        .load(types::I64, MemFlagsData::trusted(), frame, 1504);
    c.ins().nixe_fault_start(30, &state);
    let value = if operation == 1 {
        c.ins().atomic_rmw(
            types::I64,
            MemFlagsData::new(),
            ir::AtomicRmwOp::Xor,
            address,
            state[0],
        )
    } else {
        c.ins().load(types::I64, MemFlagsData::new(), address, 0)
    };
    if operation == 2 {
        let updated = c.ins().iadd(value, state[0]);
        c.ins().store(MemFlagsData::new(), updated, address, 0);
    }
    c.ins().nixe_fault_end(30, &[]);
    // The old state is dead on successful completion: only the fault contract
    // keeps it live through the operation (including atomic internal defs).
    c.ins().nixe_exit(40, &[value]);
    f
}

#[test]
fn nixe_fault_maps_follow_real_memory_pcs_on_both_targets() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            for operation in [0, 1, 2] {
                let isa = target(triple, allocator, true);
                let code = compile(fault_fragment(operation), &*isa).unwrap();
                let faults = &code.buffer.nixe_faults;
                assert!(
                    !faults.is_empty(),
                    "{triple} {allocator} operation={operation}"
                );
                if operation == 1 && triple.starts_with("x86") {
                    assert_eq!(faults.len(), 2, "load and cmpxchg in the RMW sequence");
                }
                for map in faults {
                    assert_eq!(map.id, 30);
                    assert_eq!(map.values.len(), COUNT * 2);
                    assert!(!map.entry);
                    assert_eq!(map.patch_bytes, 0);
                    assert!(
                        code.buffer
                            .traps()
                            .iter()
                            .any(|trap| trap.offset == map.offset)
                    );
                    assert!(
                        map.values
                            .iter()
                            .any(|v| matches!(v.location, Location::Spill { .. }))
                    );
                    assert!(map.values.iter().all(|v| v.location != Location::Unused));
                }
                assert_eq!(
                    code.buffer.nixe_states.len(),
                    1,
                    "delimiters emit no boundary maps"
                );
            }
        }
    }
}

#[test]
fn nixe_fault_spans_reject_missing_pairs_and_unsafe_contents() {
    let isa = target("x86_64-unknown-linux-gnu", "single_pass", true);
    for (bad, expected) in [
        (0, "must match"),
        (1, "cannot nest"),
        (2, "cannot cross control flow"),
        (3, "requires a trapping memory"),
        (4, "cannot be notrap"),
        (5, "cannot contain non-memory traps"),
    ] {
        let mut f = ir::Function::new();
        let block = f.dfg.make_block();
        f.layout.append_block(block);
        let mut c = FuncCursor::new(&mut f).at_bottom(block);
        let address = c.ins().get_pinned_reg(types::I64);
        if bad != 0 {
            c.ins().nixe_fault_start(1, &[address]);
        }
        if bad == 1 {
            c.ins().nixe_state(2, &[address]);
        }
        if bad == 5 {
            c.ins().trapnz(address, ir::TrapCode::HEAP_OUT_OF_BOUNDS);
        }
        if bad == 4 {
            let value = c
                .ins()
                .load(types::I64, MemFlagsData::trusted(), address, 0);
            c.ins().nixe_fault_end(1, &[]);
            c.ins().nixe_exit(2, &[value]);
        } else {
            if bad != 2 {
                c.ins().nixe_fault_end(1, &[]);
            }
            c.ins().nixe_exit(3, &[]);
        }
        assert!(
            compile(f, &*isa).unwrap_err().contains(expected),
            "case {bad}"
        );
    }
}

#[test]
fn nixe_separate_fault_spans_keep_ids_and_values_distinct() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            let address = c.ins().get_pinned_reg(types::I64);
            c.ins().nixe_fault_start(1, &[address]);
            let first = c.ins().load(types::I64, MemFlagsData::new(), address, 0);
            c.ins().nixe_fault_end(1, &[]);
            c.ins().nixe_fault_start(2, &[first, address]);
            let second = c.ins().load(types::I64, MemFlagsData::new(), first, 0);
            c.ins().nixe_fault_end(2, &[]);
            c.ins().nixe_exit(3, &[second]);
            let code = compile(f, &*isa).unwrap();
            let maps = &code.buffer.nixe_faults;
            assert_eq!(maps.len(), 2);
            for (i, map) in maps.iter().enumerate() {
                assert_eq!(map.id, i as u64 + 1);
                assert_eq!(map.values.len(), i + 1);
            }
            assert!(maps[0].offset < maps[1].offset);
        }
    }
}

#[test]
fn nixe_jump_landings_are_included_in_fast_entry_offsets() {
    use crate::settings::{self, Configurable};
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let mut flags = settings::builder();
            flags.set("enable_pinned_reg", "true").unwrap();
            flags.set("enable_nixe_abi", "true").unwrap();
            flags.set("regalloc_algorithm", allocator).unwrap();
            flags
                .set(
                    "opt_level",
                    if allocator == "single_pass" {
                        "none"
                    } else {
                        "speed"
                    },
                )
                .unwrap();
            let mut builder = crate::isa::lookup(triple.parse().unwrap()).unwrap();
            let expected = if triple.starts_with("x86") {
                flags.set("enable_nixe_ibt", "true").unwrap();
                [0xf3, 0x0f, 0x1e, 0xfa]
            } else {
                builder.set("use_bti", "true").unwrap();
                0xd503249fu32.to_le_bytes() // BTI j, not BTI c
            };
            let isa = builder.finish(settings::Flags::new(flags)).unwrap();
            let code = compile(fast_fragment(), &*isa).unwrap();
            let entry = &code.buffer.nixe_states[0];
            assert_eq!(entry.offset, 0);
            assert_eq!(&code.code_buffer()[..4], &expected);
            assert!(code.buffer.alignment >= 8);
        }
    }
}

#[test]
fn nixe_exit_patch_range_alignment_and_direction() {
    for bytes in [4, 8] {
        let map = super::StateMap {
            id: 1,
            offset: 8,
            entry: false,
            patch_bytes: bytes,
            values: Vec::new(),
        };
        let base = 1u64 << 32;
        let source = base + 8;
        let (min, max, bias) = if bytes == 4 {
            (-(1i64 << 27), (1i64 << 27) - 4, 0)
        } else {
            (i64::from(i32::MIN), i64::from(i32::MAX), 5)
        };
        for delta in [min, -16, 0, 16, max] {
            let target = u64::try_from(source as i64 + bias + delta).unwrap();
            let mut code = [0xcc; 24];
            map.patch_exit(&mut code, base, target).unwrap();
            let decoded = if bytes == 4 {
                let word = u32::from_le_bytes(code[8..12].try_into().unwrap());
                assert_eq!(word >> 26, 5);
                i64::from(((word << 6) as i32) >> 4)
            } else {
                assert_eq!(code[8], 0xe9);
                assert_eq!(&code[13..16], &[0x90; 3]);
                i64::from(i32::from_le_bytes(code[9..13].try_into().unwrap()))
            };
            assert_eq!(decoded, delta);
            assert_eq!(&code[..8], &[0xcc; 8]);
            assert_eq!(
                &code[8 + bytes as usize..],
                &[0xcc; 24][8 + bytes as usize..]
            );
        }
        for (address, target) in [
            (base + 1, source),
            (base, (source as i64 + bias + min - 4) as u64),
            (base, (source as i64 + bias + max + 4) as u64),
        ] {
            let mut code = [0xcc; 24];
            assert!(map.patch_exit(&mut code, address, target).is_err());
            assert_eq!(
                code, [0xcc; 24],
                "failure leaves unpublished bytes untouched"
            );
        }
        assert!(map.patch_exit(&mut [0; 8], base, source).is_err());
        assert!(map.patch_exit(&mut [0; 24], u64::MAX, source).is_err());
    }
}

fn fast_fragment() -> ir::Function {
    let mut f = ir::Function::new();
    let entry = f.dfg.make_block();
    f.layout.append_block(entry);
    let mut signature = ir::Signature::new(crate::isa::CallConv::SystemV);
    for _ in 0..COUNT {
        signature.returns.push(ir::AbiParam::new(types::I64));
        signature.returns.push(ir::AbiParam::new(types::I8X16));
    }
    let signature = f.import_signature(signature);
    let mut c = FuncCursor::new(&mut f).at_bottom(entry);
    let entry = c.ins().nixe_entry(signature, 1);
    let mut args = c.func.dfg.inst_results(entry).to_vec();
    for i in 0..COUNT {
        args[i * 2] = c.ins().iadd_imm_s(args[i * 2], (i + 1) as i64);
    }
    c.ins().nixe_exit(2, &args);
    f
}

fn constrained_fragment() -> ir::Function {
    use super::EntryConstraint::{Any, Register};
    let mut f = fast_fragment();
    let mut constraints = alloc::vec![Any; COUNT * 2];
    constraints[..6].copy_from_slice(&[
        Register {
            index: 2,
            vector: false,
        },
        Register {
            index: 7,
            vector: true,
        },
        Any,
        Any,
        Register {
            index: 0,
            vector: false,
        },
        Register {
            index: 0,
            vector: true,
        },
    ]);
    f.nixe_entry_constraints.insert(1, constraints);
    f
}

#[test]
fn nixe_entry_constraints_survive_allocation_on_both_targets() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            let code = compile(constrained_fragment(), &*isa).unwrap();
            let values = &code.buffer.nixe_states[0].values;
            for (i, index, vector) in [(0, 2, false), (1, 7, true), (4, 0, false), (5, 0, true)] {
                assert_eq!(values[i].location, Location::Register { index, vector });
            }
            assert!(
                values
                    .iter()
                    .any(|v| matches!(v.location, Location::Spill { .. }))
            );
        }
    }
}

#[test]
fn nixe_entry_constraints_reject_invalid_contracts_and_clear_on_reuse() {
    use super::EntryConstraint::{Any, Register};
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let isa = target(triple, "single_pass", true);
        for (index, vector) in [
            (255, false),
            (15, true),
            (16, false),
            (18, false),
            (21, false),
        ] {
            // Index 15 in the vector bank is valid, but not for an I64 input.
            let mut f = constrained_fragment();
            f.nixe_entry_constraints.get_mut(&1).unwrap()[0] = Register { index, vector };
            assert!(
                compile(f, &*isa)
                    .unwrap_err()
                    .contains("invalid register or bank")
            );
        }
        for case in 0..3 {
            let mut f = constrained_fragment();
            let constraints = f.nixe_entry_constraints.get_mut(&1).unwrap();
            let expected = match case {
                0 => {
                    constraints.pop();
                    "match result count"
                }
                1 => {
                    constraints[2] = constraints[0];
                    "cannot overlap"
                }
                _ => {
                    f.nixe_entry_constraints.insert(100, alloc::vec![Any]);
                    "missing entry ID"
                }
            };
            assert!(compile(f, &*isa).unwrap_err().contains(expected));
        }
    }
    let mut f = constrained_fragment();
    f.clear();
    assert!(f.nixe_entry_constraints.is_empty());
}

#[test]
fn nixe_fast_multi_entry_keeps_loop_inputs_local() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let mut f = ir::Function::new();
            let a = f.dfg.make_block();
            let b = f.dfg.make_block();
            let body = f.dfg.make_block();
            let end = f.dfg.make_block();
            for block in [a, b, body, end] {
                f.layout.append_block(block);
            }
            f.layout.set_cold(b);
            let mut signature = ir::Signature::new(crate::isa::CallConv::SystemV);
            for _ in 0..COUNT {
                signature.returns.push(ir::AbiParam::new(types::I64));
            }
            let signature = f.import_signature(signature);
            let values: Vec<_> = (0..COUNT)
                .map(|_| f.dfg.append_block_param(body, types::I64))
                .collect();
            let count = f.dfg.append_block_param(body, types::I64);
            for (block, id) in [(a, 1), (b, 2)] {
                let mut constraints = alloc::vec![super::EntryConstraint::Any; COUNT];
                constraints[0] = super::EntryConstraint::Register {
                    index: id as u8,
                    vector: false,
                };
                f.nixe_entry_constraints.insert(id as u64, constraints);
                let mut c = FuncCursor::new(&mut f).at_bottom(block);
                let input = c.ins().nixe_entry(signature, id);
                let mut args: Vec<ir::BlockArg> = c
                    .func
                    .dfg
                    .inst_results(input)
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect();
                args.push(c.ins().iconst(types::I64, 3).into());
                c.ins().jump(body, &args);
            }
            let mut c = FuncCursor::new(&mut f).at_bottom(body);
            let count = c.ins().iadd_imm_s(count, -1);
            let mut args: Vec<ir::BlockArg> = values.iter().copied().map(Into::into).collect();
            args.push(count.into());
            c.ins().brif(count, body, &args, end, &[]);
            FuncCursor::new(&mut f)
                .at_bottom(end)
                .ins()
                .nixe_exit(3, &values);
            super::set_entries(&mut f, &[a, b]).unwrap();
            let isa = target(triple, allocator, true);
            let code = compile(f, &*isa).unwrap();
            for (block, id) in [(a, 1), (b, 2)] {
                let entry = code
                    .buffer
                    .nixe_states
                    .iter()
                    .find(|map| map.id == id)
                    .unwrap();
                let label = code
                    .buffer
                    .nixe_entries
                    .iter()
                    .find(|(label, _)| *label == block)
                    .unwrap()
                    .1;
                assert_eq!(
                    entry.offset, label,
                    "entry cannot skip constants or allocator edits"
                );
                assert!(entry.entry);
                assert_eq!(entry.values.len(), COUNT);
                assert_eq!(
                    entry.values[0].location,
                    Location::Register {
                        index: id as u8,
                        vector: false,
                    }
                );
            }
        }
    }
}

#[test]
fn nixe_fast_entry_with_unused_results() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let mut f = fast_fragment();
            let entry = f.layout.entry_block().unwrap();
            let end = f.layout.last_inst(entry).unwrap();
            let first = f.dfg.inst_results(f.layout.first_inst(entry).unwrap())[0];
            f.layout.remove_inst(end);
            FuncCursor::new(&mut f)
                .at_bottom(entry)
                .ins()
                .nixe_exit(2, &[first]);
            let isa = target(triple, allocator, true);
            let code = compile(f, &*isa).unwrap();
            assert!(code.buffer.nixe_states[0].entry);
            assert!(!matches!(
                code.buffer.nixe_states[0].values[0].location,
                Location::Unused
            ));
            assert!(
                code.buffer.nixe_states[0].values[1..]
                    .iter()
                    .all(|value| value.location == Location::Unused)
            );
        }
    }
}

#[test]
fn nixe_fast_entries_define_register_and_spill_inputs_simultaneously() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            let code = compile(fast_fragment(), &*isa).unwrap();
            let entry = &code.buffer.nixe_states[0];
            assert!(entry.entry);
            assert_eq!(
                entry.offset, 0,
                "no executable definitions before physical inputs"
            );
            assert_eq!(entry.values.len(), COUNT * 2);
            assert!(
                entry
                    .values
                    .iter()
                    .any(|value| matches!(value.location, Location::Spill { .. }))
            );
            // All independent live inputs need non-overlapping locations.
            for (i, value) in entry.values.iter().enumerate() {
                for other in &entry.values[..i] {
                    match (value.location, other.location) {
                        (Location::Spill { offset: a }, Location::Spill { offset: b }) => {
                            assert!(a + value.ty.bytes() <= b || b + other.ty.bytes() <= a);
                        }
                        (a, b) => assert_ne!(a, b),
                    }
                }
            }
            assert!(!code.buffer.nixe_states[1].entry);
        }
    }
}

fn fragment(cold: bool) -> ir::Function {
    let mut f = ir::Function::new();
    let entry = f.dfg.make_block();
    let end = f.dfg.make_block();
    f.layout.append_block(entry);
    f.layout.append_block(end);
    if cold {
        f.layout.set_cold(end);
    }
    let slot = f.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 48, 4));
    let mut c = FuncCursor::new(&mut f).at_bottom(entry);
    let frame = c.ins().get_pinned_reg(types::I64);
    let address = c.ins().stack_addr(types::I64, slot, 0);
    c.ins().store(MemFlagsData::trusted(), address, frame, 1504);
    let mut args = Vec::new();
    for i in 0..COUNT {
        let value = c
            .ins()
            .load(types::I64, MemFlagsData::trusted(), frame, (i * 8) as i32);
        args.push(c.ins().iadd_imm_s(value, (i + 1) as i64));
        args.push(c.ins().load(
            types::I8X16,
            MemFlagsData::trusted(),
            frame,
            (512 + i * 16) as i32,
        ));
    }
    c.ins().nixe_state(10, &args);
    c.ins().jump(end, &[]);
    let mut c = FuncCursor::new(&mut f).at_bottom(end);
    // Repeat one operand: the map must retain caller order and aliases.
    args.push(args[0]);
    c.ins().nixe_exit(20, &args);
    f
}

#[test]
fn nixe_boundaries_export_final_allocations_and_aligned_patch_units() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            for cold in [false, true] {
                let isa = target(triple, allocator, true);
                let code = compile(fragment(cold), &*isa).unwrap();
                assert_eq!(code.buffer.nixe_states.len(), 2);
                let extent = code.buffer.frame_layout().unwrap().nixe_frame_size.unwrap();
                for (index, map) in code.buffer.nixe_states.iter().enumerate() {
                    assert_eq!(map.id, if index == 0 { 10 } else { 20 });
                    assert_eq!(map.values.len(), COUNT * 2 + index);
                    assert!(
                        map.values
                            .iter()
                            .any(|value| matches!(value.location, Location::Spill { .. }))
                    );
                    for value in &map.values {
                        match value.location {
                            Location::Spill { offset } => {
                                assert!(offset >= TRANSFER_BYTES + 48);
                                assert!(offset + value.ty.bytes() <= extent);
                                assert!(extent <= FRAME_BYTES);
                            }
                            Location::Register {
                                index,
                                vector: false,
                            } => {
                                let reserved: &[u8] = if triple.starts_with("x86") {
                                    &[4, 5, 11, 13, 14, 15]
                                } else {
                                    &[16, 17, 18, 19, 20, 21, 29, 30, 31]
                                };
                                assert!(!reserved.contains(&index));
                            }
                            Location::Register { .. } => {}
                            Location::Unused => panic!("boundary operand was lost"),
                        }
                    }
                }
                let exit = &code.buffer.nixe_states[1];
                let patch = exit.offset as usize;
                if triple.starts_with("x86") {
                    assert_eq!(exit.patch_bytes, 8);
                    assert_eq!(patch % 8, 0);
                    assert_eq!(
                        &code.code_buffer()[patch..patch + 8],
                        &[0x0f, 0x0b, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]
                    );
                } else {
                    assert_eq!(exit.patch_bytes, 4);
                    assert_eq!(patch % 4, 0);
                    assert_eq!(
                        &code.code_buffer()[patch..patch + 4],
                        &0xd4200000u32.to_le_bytes()
                    );
                }
                assert_eq!(exit.values[0], *exit.values.last().unwrap());
                assert_eq!(code.buffer.nixe_states[0].patch_bytes, 0);
            }
        }
    }
}

#[test]
fn nixe_boundary_configuration_and_types_are_checked() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let ordinary = target(triple, "single_pass", false);
        assert!(
            compile(fragment(false), &*ordinary)
                .unwrap_err()
                .contains("requires Nixe ABI")
        );
        let isa = target(triple, "backtracking", true);
        for id in [-1, 10] {
            let mut f = fragment(false);
            let last = f.layout.last_inst(f.layout.last_block().unwrap()).unwrap();
            if let ir::InstructionData::NixeBoundary { imm, .. } = &mut f.dfg.insts[last] {
                *imm = id.into();
            }
            assert!(
                compile(f, &*isa)
                    .unwrap_err()
                    .contains("unique nonnegative IDs")
            );
        }
        let mut f = ir::Function::new();
        let entry = f.dfg.make_block();
        f.layout.append_block(entry);
        let mut c = FuncCursor::new(&mut f).at_bottom(entry);
        c.ins().nixe_exit(0, &[]);
        let code = compile(f, &*isa).unwrap();
        assert!(code.buffer.nixe_states[0].values.is_empty());
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn nixe_exit_maps_reconstruct_native_registers_and_spills() {
    use super::multi_entry::native::{Executable, nixe_probe_enter};
    #[repr(C, align(64))]
    struct Frame([u8; FRAME_BYTES as usize + 384]);
    for allocator in ["single_pass", "backtracking"] {
        for (cold, fast, constrained) in [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, true, true),
        ] {
            let isa = target("x86_64-unknown-linux-gnu", allocator, true);
            let code = compile(
                if constrained {
                    constrained_fragment()
                } else if fast {
                    fast_fragment()
                } else {
                    fragment(cold)
                },
                &*isa,
            )
            .unwrap();
            let map = &code.buffer.nixe_states[1];
            let mut bytes = Vec::new();
            if fast {
                // Test-owned canonical adapter, built only from the entry map.
                // Source bytes are in the test's transfer area, never in host
                // registers that an earlier load could overwrite.
                let entry = &code.buffer.nixe_states[0];
                assert_eq!(entry.offset, 0);
                for (i, value) in entry.values.iter().enumerate() {
                    let input = if i % 2 == 0 {
                        (i / 2) * 8
                    } else {
                        512 + (i / 2) * 16
                    } as u32;
                    match value.location {
                        Location::Unused => continue,
                        Location::Register {
                            index,
                            vector: false,
                        } => {
                            bytes.extend_from_slice(&[
                                0x49 | ((index >> 3) << 2),
                                0x8b,
                                0x87 | ((index & 7) << 3),
                            ]);
                            bytes.extend_from_slice(&input.to_le_bytes());
                        }
                        Location::Register {
                            index,
                            vector: true,
                        } => {
                            bytes.extend_from_slice(&[
                                0xf3,
                                0x41 | ((index >> 3) << 2),
                                0x0f,
                                0x6f,
                                0x87 | ((index & 7) << 3),
                            ]);
                            bytes.extend_from_slice(&input.to_le_bytes());
                        }
                        Location::Spill { offset } => {
                            for part in (0..value.ty.bytes()).step_by(8) {
                                bytes.extend_from_slice(&[0x4d, 0x8b, 0x9f]); // mov r11, [r15+source]
                                bytes.extend_from_slice(&(input + part).to_le_bytes());
                                bytes.extend_from_slice(&[0x4d, 0x89, 0x9f]); // mov [r15+slot], r11
                                bytes.extend_from_slice(&(offset + part).to_le_bytes());
                            }
                        }
                    }
                }
                while bytes.len() % 16 != 0 {
                    bytes.push(0x90);
                }
            }
            let body = bytes.len();
            bytes.extend_from_slice(code.code_buffer());
            let capture = bytes.len();
            // A test-only snapshot stub, reached through the REAL exit patch.
            // MOV instructions preserve every source register and host flags.
            for register in 0u8..16 {
                bytes.extend_from_slice(&[
                    0x49 | ((register >> 3) << 2),
                    0x89,
                    0x87 | ((register & 7) << 3),
                ]);
                bytes.extend_from_slice(&(FRAME_BYTES + u32::from(register) * 8).to_le_bytes());
            }
            for register in 0u8..16 {
                bytes.extend_from_slice(&[
                    0xf3,
                    0x41 | ((register >> 3) << 2),
                    0x0f,
                    0x7f,
                    0x87 | ((register & 7) << 3),
                ]);
                bytes.extend_from_slice(
                    &(FRAME_BYTES + 128 + u32::from(register) * 16).to_le_bytes(),
                );
            }
            bytes.push(0xc3); // Return only from the test-owned boundary stub.
            // Both positions share a relocation base, so this displacement is
            // unchanged when the owned copy is mapped executable.
            map.patch_exit(&mut bytes[body..], body as u64, capture as u64)
                .unwrap();
            let executable = Executable::new(&bytes);
            for seed in [1u64, 0x8123456789abcdef, u64::MAX - 80] {
                let mut frame = Frame([0xa5; FRAME_BYTES as usize + 384]);
                let mut expected = Vec::new();
                for i in 0..COUNT {
                    let input = seed.wrapping_add(i as u64 * 103);
                    frame.0[i * 8..i * 8 + 8].copy_from_slice(&input.to_le_bytes());
                    expected.push(input.wrapping_add((i + 1) as u64).to_le_bytes().to_vec());
                    let vector = u128::from(input) | (u128::from(!input) << 64);
                    frame.0[512 + i * 16..528 + i * 16].copy_from_slice(&vector.to_le_bytes());
                    expected.push(vector.to_le_bytes().to_vec());
                }
                if !fast {
                    expected.push(expected[0].clone());
                }
                // SAFETY: validated generated leaf, owned aligned frame, RX
                // mapping, and local capture stub returning to the ABI adapter.
                unsafe {
                    nixe_probe_enter(
                        frame.0.as_mut_ptr().cast(),
                        executable.ptr.cast(),
                        0xdeadbeef,
                    );
                }
                for (value, expected) in map.values.iter().zip(expected) {
                    let offset = match value.location {
                        Location::Unused => panic!("exit operand was lost"),
                        Location::Spill { offset } => offset as usize,
                        Location::Register { index, vector } => {
                            FRAME_BYTES as usize
                                + if vector {
                                    128 + index as usize * 16
                                } else {
                                    index as usize * 8
                                }
                        }
                    };
                    assert_eq!(
                        &frame.0[offset..offset + value.ty.bytes() as usize],
                        expected,
                        "{allocator}: {value:?}"
                    );
                }
            }
        }
    }
}
