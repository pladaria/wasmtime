use super::tests::{compile, target};
use super::{FRAME_BYTES, Location, TRANSFER_BYTES};
use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{self, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind, types};
use alloc::vec::Vec;

const COUNT: usize = 40;

#[test]
fn exit_comparisons_own_flags_on_both_poll_paths_under_pressure() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            for ty in [types::I32, types::I64] {
                for cost in [None, Some(0), Some(2048)] {
                    let mut f = ir::Function::new();
                    let block = f.dfg.make_block();
                    f.layout.append_block(block);
                    let mut signature = ir::Signature::new(crate::isa::CallConv::SystemV);
                    signature.returns = (0..COUNT).map(|_| ir::AbiParam::new(ty)).collect();
                    let signature = f.import_signature(signature);
                    let mut c = FuncCursor::new(&mut f).at_bottom(block);
                    let entry = c.ins().nixe_entry(signature, 1);
                    let values = c.func.dfg.inst_results(entry).to_vec();
                    c.ins().nixe_exit(2, &values);
                    f.nixe_exit_compares.insert(
                        2,
                        super::ExitCompare {
                            lhs: 0,
                            rhs: COUNT - 1,
                        },
                    );
                    if let Some(cost) = cost {
                        f.nixe_exit_costs.insert(2, cost);
                    }
                    let code = compile(f, &*isa).unwrap();
                    let maps = &code.buffer.nixe_states;
                    assert!(!maps[0].subtract_flags);
                    let map = &maps[1];
                    assert!(map.subtract_flags);
                    let reg = |index: usize| match map.values[index].location {
                        Location::Register {
                            index,
                            vector: false,
                        } => index,
                        other => panic!("compare operand was not constrained: {other:?}"),
                    };
                    let (lhs, rhs) = (reg(0), reg(COUNT - 1));
                    assert!(
                        map.values
                            .iter()
                            .any(|v| matches!(v.location, Location::Spill { .. }))
                    );
                    let cmp = if triple.starts_with("x86") {
                        vec![
                            0x40 | if ty == types::I64 { 8 } else { 0 }
                                | ((rhs >> 3) << 2)
                                | (lhs >> 3),
                            0x39,
                            0xc0 | ((rhs & 7) << 3) | (lhs & 7),
                        ]
                    } else {
                        let word = if ty == types::I64 {
                            0xeb00001fu32
                        } else {
                            0x6b00001f
                        } | (u32::from(rhs) << 16)
                            | (u32::from(lhs) << 5);
                        word.to_le_bytes().to_vec()
                    };
                    let bytes = code.code_buffer();
                    let end = map.offset as usize;
                    assert_eq!(&bytes[end - cmp.len()..end], &cmp);
                    if let Some(poll) = map.poll {
                        assert_eq!(poll.offset, map.offset + 2 * u32::from(map.patch_bytes));
                        let cold_cmp = end + usize::from(map.patch_bytes);
                        assert_eq!(&bytes[cold_cmp..cold_cmp + cmp.len()], &cmp);
                        assert_eq!(Some(poll.completed), cost);
                    } else {
                        assert_eq!(cost, None);
                    }
                }
            }
        }
    }
}

#[test]
fn exit_comparisons_reject_invalid_operands_and_clear_with_function() {
    let isa = target("x86_64-unknown-linux-gnu", "backtracking", true);
    for (a, b, lhs, rhs, id) in [
        (types::I32, types::I64, 0, 1, 2),
        (types::I8, types::I8, 0, 1, 2),
        (types::I64, types::I64, 0, 2, 2),
        (types::I64, types::I64, 0, 1, 99),
    ] {
        let mut f = ir::Function::new();
        let block = f.dfg.make_block();
        f.layout.append_block(block);
        let mut c = FuncCursor::new(&mut f).at_bottom(block);
        let a = c.ins().iconst(a, 1);
        let b = c.ins().iconst(b, 2);
        c.ins().nixe_exit(2, &[a, b]);
        f.nixe_exit_compares
            .insert(id, super::ExitCompare { lhs, rhs });
        assert!(
            compile(f.clone(), &*isa)
                .unwrap_err()
                .contains("Nixe exit comparison")
        );
        f.clear();
        assert!(f.nixe_exit_compares.is_empty());
    }
}

#[test]
fn literal_boundaries_preserve_exact_bits_without_allocating_registers() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let vector = 0xff80_0000_7654_3210_fedc_ba98_0123_4567u128;
            let handle = f
                .dfg
                .constants
                .insert(vector.to_le_bytes().as_slice().into());
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            let pointer = c.ins().get_pinned_reg(types::I64);
            let args = [
                c.ins().iconst(types::I8, -1),
                c.ins().iconst(types::I16, 0x8123),
                c.ins().iconst(types::I32, 0xfedc_ba98),
                c.ins().iconst(types::I64, -2),
                c.ins()
                    .f32const(ir::immediates::Ieee32::with_bits(0x7f80_0123)),
                c.ins()
                    .f64const(ir::immediates::Ieee64::with_bits(0x8000_0000_0000_0000)),
                c.ins().vconst(types::I8X16, handle),
            ];
            c.ins().nixe_state(1, &args);
            c.ins().nixe_fault_start(2, &args);
            let loaded = c.ins().load(types::I64, MemFlagsData::new(), pointer, 0);
            c.ins().nixe_fault_end(2, &[]);
            let mut exit = args.to_vec();
            exit.push(loaded);
            c.ins().nixe_exit(3, &exit);
            let code = compile(f, &*isa).unwrap();
            let expected = [
                [0xff, 0],
                [0x8123, 0],
                [0xfedc_ba98, 0],
                [u64::MAX - 1, 0],
                [0x7f80_0123, 0],
                [0x8000_0000_0000_0000, 0],
                [vector as u64, (vector >> 64) as u64],
            ];
            for map in code
                .buffer
                .nixe_states
                .iter()
                .chain(&code.buffer.nixe_faults)
            {
                for (actual, expected) in map.values.iter().zip(expected) {
                    assert_eq!(
                        actual.location,
                        Location::Constant(expected),
                        "{triple} {allocator}"
                    );
                }
            }
            assert_eq!(code.buffer.nixe_states.len(), 2);
            assert_eq!(code.buffer.nixe_faults.len(), 1);
            assert!(code.buffer.nixe_faults[0].fault_bytes > 0);
            assert!(matches!(
                code.buffer.nixe_states[1].values.last().unwrap().location,
                Location::Register { .. } | Location::Spill { .. }
            ));

            // Boundary-only literals add metadata but no machine instructions,
            // spill area, or constant pool. No alternate emission path is used.
            let sizes: Vec<_> = [0, 80]
                .into_iter()
                .map(|count| {
                    let mut f = ir::Function::new();
                    let block = f.dfg.make_block();
                    f.layout.append_block(block);
                    let mut c = FuncCursor::new(&mut f).at_bottom(block);
                    let values: Vec<_> = (0..count)
                        .map(|i| c.ins().iconst(types::I64, 0x1234_5678_0000_0000 + i))
                        .collect();
                    c.ins().nixe_exit(1, &values);
                    let code = compile(f, &*isa).unwrap();
                    (
                        code.code_buffer().len(),
                        code.buffer.frame_layout().unwrap().nixe_frame_size,
                    )
                })
                .collect();
            assert_eq!(sizes[0], sizes[1], "{triple} {allocator}");
        }
    }
}

#[cfg(feature = "disas")]
#[test]
fn guest_fault_delimiters_prevent_cross_access_forwarding_without_hardware_fences() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let isa = target(triple, "backtracking", true);
        for mode in 0..4 {
            for delimited in [false, true] {
                let mut f = ir::Function::new();
                let block = f.dfg.make_block();
                f.layout.append_block(block);
                let mut c = FuncCursor::new(&mut f).at_bottom(block);
                let address = c.ins().get_pinned_reg(types::I64);
                let one = c.ins().iconst(types::I64, 1);
                let two = c.ins().iconst(types::I64, 2);
                let flags = MemFlagsData::new();
                let mut results = Vec::new();
                for id in 1..=2 {
                    if delimited {
                        c.ins().nixe_fault_start(id, &[address]);
                    }
                    if mode == 0 || mode == 1 && id == 2 {
                        results.push(c.ins().load(types::I64, flags, address, 0));
                    } else {
                        let value = if mode == 3 && id == 2 { two } else { one };
                        c.ins().store(flags, value, address, 0);
                    }
                    if delimited {
                        c.ins().nixe_fault_end(id, &[]);
                    }
                }
                c.ins().nixe_exit(3, &results);
                let mut context = crate::Context::for_function(f);
                context
                    .compile(&*isa, &mut crate::control::ControlPlane::default())
                    .unwrap();
                let code = context.compiled_code().unwrap();
                assert_eq!(code.buffer.nixe_faults.len(), if delimited { 2 } else { 0 });
                let loads = context
                    .func
                    .layout
                    .blocks()
                    .flat_map(|block| context.func.layout.block_insts(block))
                    .filter(|&inst| context.func.dfg.insts[inst].opcode() == ir::Opcode::Load)
                    .count();
                if mode == 0 {
                    assert_eq!(loads, if delimited { 2 } else { 1 });
                } else if mode == 1 {
                    assert_eq!(loads, usize::from(delimited));
                }
                for (index, map) in code.buffer.nixe_faults.iter().enumerate() {
                    assert_eq!(map.id, index as u64 + 1);
                    assert!(map.fault_bytes > 0);
                }
                let decoder = isa.to_capstone().unwrap();
                let disassembly = decoder.disasm_all(code.code_buffer(), 0).unwrap();
                assert!(
                    disassembly
                        .iter()
                        .all(|inst| !matches!(inst.mnemonic(), Some("mfence" | "dmb" | "dsb")))
                );
            }
        }
    }
}

#[test]
fn constant_branch_removes_dead_checkpoint_costs_without_hiding_invalid_input() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let isa = target(triple, "backtracking", true);
        let mut f = ir::Function::new();
        let entry = f.dfg.make_block();
        let live = f.dfg.make_block();
        let dead = f.dfg.make_block();
        for block in [entry, live, dead] {
            f.layout.append_block(block);
        }
        let mut cursor = FuncCursor::new(&mut f);
        cursor.goto_bottom(entry);
        let condition = cursor.ins().iconst(types::I8, 1);
        let first = cursor.ins().iconst(types::I64, 4);
        let second = cursor.ins().iconst(types::I64, 6);
        cursor.ins().brif(condition, live, &[], dead, &[]);
        cursor.goto_bottom(live);
        cursor.ins().nixe_exit(1, &[first, second]);
        cursor.goto_bottom(dead);
        cursor.ins().nixe_exit(2, &[first, second]);
        f.nixe_exit_costs.insert(1, 7);
        f.nixe_exit_costs.insert(2, 13);
        for id in [1, 2] {
            f.nixe_exit_compares
                .insert(id, super::ExitCompare { lhs: 0, rhs: 1 });
        }
        let mut context = crate::Context::for_function(f.clone());
        context
            .compile(&*isa, &mut crate::control::ControlPlane::default())
            .unwrap();
        assert_eq!(context.func.nixe_exit_costs.len(), 1);
        assert_eq!(context.func.nixe_exit_costs.get(&1), Some(&7));
        assert_eq!(context.func.nixe_exit_compares.len(), 1);
        assert!(context.func.nixe_exit_compares.contains_key(&1));
        let maps = &context.compiled_code().unwrap().buffer.nixe_states;
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].id, 1);
        assert_eq!(maps[0].poll.unwrap().completed, 7);
        assert!(maps[0].subtract_flags);

        let mut invalid = f.clone();
        invalid
            .nixe_exit_compares
            .insert(2, super::ExitCompare { lhs: 0, rhs: 2 });
        assert!(
            compile(invalid, &*isa)
                .unwrap_err()
                .contains("Nixe exit comparison")
        );

        // Invalid costs must fail even if the corresponding exit would die.
        f.nixe_exit_costs.insert(2, 2049);
        assert!(compile(f, &*isa).unwrap_err().contains("Nixe checkpoint"));
    }
}

#[test]
fn internal_checks_export_cold_patch_and_keep_the_ssa_continuation() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            let mut f = fast_fragment();
            let exit = f.layout.last_inst(f.layout.entry_block().unwrap()).unwrap();
            let args = f.dfg.inst_args(exit).to_vec();
            FuncCursor::new(&mut f)
                .at_inst(exit)
                .ins()
                .nixe_check(77, &args);
            let code = compile(f, &*isa).unwrap();
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.id == 77)
                .unwrap();
            let terminal = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.id == 2)
                .unwrap();
            assert!(!map.entry && map.poll.is_none());
            assert_eq!(map.values.len(), args.len());
            assert!(map.values.iter().all(|v| v.location != Location::Unused));
            assert!(map.offset + u32::from(map.patch_bytes) <= terminal.offset);
            let expected = if triple.starts_with("x86") {
                vec![0x4d, 0x85, 0xf6, 0x7f, 8]
            } else {
                [0xf100029f_u32, 0x5400004c]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect()
            };
            assert_eq!(
                &code.code_buffer()[map.offset as usize - expected.len()..map.offset as usize],
                expected
            );
            let mut bytes = code.code_buffer().to_vec();
            map.patch_exit(
                &mut bytes,
                0,
                u64::from(map.offset + u32::from(map.patch_bytes)),
            )
            .unwrap();
        }
    }
}

#[test]
fn block_charges_preserve_flags_and_do_not_export_boundaries() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            for cost in [1_i64, 2048] {
                let mut f = fast_fragment();
                let block = f.layout.entry_block().unwrap();
                let exit = f.layout.last_inst(block).unwrap();
                let mut c = FuncCursor::new(&mut f).at_inst(exit);
                c.ins().nixe_charge(cost);
                c.ins().nixe_charge(cost); // costs are not unique boundary IDs
                let code = compile(f, &*isa).unwrap();
                let bytes = if triple.starts_with("x86") {
                    [0x4d, 0x8d, 0xb6]
                        .into_iter()
                        .chain((-(cost as i32)).to_le_bytes())
                        .collect::<Vec<_>>()
                } else {
                    (0xd1000294 | ((cost as u32) << 10)).to_le_bytes().to_vec()
                };
                assert_eq!(
                    code.code_buffer()
                        .windows(bytes.len())
                        .filter(|w| *w == bytes)
                        .count(),
                    2
                );
                assert!(
                    code.buffer
                        .nixe_states
                        .iter()
                        .all(|map| map.id == 1 || map.id == 2)
                );
            }
            for cost in [-1_i64, 0, 2049, 65537] {
                let mut f = fast_fragment();
                let exit = f.layout.last_inst(f.layout.entry_block().unwrap()).unwrap();
                FuncCursor::new(&mut f)
                    .at_inst(exit)
                    .ins()
                    .nixe_charge(cost);
                assert!(compile(f, &*isa).unwrap_err().contains("Nixe charge"));
            }
            let mut f = fast_fragment();
            let exit = f.layout.last_inst(f.layout.entry_block().unwrap()).unwrap();
            let mut c = FuncCursor::new(&mut f).at_inst(exit);
            c.ins().nixe_fault_start(3, &[]);
            c.ins().nixe_charge(1);
            c.ins().nixe_fault_end(3, &[]);
            assert!(compile(f, &*isa).unwrap_err().contains("Nixe charge"));

            // Isolate the charge: no other Nixe opcode can cause rejection.
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            c.ins().nixe_charge(1);
            c.ins().return_(&[]);
            assert!(
                compile(f, &*target(triple, allocator, false))
                    .unwrap_err()
                    .contains("Nixe charge")
            );
        }
    }
}

#[test]
fn checked_terminals_keep_allocated_operands_and_export_both_patches() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            for cost in [0, 1, 512, 2048] {
                let mut f = fast_fragment();
                f.nixe_exit_costs.insert(2, cost);
                let code = compile(f, &*isa).unwrap();
                let map = code
                    .buffer
                    .nixe_states
                    .iter()
                    .find(|map| map.id == 2)
                    .unwrap();
                let poll = map.poll.unwrap();
                assert_eq!(poll.completed, cost);
                assert_eq!(poll.offset, map.offset + u32::from(map.patch_bytes));
                assert_eq!(map.values.len(), COUNT * 2);
                assert!(
                    map.values
                        .iter()
                        .any(|value| matches!(value.location, Location::Spill { .. }))
                );
                assert!(
                    map.values
                        .iter()
                        .all(|value| value.location != Location::Unused)
                );
                let end = map.offset as usize;
                let expected: Vec<u8> = if map.patch_bytes == 8 {
                    [0x49, 0x81, 0xee]
                        .into_iter()
                        .chain(u32::from(cost).to_le_bytes())
                        .chain([0x7e, 8])
                        .collect()
                } else {
                    [0xf1000294 | (u32::from(cost) << 10), 0x5400004d]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect()
                };
                assert_eq!(&code.code_buffer()[end - expected.len()..end], expected);
                let mut bytes = code.code_buffer().to_vec();
                map.patch_exit(&mut bytes, 0, 0).unwrap();
                map.patch_poll(&mut bytes, 0, 0).unwrap();
                assert_ne!(
                    &bytes[end..end + map.patch_bytes as usize],
                    &bytes[poll.offset as usize..poll.offset as usize + map.patch_bytes as usize]
                );
                assert!(
                    map.patch_poll(&mut bytes[..poll.offset as usize], 0, 8)
                        .is_err()
                );
            }
            for (id, cost) in [(1, 1), (3, 1), (2, 2049)] {
                let mut f = fast_fragment();
                f.nixe_exit_costs.insert(id, cost);
                assert!(compile(f, &*isa).unwrap_err().contains("Nixe checkpoint"));
            }
        }
    }
    let mut f = fast_fragment();
    f.nixe_exit_costs.insert(2, 1);
    f.clear();
    assert!(f.nixe_exit_costs.is_empty());
}

#[cfg(feature = "disas")]
#[test]
fn cas128_maps_preserve_pre_values_through_lse_and_validating_pair_loop() {
    use crate::settings::Configurable;
    for allocator in ["single_pass", "backtracking"] {
        for lse in [false, true] {
            let base = target("aarch64-unknown-linux-gnu", allocator, true);
            let mut builder = crate::isa::lookup(base.triple().clone()).unwrap();
            builder
                .set("has_lse", if lse { "true" } else { "false" })
                .unwrap();
            let isa = builder.finish(base.flags().clone()).unwrap();
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            let address = c.ins().get_pinned_reg(types::I64);
            let values: Vec<_> = (0..COUNT)
                .map(|i| c.ins().iadd_imm_u(address, i as i64))
                .collect();
            let expected = c.ins().iconcat(values[1], values[2]);
            let replacement = c.ins().iconcat(values[3], values[4]);
            c.ins().nixe_fault_start(1, &values);
            let old = c
                .ins()
                .atomic_cas(MemFlagsData::new(), address, expected, replacement);
            c.ins().nixe_fault_end(1, &[]);
            let (lo, hi) = c.ins().isplit(old);
            let mut out = values.clone();
            out.extend([lo, hi]);
            c.ins().nixe_exit(2, &out);
            let code = compile(f, &*isa).unwrap();
            assert_eq!(code.buffer.nixe_faults.len(), if lse { 1 } else { 3 });
            let decoder = isa.to_capstone().unwrap();
            let decoded = decoder.disasm_all(code.code_buffer(), 0).unwrap();
            for (i, map) in code.buffer.nixe_faults.iter().enumerate() {
                assert_eq!(map.id, 1);
                assert_eq!(map.fault_bytes, 4);
                assert_eq!(map.values.len(), COUNT);
                for value in &map.values {
                    assert!(
                        !matches!(value.location, Location::Register { index, vector: false }
                        if matches!(index, 0 | 1) || (!lse && index == 7))
                    );
                }
                let inst = decoded
                    .iter()
                    .find(|inst| inst.address() == u64::from(map.offset))
                    .unwrap();
                assert_eq!(
                    inst.mnemonic(),
                    Some(if lse {
                        "caspal"
                    } else if i == 0 {
                        "ldaxp"
                    } else {
                        "stlxp"
                    })
                );
                if !lse && i == 2 {
                    assert_eq!(inst.op_str(), Some("w7, x0, x1, [x6]"));
                }
            }
        }
    }
}

#[cfg(feature = "disas")]
#[test]
fn nixe_rmw_unsigned_loop_masks_narrow_operands_and_lse_swap_is_direct() {
    use crate::settings::Configurable;
    for allocator in ["single_pass", "backtracking"] {
        for lse in [false, true] {
            let base = target("aarch64-unknown-linux-gnu", allocator, true);
            let mut builder = crate::isa::lookup(base.triple().clone()).unwrap();
            builder
                .set("has_lse", if lse { "true" } else { "false" })
                .unwrap();
            let isa = builder.finish(base.flags().clone()).unwrap();
            for ty in [types::I8, types::I16, types::I32, types::I64] {
                for op in [
                    ir::AtomicRmwOp::Umin,
                    ir::AtomicRmwOp::Umax,
                    ir::AtomicRmwOp::Xchg,
                ] {
                    let mut f = ir::Function::new();
                    let block = f.dfg.make_block();
                    f.layout.append_block(block);
                    let mut c = FuncCursor::new(&mut f).at_bottom(block);
                    let address = c.ins().get_pinned_reg(types::I64);
                    let operand = if ty == types::I64 {
                        address
                    } else {
                        c.ins().ireduce(ty, address)
                    };
                    c.ins().nixe_fault_start(1, &[address, operand]);
                    let old = c
                        .ins()
                        .atomic_rmw(ty, MemFlagsData::new(), op, address, operand);
                    c.ins().nixe_fault_end(1, &[]);
                    c.ins().nixe_exit(2, &[old]);
                    let code = compile(f, &*isa).unwrap();
                    assert_eq!(code.buffer.nixe_faults.len(), if lse { 1 } else { 2 });
                    let decoder = isa.to_capstone().unwrap();
                    let instructions = decoder.disasm_all(code.code_buffer(), 0).unwrap();
                    if lse && op == ir::AtomicRmwOp::Xchg {
                        assert!(
                            instructions
                                .iter()
                                .any(|i| i.mnemonic().unwrap().starts_with("swpal"))
                        );
                    } else if !lse && op != ir::AtomicRmwOp::Xchg && ty.bits() < 32 {
                        let extension = if ty == types::I8 { "uxtb" } else { "uxth" };
                        assert!(instructions.iter().any(|i| i.mnemonic() == Some("cmp")
                            && i.op_str().unwrap().contains(extension)));
                    }
                }
            }
        }
    }
}

#[cfg(feature = "disas")]
#[test]
fn nixe_cas_loop_has_fault_maps_for_both_conditional_stores() {
    for allocator in ["single_pass", "backtracking"] {
        let isa = target("aarch64-unknown-linux-gnu", allocator, true);
        for ty in [types::I8, types::I16, types::I32, types::I64] {
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            let address = c.ins().get_pinned_reg(types::I64);
            let expected = c.ins().iconst(ty, 3);
            let replacement = c.ins().iconst(ty, 9);
            c.ins()
                .nixe_fault_start(1, &[address, expected, replacement]);
            let old = c
                .ins()
                .atomic_cas(MemFlagsData::new(), address, expected, replacement);
            c.ins().nixe_fault_end(1, &[]);
            c.ins().nixe_exit(2, &[old]);
            let code = compile(f, &*isa).unwrap();
            assert_eq!(code.buffer.nixe_faults.len(), 3);
            let decoder = isa.to_capstone().unwrap();
            let instructions = decoder.disasm_all(code.code_buffer(), 0).unwrap();
            for (index, map) in code.buffer.nixe_faults.iter().enumerate() {
                assert_eq!(map.id, 1);
                assert_eq!(map.fault_bytes, 4);
                let inst = instructions
                    .iter()
                    .find(|i| i.address() == u64::from(map.offset))
                    .unwrap();
                assert!(inst.mnemonic().unwrap().starts_with(if index == 0 {
                    "ldaxr"
                } else {
                    "stlxr"
                }));
                if index == 2 {
                    assert!(inst.op_str().unwrap().contains(if ty == types::I64 {
                        "x27"
                    } else {
                        "w27"
                    }));
                }
            }
        }
    }
}

#[cfg(feature = "disas")]
#[test]
fn arena_address_uses_one_flag_preserving_instruction_and_requires_nixe() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let mut f = ir::Function::new();
            let block = f.dfg.make_block();
            f.layout.append_block(block);
            let mut c = FuncCursor::new(&mut f).at_bottom(block);
            let offset = c.ins().get_pinned_reg(types::I64);
            let address = c.ins().nixe_arena_addr(offset);
            c.ins().nixe_exit(1, &[address]);
            let disabled = target(triple, allocator, false);
            assert!(
                compile(f.clone(), &*disabled)
                    .unwrap_err()
                    .contains("requires the Nixe ABI")
            );
            let isa = target(triple, allocator, true);
            let code = compile(f, &*isa).unwrap();
            let decoder = isa.to_capstone().unwrap();
            let instructions = decoder.disasm_all(code.code_buffer(), 0).unwrap();
            let arena = if triple.starts_with("x86") {
                "r13"
            } else {
                "x19"
            };
            let uses: Vec<_> = instructions
                .iter()
                .filter(|inst| inst.op_str().unwrap_or("").contains(arena))
                .collect();
            assert_eq!(uses.len(), 1);
            assert_eq!(
                uses[0].mnemonic(),
                Some(if triple.starts_with("x86") {
                    "leaq"
                } else {
                    "add"
                })
            );
        }
    }
}

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
                    if triple.starts_with("aarch64") {
                        assert_eq!(map.fault_bytes, 4);
                        assert_eq!(map.offset % 4, 0);
                    } else {
                        assert!((1..=15).contains(&map.fault_bytes));
                    }
                    #[cfg(feature = "disas")]
                    {
                        // Decode independently in tests only. In production the
                        // emitter supplies the extent, including LOCK/prefixes.
                        let decoder = isa.to_capstone().unwrap();
                        let decoded = decoder
                            .disasm_count(&code.code_buffer()[map.offset as usize..], 0, 1)
                            .unwrap();
                        assert_eq!(decoded.len(), 1);
                        assert_eq!(
                            decoded.iter().next().unwrap().bytes().len(),
                            usize::from(map.fault_bytes)
                        );
                    }
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

#[cfg(feature = "disas")]
#[test]
fn nixe_fault_extents_exclude_following_instructions_and_padding() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            for ty in [types::I8, types::I16, types::I32, types::I64, types::I8X16] {
                for displacement in [0, 8, 8192] {
                    let mut f = ir::Function::new();
                    let block = f.dfg.make_block();
                    f.layout.append_block(block);
                    let mut c = FuncCursor::new(&mut f).at_bottom(block);
                    let address = c.ins().get_pinned_reg(types::I64);
                    c.ins().nixe_fault_start(1, &[address]);
                    let loaded = c.ins().load(ty, MemFlagsData::new(), address, displacement);
                    c.ins().nixe_fault_end(1, &[]);
                    c.ins().nixe_fault_start(2, &[address, loaded]);
                    c.ins()
                        .store(MemFlagsData::new(), loaded, address, displacement + 16);
                    c.ins().nixe_fault_end(2, &[]);
                    c.ins().nixe_exit(3, &[loaded]);
                    let code = compile(f, &*isa).unwrap();
                    assert_eq!(code.buffer.nixe_faults.len(), 2);
                    let decoder = isa.to_capstone().unwrap();
                    for map in &code.buffer.nixe_faults {
                        let instructions = decoder
                            .disasm_count(&code.code_buffer()[map.offset as usize..], 0, 2)
                            .unwrap();
                        assert_eq!(instructions.len(), 2);
                        let first = instructions.iter().next().unwrap();
                        assert_eq!(
                            usize::from(map.fault_bytes),
                            first.bytes().len(),
                            "{triple}/{allocator}/{ty}/{displacement}: {first}"
                        );
                    }
                    let maps = &code.buffer.nixe_faults;
                    assert!(maps[0].offset + u32::from(maps[0].fault_bytes) <= maps[1].offset);
                }
            }
        }
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
            fault_bytes: 0,
            poll: None,
            subtract_flags: false,
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
                            Location::Constant(_) => panic!("loaded inputs are not literals"),
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
                        Location::Constant(_) => panic!("entry definitions are not constants"),
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
                        Location::Constant(_) => panic!("loaded inputs are not literals"),
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
