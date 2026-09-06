//! Two independently allocated fragments joined using their actual contracts.
//! Adapters and the explicit three-register shuffle are test-owned; this is
//! not Nixe's production gateway, generic bridge emitter or epoch protocol.

use super::tests::{compile, target};
use super::{EntryConstraint, Location, StateMap};
use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{self, InstBuilder, MemFlagsData, types};
use alloc::{vec, vec::Vec};

fn source() -> ir::Function {
    let mut f = ir::Function::new();
    let block = f.dfg.make_block();
    f.layout.append_block(block);
    let mut c = FuncCursor::new(&mut f).at_bottom(block);
    let frame = c.ins().get_pinned_reg(types::I64);
    let mut values = Vec::new();
    for i in 0..3 {
        let value = c
            .ins()
            .load(types::I64, MemFlagsData::trusted(), frame, i * 8);
        values.push(c.ins().iadd_imm_s(value, i64::from(i + 1)));
        values.push(
            c.ins()
                .load(types::I8X16, MemFlagsData::trusted(), frame, 512 + i * 16),
        );
    }
    c.ins().nixe_exit(10, &values);
    f
}

fn destination(locations: &[Location]) -> ir::Function {
    let mut f = ir::Function::new();
    let block = f.dfg.make_block();
    f.layout.append_block(block);
    let mut sig = ir::Signature::new(crate::isa::CallConv::SystemV);
    for _ in 0..3 {
        sig.returns.extend([
            ir::AbiParam::new(types::I64),
            ir::AbiParam::new(types::I8X16),
        ]);
    }
    let sig = f.import_signature(sig);
    f.nixe_entry_constraints.insert(
        20,
        locations
            .iter()
            .map(|location| {
                let Location::Register { index, vector } = *location else {
                    panic!("six live inputs should fit in registers")
                };
                EntryConstraint::Register { index, vector }
            })
            .collect(),
    );
    let mut c = FuncCursor::new(&mut f).at_bottom(block);
    let entry = c.ins().nixe_entry(sig, 20);
    let values = c.func.dfg.inst_results(entry).to_vec();
    let frame = c.ins().get_pinned_reg(types::I64);
    for i in 0..3 {
        c.ins().store(
            MemFlagsData::trusted(),
            values[i * 2],
            frame,
            128 + i as i32 * 8,
        );
        c.ins().store(
            MemFlagsData::trusted(),
            values[i * 2 + 1],
            frame,
            1024 + i as i32 * 16,
        );
    }
    c.ins().nixe_exit(30, &[]);
    f
}

fn move_register(bytes: &mut Vec<u8>, x64: bool, vector: bool, dst: u8, src: u8) {
    if x64 {
        if vector {
            bytes.extend_from_slice(&[
                0xf3,
                0x40 | ((dst >> 3) << 2) | (src >> 3),
                0x0f,
                0x6f,
                0xc0 | ((dst & 7) << 3) | (src & 7),
            ]);
        } else {
            bytes.extend_from_slice(&[
                0x48 | ((src >> 3) << 2) | (dst >> 3),
                0x89,
                0xc0 | ((src & 7) << 3) | (dst & 7),
            ]);
        }
    } else {
        let word = if vector {
            0x4ea01c00 | (u32::from(src) << 16) | (u32::from(src) << 5) | u32::from(dst)
        } else {
            0xaa0003e0 | (u32::from(src) << 16) | u32::from(dst)
        };
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}

fn vector_transfer(bytes: &mut Vec<u8>, x64: bool, reg: u8, load: bool) {
    // Slot 1536 is inside the reserved transfer area, outside test inputs and
    // outputs. No allocator-visible SIMD register is borrowed as scratch.
    if x64 {
        bytes.extend_from_slice(&[
            0xf3,
            0x41 | ((reg >> 3) << 2),
            0x0f,
            if load { 0x6f } else { 0x7f },
            0x87 | ((reg & 7) << 3),
        ]);
        bytes.extend_from_slice(&1536u32.to_le_bytes());
    } else {
        let word = (if load { 0x3dc00000 } else { 0x3d800000 })
            | ((1536 / 16) << 10)
            | (21 << 5)
            | u32::from(reg);
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}

fn cycle(bytes: &mut Vec<u8>, x64: bool, registers: &[u8], vector: bool) {
    let [a, b, c]: [u8; 3] = registers.try_into().unwrap();
    let scratch = if x64 { 11 } else { 16 };
    if vector {
        vector_transfer(bytes, x64, a, false);
    } else {
        move_register(bytes, x64, false, scratch, a);
    }
    move_register(bytes, x64, vector, a, c);
    move_register(bytes, x64, vector, c, b);
    if vector {
        vector_transfer(bytes, x64, b, true);
    } else {
        move_register(bytes, x64, false, b, scratch);
    }
}

#[test]
fn nixe_independent_fragments_accept_empty_and_cyclic_transfers() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let x64 = triple.starts_with("x86");
        for source_allocator in ["single_pass", "backtracking"] {
            for target_allocator in ["single_pass", "backtracking"] {
                for permute in [false, true] {
                    let a = compile(source(), &*target(triple, source_allocator, true)).unwrap();
                    let exit = &a.buffer.nixe_states[0];
                    let locations: Vec<_> = exit.values.iter().map(|v| v.location).collect();
                    let ingress: Vec<_> = (0..6)
                        .map(|i| locations[if permute { (i + 2) % 6 } else { i }])
                        .collect();
                    let b = compile(
                        destination(&ingress),
                        &*target(triple, target_allocator, true),
                    )
                    .unwrap();
                    assert_eq!(
                        b.buffer.nixe_states[0]
                            .values
                            .iter()
                            .map(|v| v.location)
                            .collect::<Vec<_>>(),
                        ingress
                    );
                    assert_eq!(b.buffer.nixe_states[0].offset, 0);
                    for code in [&a, &b] {
                        let text = code.vcode.as_ref().unwrap();
                        assert!(!text.contains("%rsp"));
                        assert!(!text.contains(", sp"));
                        assert!(
                            !text
                                .lines()
                                .any(|line| line.trim_start().starts_with("call")
                                    || line.trim_start().starts_with("ret"))
                        );
                    }
                    let mut bytes = a.code_buffer().to_vec();
                    let bridge_start = bytes.len();
                    let bridge_exit = if permute {
                        for vector in [false, true] {
                            let registers: Vec<_> = locations
                                .iter()
                                .filter_map(|location| match *location {
                                    Location::Register {
                                        index,
                                        vector: bank,
                                    } if bank == vector => Some(index),
                                    _ => None,
                                })
                                .collect();
                            cycle(&mut bytes, x64, &registers, vector);
                        }
                        let decoder = target(triple, source_allocator, true)
                            .to_capstone()
                            .unwrap();
                        let moves = decoder.disasm_all(&bytes[bridge_start..], 0).unwrap();
                        assert_eq!(moves.len(), 8);
                        assert_eq!(
                            moves.iter().map(|inst| inst.bytes().len()).sum::<usize>(),
                            bytes.len() - bridge_start
                        );
                        for inst in moves.iter() {
                            assert!(
                                matches!(
                                    inst.mnemonic(),
                                    Some("mov" | "movq" | "movdqu" | "str" | "ldr")
                                ),
                                "{triple}: {inst}"
                            );
                            let operands = inst.op_str().unwrap();
                            assert!(!operands.contains("rsp") && !operands.contains("sp"));
                        }
                        while bytes.len() % 8 != 0 {
                            if x64 {
                                bytes.push(0x90);
                            } else {
                                bytes.extend_from_slice(&0xd503201fu32.to_le_bytes());
                            }
                        }
                        let patch = StateMap {
                            id: 0,
                            offset: bytes.len() as u32,
                            entry: false,
                            patch_bytes: if x64 { 8 } else { 4 },
                            values: vec![],
                        };
                        bytes.resize(bytes.len() + patch.patch_bytes as usize, 0);
                        Some(patch)
                    } else {
                        None
                    };
                    // Align with actual target NOPs, never data executed on an edge.
                    while bytes.len() % 8 != 0 {
                        if x64 {
                            bytes.push(0x90);
                        } else {
                            bytes.extend_from_slice(&0xd503201fu32.to_le_bytes());
                        }
                    }
                    let target_start = bytes.len();
                    bytes.extend_from_slice(b.code_buffer());
                    let return_stub = bytes.len();
                    if x64 {
                        bytes.push(0xc3);
                    } else {
                        bytes.extend_from_slice(&0xd65f03c0u32.to_le_bytes());
                    }
                    exit.patch_exit(
                        &mut bytes,
                        0,
                        if permute { bridge_start } else { target_start } as u64,
                    )
                    .unwrap();
                    if let Some(patch) = bridge_exit {
                        patch
                            .patch_exit(&mut bytes, 0, target_start as u64)
                            .unwrap();
                    }
                    b.buffer.nixe_states[1]
                        .patch_exit(
                            &mut bytes[target_start..],
                            target_start as u64,
                            return_stub as u64,
                        )
                        .unwrap();
                    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                    if x64 {
                        execute(&bytes);
                    }
                }
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn execute(bytes: &[u8]) {
    use super::multi_entry::native::{Executable, nixe_probe_enter};
    #[repr(C, align(64))]
    struct Frame([u8; super::FRAME_BYTES as usize]);
    let executable = Executable::new(bytes);
    for seed in [1u64, 0x8123456789abcdef, u64::MAX - 80] {
        let mut frame = Frame([0xa5; super::FRAME_BYTES as usize]);
        for i in 0..3 {
            let value = seed.wrapping_add(i as u64 * 103);
            frame.0[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
            let vector = u128::from(value) | (u128::from(!value) << 64);
            frame.0[512 + i * 16..528 + i * 16].copy_from_slice(&vector.to_le_bytes());
        }
        // SAFETY: owned RX mapping/frame and a terminal test return stub.
        unsafe {
            nixe_probe_enter(
                frame.0.as_mut_ptr().cast(),
                executable.ptr.cast(),
                0xdeadbeef,
            );
        }
        for i in 0..3 {
            let expected = seed.wrapping_add(i as u64 * 103).wrapping_add(i as u64 + 1);
            assert_eq!(&frame.0[128 + i * 8..136 + i * 8], &expected.to_le_bytes());
            assert_eq!(
                &frame.0[1024 + i * 16..1040 + i * 16],
                &frame.0[512 + i * 16..528 + i * 16]
            );
        }
    }
}
