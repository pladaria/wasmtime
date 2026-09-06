//! Native prefault proof using the ordinary backend's load and atomic lowering.
//! This is deliberately isolated from Nixe's production signal/retry machinery.
#![cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{self, InstBuilder, MemFlagsData, types};
use cranelift_codegen::nixe::{FRAME_BYTES, Location};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, isa};
use cranelift_control::ControlPlane;
use std::sync::atomic::{AtomicPtr, Ordering};

const COUNT: usize = 40;

#[repr(C, align(64))]
struct Frame {
    bytes: [u8; FRAME_BYTES as usize],
    gregs: [libc::greg_t; 23],
    vectors: [[u32; 4]; 16],
    resume: usize,
    faults: usize,
}

static ACTIVE: AtomicPtr<Frame> = AtomicPtr::new(std::ptr::null_mut());

extern "C" fn capture(_signal: i32, _info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: installed with SA_SIGINFO in an isolated child. The active frame
    // outlives execution and signal delivery. No allocation or unwinding here.
    unsafe {
        let context = &mut *context.cast::<libc::ucontext_t>();
        let frame = ACTIVE.load(Ordering::Relaxed);
        if frame.is_null()
            || context.uc_mcontext.gregs[libc::REG_R15 as usize] as usize != frame as usize
        {
            libc::_exit(101);
        }
        let frame = &mut *frame;
        frame.gregs = context.uc_mcontext.gregs;
        let fp = &*context.uc_mcontext.fpregs;
        for (dst, src) in frame.vectors.iter_mut().zip(&fp._xmm) {
            *dst = src.element;
        }
        frame.faults += 1;
        context.uc_mcontext.gregs[libc::REG_RIP as usize] = frame.resume as libc::greg_t;
    }
}

// Test-owned adapter: the generated body has no system prologue/return. On a
// fault, sigreturn redirects to a test RET stub returning to this adapter.
core::arch::global_asm!(
    ".pushsection .text",
    ".global nixe_fault_probe_enter",
    ".hidden nixe_fault_probe_enter",
    ".type nixe_fault_probe_enter,@function",
    "nixe_fault_probe_enter:",
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "mov r15, rdi",
    "call rsi",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    ".size nixe_fault_probe_enter, .-nixe_fault_probe_enter",
    ".popsection",
);

unsafe extern "C" {
    fn nixe_fault_probe_enter(frame: *mut Frame, entry: *const u8);
}

fn fragment(operation: u8, count: usize) -> ir::Function {
    let mut f = ir::Function::new();
    let block = f.dfg.make_block();
    f.layout.append_block(block);
    let mut c = FuncCursor::new(&mut f).at_bottom(block);
    let frame = c.ins().get_pinned_reg(types::I64);
    let mut values = Vec::new();
    for i in 0..count {
        values.push(
            c.ins()
                .load(types::I64, MemFlagsData::trusted(), frame, (i * 8) as i32),
        );
        values.push(c.ins().load(
            types::I8X16,
            MemFlagsData::trusted(),
            frame,
            (512 + i * 16) as i32,
        ));
    }
    values.push(values[0]); // Caller aliases retain their original map order.
    let address = c
        .ins()
        .load(types::I64, MemFlagsData::trusted(), frame, 1504);
    c.ins().nixe_fault_start(1, &values);
    let result = if operation == 1 {
        c.ins().atomic_rmw(
            types::I64,
            MemFlagsData::new(),
            ir::AtomicRmwOp::Xor,
            address,
            values[0],
        )
    } else {
        c.ins().load(types::I64, MemFlagsData::new(), address, 0)
    };
    if operation == 2 {
        let updated = c.ins().iadd(result, values[0]);
        c.ins().store(MemFlagsData::new(), updated, address, 0);
    }
    c.ins().nixe_fault_end(1, &[]);
    c.ins().nixe_exit(2, &[result]);
    f
}

#[test]
fn nixe_native_prefault_registers_and_spills() {
    const CHILD: &str = "NIXE_NATIVE_PREFAULT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "nixe_native_prefault_registers_and_spills",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }

    // SAFETY: signal disposition is changed only in this isolated test process.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = capture as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()),
            0
        );
    }

    for allocator in ["single_pass", "backtracking"] {
        let mut flags = settings::builder();
        flags.set("enable_pinned_reg", "true").unwrap();
        flags.set("enable_nixe_abi", "true").unwrap();
        flags.set("regalloc_checker", "true").unwrap();
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
        let isa = isa::lookup("x86_64-unknown-linux-gnu".parse().unwrap())
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap();
        for count in [1, 9, COUNT] {
            for (operation, second) in [(0, false), (1, false), (1, true), (2, true)] {
                let mut cx = Context::for_function(fragment(operation, count));
                let code = cx.compile(&*isa, &mut ControlPlane::default()).unwrap();
                let mut bytes = code.code_buffer().to_vec();
                let resume = bytes.len();
                bytes.push(0xc3); // owned return stub, never part of generated body
                code.buffer.nixe_states[0]
                    .patch_exit(&mut bytes, 0, resume as u64)
                    .unwrap();
                let mut mapping = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
                mapping[..bytes.len()].copy_from_slice(&bytes);
                let executable = mapping.make_exec().unwrap();
                let map = &code.buffer.nixe_faults[usize::from(second)];
                for seed in [1u64, 0x8123456789abcdef, u64::MAX - 80] {
                    let mut page = memmap2::MmapMut::map_anon(4096).unwrap();
                    page[..8].copy_from_slice(&0xfeed0123456789abu64.to_le_bytes());
                    // Read-only memory faults at CMPXCHG, after the initial load
                    // has overwritten RAX and the sequence has overwritten temp.
                    // PROT_NONE instead exercises the first memory instruction.
                    unsafe {
                        assert_eq!(
                            libc::mprotect(
                                page.as_mut_ptr().cast(),
                                page.len(),
                                if second {
                                    libc::PROT_READ
                                } else {
                                    libc::PROT_NONE
                                }
                            ),
                            0
                        );
                    }
                    let mut frame = Frame {
                        bytes: [0xa5; FRAME_BYTES as usize],
                        gregs: [0; 23],
                        vectors: [[0; 4]; 16],
                        resume: executable.as_ptr() as usize + resume,
                        faults: 0,
                    };
                    frame.bytes[1504..1512].copy_from_slice(&(page.as_ptr() as u64).to_le_bytes());
                    let mut expected = Vec::new();
                    for i in 0..count {
                        let input = seed.wrapping_add(i as u64 * 103);
                        let vector = u128::from(input) | (u128::from(!input) << 64);
                        frame.bytes[i * 8..i * 8 + 8].copy_from_slice(&input.to_le_bytes());
                        frame.bytes[512 + i * 16..528 + i * 16]
                            .copy_from_slice(&vector.to_le_bytes());
                        expected.push(input.to_le_bytes().to_vec());
                        expected.push(vector.to_le_bytes().to_vec());
                    }
                    expected.push(expected[0].clone());
                    ACTIVE.store(&mut frame, Ordering::Relaxed);
                    // SAFETY: owned frame/RX code; only the test adapter touches SP.
                    unsafe {
                        nixe_fault_probe_enter(&mut frame, executable.as_ptr());
                    }
                    ACTIVE.store(std::ptr::null_mut(), Ordering::Relaxed);
                    assert_eq!(frame.faults, 1);
                    if second && operation == 1 {
                        assert_eq!(
                            frame.gregs[libc::REG_RAX as usize] as u64,
                            0xfeed0123456789ab
                        );
                        assert_eq!(&page[..8], &0xfeed0123456789abu64.to_le_bytes());
                    }
                    assert_eq!(
                        frame.gregs[libc::REG_RIP as usize] as usize,
                        executable.as_ptr() as usize + map.offset as usize
                    );
                    for (value, expected) in map.values.iter().zip(expected) {
                        let actual = match value.location {
                            Location::Unused => panic!("lost prefault value"),
                            Location::Spill { offset } => frame.bytes
                                [offset as usize..offset as usize + value.ty.bytes() as usize]
                                .to_vec(),
                            Location::Register {
                                index,
                                vector: true,
                            } => frame.vectors[index as usize]
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect(),
                            Location::Register {
                                index,
                                vector: false,
                            } => {
                                let slots = [
                                    libc::REG_RAX,
                                    libc::REG_RCX,
                                    libc::REG_RDX,
                                    libc::REG_RBX,
                                    libc::REG_RSP,
                                    libc::REG_RBP,
                                    libc::REG_RSI,
                                    libc::REG_RDI,
                                    libc::REG_R8,
                                    libc::REG_R9,
                                    libc::REG_R10,
                                    libc::REG_R11,
                                    libc::REG_R12,
                                    libc::REG_R13,
                                    libc::REG_R14,
                                    libc::REG_R15,
                                ];
                                frame.gregs[slots[index as usize] as usize]
                                    .to_le_bytes()
                                    .to_vec()
                            }
                        };
                        assert_eq!(
                            actual, expected,
                            "{allocator} operation={operation} second={second} {value:?}"
                        );
                    }
                }
            }
        }
    }
}
