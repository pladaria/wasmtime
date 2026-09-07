use super::tests::{compile, target};
use crate::Context;
use crate::cursor::{Cursor, FuncCursor};
use crate::inst_predicates::{
    has_observable_fp_effect, is_mergeable_for_egraph, is_pure_for_egraph,
};
use crate::ir::{self, InstBuilder, Opcode, types};
use alloc::{string::ToString, vec::Vec};
use cranelift_control::ControlPlane;

#[test]
fn unsigned_vector_bias_sequences_clear_zero_sign_only_for_observable_fp() {
    for allocator in ["single_pass", "backtracking"] {
        let isa = target("x86_64-unknown-linux-gnu", allocator, true);
        for widened in [false, true] {
            for observable in [false, true] {
                let mut f = ir::Function::new();
                f.nixe_observable_fp = observable;
                let block = f.dfg.make_block();
                f.layout.append_block(block);
                let mut c = FuncCursor::new(&mut f).at_bottom(block);
                let ptr = c.ins().get_pinned_reg(types::I64);
                let input = c.ins().load(
                    if widened { types::I32X4 } else { types::I64X2 },
                    ir::MemFlagsData::trusted(),
                    ptr,
                    0,
                );
                let input = if widened {
                    c.ins().uwiden_low(input)
                } else {
                    input
                };
                let value = c.ins().fcvt_from_uint(types::F64X2, input);
                c.ins().nixe_exit(1, &[value]);
                let code = compile(f, &*isa).unwrap();
                let decoder = isa.to_capstone().unwrap();
                let exit = code.buffer.nixe_states.last().unwrap();
                let instructions = decoder
                    .disasm_all(&code.code_buffer()[..exit.offset as usize], 0)
                    .unwrap();
                // The final sign mask is only needed with dynamic rounding.
                // Exclude constant-pool data and exit padding from decoding.
                let last = instructions
                    .iter()
                    .filter(|i| i.mnemonic() != Some("nop"))
                    .last()
                    .unwrap();
                assert_eq!(
                    last.mnemonic() == Some("pand"),
                    observable,
                    "{allocator}, widened={widened}: {instructions}"
                );
            }
        }
    }
}

fn discarded_conversions(observable: bool) -> ir::Function {
    let mut f = ir::Function::new();
    f.nixe_observable_fp = observable;
    let block = f.dfg.make_block();
    f.layout.append_block(block);
    let mut c = FuncCursor::new(&mut f).at_bottom(block);
    let integer = c.ins().iconst(types::I64, (1 << 24) + 1);
    c.ins().fcvt_from_sint(types::F32, integer);
    // A real status/control boundary may be supplied by the owner here.
    c.ins().nixe_state(1, &[]);
    c.ins().fcvt_from_sint(types::F32, integer);
    c.ins().nixe_exit(2, &[]);
    f
}

#[test]
fn discarded_fp_is_emitted_once_per_operation_with_both_allocators() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        // These settings exercise both opt_level=none and the egraph at speed.
        for allocator in ["single_pass", "backtracking"] {
            let isa = target(triple, allocator, true);
            for observable in [false, true] {
                let code = compile(discarded_conversions(observable), &*isa).unwrap();
                let decoder = isa.to_capstone().unwrap();
                let instructions = decoder.disasm_all(code.code_buffer(), 0).unwrap();
                let mnemonic = if triple.starts_with("x86") {
                    "cvtsi2ss"
                } else {
                    "scvtf"
                };
                let conversions: Vec<_> = instructions
                    .iter()
                    .filter(|i| i.mnemonic().is_some_and(|m| m.starts_with(mnemonic)))
                    .map(|i| i.address())
                    .collect();
                assert_eq!(
                    conversions.len(),
                    if observable { 2 } else { 0 },
                    "{triple} {allocator} observable={observable}: {instructions}"
                );
                if observable {
                    let boundary = code.buffer.nixe_states.iter().find(|s| s.id == 1).unwrap();
                    assert!(conversions[0] < u64::from(boundary.offset));
                    assert!(conversions[1] >= u64::from(boundary.offset));
                }
            }
        }
    }
}

#[test]
fn fp_effects_are_opaque_to_value_rewrites_and_cse() {
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let isa = target(triple, "backtracking", true);
        let mut f = discarded_conversions(true);
        let block = f.layout.entry_block().unwrap();
        let exit = f.layout.last_inst(block).unwrap();
        let mut c = FuncCursor::new(&mut f).at_inst(exit);
        let x = c.ins().f64const(1.5);
        let y = c.ins().f64const(0.5);
        c.ins().fadd(x, y);
        c.ins().fadd(x, y); // Identical values still have distinct effects.
        let sub = c.ins().fsub(x, y);
        let neg = c.ins().fneg(sub); // Do not rewrite to a reversed subtraction.
        c.ins().fmul(x, y);
        c.ins().fdiv(x, y);
        c.ins().sqrt(x);
        c.ins().fma(x, y, x);
        c.ins().fmin(x, y);
        c.ins().fmax(x, y);
        c.ins().fcmp(ir::condcodes::FloatCC::Equal, x, y);
        c.ins().ceil(x);
        c.ins().floor(x);
        c.ins().trunc(x);
        c.ins().nearest(x);
        let narrow = c.ins().fdemote(types::F32, x);
        c.ins().fpromote(types::F64, narrow);
        c.ins().fcvt_to_sint(types::I64, x);
        c.ins().fcvt_to_uint(types::I64, x);
        c.ins().fcvt_to_sint_sat(types::I64, x);
        c.ins().fcvt_to_uint_sat(types::I64, x);
        let integer = c.ins().iconst(types::I64, (1 << 53) + 1);
        c.ins().fcvt_from_uint(types::F64, integer);
        let vector = c.ins().splat(types::F64X2, x);
        c.ins().fadd(vector, vector);
        c.ins()
            .fcmp(ir::condcodes::FloatCC::LessThan, vector, vector);
        let small_vector = c.ins().fvdemote(vector);
        c.ins().fvpromote_low(small_vector);
        // The contract must not disable unrelated constant folding.
        let two = c.ins().iconst(types::I64, 2);
        let sum = c.ins().iadd(two, two);
        c.func.replace(exit).nixe_exit(2, &[neg, sum]);
        let effects = |f: &ir::Function| -> Vec<Opcode> {
            f.layout
                .blocks()
                .flat_map(|b| f.layout.block_insts(b))
                .filter(|&i| has_observable_fp_effect(f, i))
                .map(|i| {
                    assert!(!is_pure_for_egraph(f, i));
                    assert!(!is_mergeable_for_egraph(f, i));
                    f.dfg.insts[i].opcode()
                })
                .collect()
        };
        let before = effects(&f);
        let mut cx = Context::for_function(f);
        cx.optimize(&*isa, &mut ControlPlane::default()).unwrap();
        assert_eq!(effects(&cx.func), before, "{}", cx.func.display());
        assert!(
            !cx.func
                .layout
                .blocks()
                .flat_map(|b| cx.func.layout.block_insts(b))
                .any(|i| cx.func.dfg.insts[i].opcode() == Opcode::Iadd)
        );
        cx.clear();
        assert!(!cx.func.nixe_observable_fp);
    }
}

#[test]
fn observable_fp_rejects_nan_canonicalization() {
    let isa = target("x86_64-unknown-linux-gnu", "backtracking", true);
    let mut cx = Context::for_function(discarded_conversions(true));
    let before = cx.func.clone();
    let error = cx.canonicalize_nans(&*isa).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("incompatible with NaN canonicalization")
    );
    assert_eq!(cx.func, before);
}

#[test]
fn observable_comparison_is_not_absorbed_into_min_selection() {
    for allocator in ["single_pass", "backtracking"] {
        let isa = target("x86_64-unknown-linux-gnu", allocator, true);
        let mut f = ir::Function::new();
        f.nixe_observable_fp = true;
        let block = f.dfg.make_block();
        f.layout.append_block(block);
        let mut c = FuncCursor::new(&mut f).at_bottom(block);
        let x = c.ins().f64const(1.5);
        let y = c.ins().f64const(0.5);
        let less = c.ins().fcmp(ir::condcodes::FloatCC::LessThan, x, y);
        let selected = c.ins().select(less, x, y);
        c.ins().nixe_exit(1, &[selected]);
        let code = compile(f, &*isa).unwrap();
        let decoder = isa.to_capstone().unwrap();
        let instructions = decoder.disasm_all(code.code_buffer(), 0).unwrap();
        // The ordinary x86 select/fcmp -> minsd pattern preserves values in
        // CLIF's domain, but not the FP exception behavior of the comparison.
        assert!(
            !instructions.iter().any(|i| i.mnemonic() == Some("minsd")),
            "{instructions}"
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|i| i.mnemonic() == Some("ucomisd"))
                .count(),
            1,
            "{instructions}"
        );
    }
}

#[test]
fn inlining_does_not_silently_change_the_fp_contract() {
    struct Inliner(ir::Function);
    impl crate::inline::Inline for Inliner {
        fn inline(
            &mut self,
            _: &ir::Function,
            _: ir::Inst,
            _: Opcode,
            _: ir::FuncRef,
            _: &[ir::Value],
        ) -> crate::inline::InlineCommand<'_> {
            crate::inline::InlineCommand::Inline {
                callee: alloc::borrow::Cow::Borrowed(&self.0),
                visit_callee: false,
            }
        }
    }
    for caller_mode in [false, true] {
        for callee_mode in [false, true] {
            let mut callee = ir::Function::new();
            callee.nixe_observable_fp = callee_mode;
            let block = callee.dfg.make_block();
            callee.layout.append_block(block);
            FuncCursor::new(&mut callee)
                .at_bottom(block)
                .ins()
                .return_(&[]);
            let mut caller = ir::Function::new();
            caller.nixe_observable_fp = caller_mode;
            let signature = caller.import_signature(callee.signature.clone());
            let func = caller.import_function(ir::ExtFuncData {
                name: ir::ExternalName::testcase("callee"),
                signature,
                colocated: false,
                patchable: false,
            });
            let block = caller.dfg.make_block();
            caller.layout.append_block(block);
            let mut c = FuncCursor::new(&mut caller).at_bottom(block);
            c.ins().call(func, &[]);
            c.ins().return_(&[]);
            let mut cx = Context::for_function(caller);
            let result = cx.inline(Inliner(callee));
            if caller_mode == callee_mode {
                assert!(result.unwrap());
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("different Nixe FP")
                );
            }
            assert_eq!(cx.func.nixe_observable_fp, caller_mode);
        }
    }
}
