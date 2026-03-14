# torch.export & Kernel Composability Foundation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Lay the foundation for torch.export support and device-side kernel composability in Helion.

**Architecture:** Register Helion's HOP with torch.library for export compatibility. Add a `@helion.device_function` decorator that marks pure-computation helpers for device-side inlining, emitting them as Triton `@triton.jit` helper functions.

**Tech Stack:** PyTorch torch.library, torch.export, Triton JIT, Helion compiler pipeline

---

### Task 1: Register HOP with torch.library for export compatibility

**Files:**
- Modify: `helion/_compiler/_dynamo/higher_order_ops.py`

**What:** Register `helion_kernel_wrapper_mutation` with `torch.library` so torch.export can serialize it by name rather than relying on side-table indices.

### Task 2: Add `@helion.device_function` decorator

**Files:**
- Create: `helion/language/device_function.py`
- Modify: `helion/language/__init__.py`

**What:** A decorator that marks a function as callable from inside `hl.tile()` loops. During codegen, these emit as Triton `@triton.jit` helper functions.

### Task 3: Extend type propagation for device-callable functions

**Files:**
- Modify: `helion/_compiler/type_propagation.py`

**What:** Recognize `@helion.device_function`-decorated callables during tracing and allow them in device context.

### Task 4: Emit Triton helper functions in codegen

**Files:**
- Modify: `helion/_compiler/device_function.py`

**What:** When a device function is called inside a tile loop, emit it as a separate `@triton.jit` function before the main kernel.

### Task 5: Tests and verification

**Files:**
- Create: `test/test_device_function.py`

### Task 6: Commit, push, create PR
</content>
</invoke>