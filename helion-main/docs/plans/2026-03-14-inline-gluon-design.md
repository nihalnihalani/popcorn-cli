# `hl.inline_gluon()` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `hl.inline_gluon()` and `hl.gluon_kernel()` escape hatches so power users can embed raw Gluon code (Triton's Blackwell-native API for TMEM, warp specialization, TMA descriptors) inside Helion kernels.

**Architecture:** Mirror `inline_triton` / `triton_kernel` exactly. Add `gl` (triton.gluon) to library_imports so generated code auto-imports it when `gl.*` is referenced. New file `inline_gluon_ops.py` with API + codegen, tests, and docs.

**Tech Stack:** Python 3.10+, Helion DSL, Triton Gluon API, PyTest

---

### Task 1: Add `gl` to library imports

**Files:**
- Modify: `helion/_compiler/output_header.py:17-28`
- Modify: `helion/_compiler/backend.py:597-611`

**Step 1: Add `gl` import to default library_imports in `output_header.py`**

Add after the `tl` line:
```python
"gl": "import triton.gluon as gl",
```

**Step 2: Add `gl` import to TritonBackend.library_imports in `backend.py`**

Add after the `tl` line:
```python
"gl": "import triton.gluon as gl",
```

**Step 3: Verify no conflicts**

Run: `cd helion-main && python -c "from helion._compiler.output_header import library_imports; print('gl' in library_imports)"`
Expected: `True`

---

### Task 2: Create `inline_gluon_ops.py`

**Files:**
- Create: `helion/language/inline_gluon_ops.py`

This file mirrors `inline_triton_ops.py` but for Gluon. It reuses the shared helpers from `inline_triton_ops.py` (`_validate_args`, `_fake_outputs`, `_ensure_name`, `_format_triton_source`, `_parse_triton_source`, `_collect_output_metadata`, `_emit_output_assertions`, `_get_or_add_triton_function_preamble`).

**Step 1: Create the file with `inline_gluon` and `gluon_kernel` functions**

```python
from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Sequence
import inspect
from typing import TYPE_CHECKING
from typing import TypeVar

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from . import _decorators
from .inline_triton_ops import _collect_output_metadata
from .inline_triton_ops import _emit_output_assertions
from .inline_triton_ops import _ensure_name
from .inline_triton_ops import _fake_outputs
from .inline_triton_ops import _format_triton_source
from .inline_triton_ops import _get_or_add_triton_function_preamble
from .inline_triton_ops import _parse_triton_source
from .inline_triton_ops import _validate_args

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T")

__all__ = ["inline_gluon", "gluon_kernel"]


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def inline_gluon(
    gluon_source: str,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Inline a raw Gluon snippet inside a Helion kernel.

    Gluon is Triton's lower-level API for Blackwell (SM100+) GPUs,
    providing direct access to TMEM, explicit warp specialization,
    and TMA descriptors.

    Args:
        gluon_source: The Gluon code snippet using ``gl.*`` primitives.
            The last statement must be an expression representing the
            return value. Common indentation is stripped automatically.
        args: Positional or keyword placeholders that will be substituted
            via ``str.format`` before code generation.
        output_like: Example tensors describing the expected outputs.
            A single tensor indicates a single output; a tuple/list of
            tensors indicates multiple outputs. Pass ``None`` for
            side-effect-only snippets.

    Returns:
        The value(s) produced by the snippet. Matches the structure of
        ``output_like``.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(inline_gluon)
def _(
    gluon_source: str,
    args: object,
    output_like: object,
) -> object:
    if not isinstance(gluon_source, str):
        raise exc.InvalidAPIUsage(
            f"gluon_source must be a string, got {type(gluon_source)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


@_decorators.codegen(inline_gluon, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    gluon_source = state.proxy_arg(0)
    args_obj = state.proxy_arg(1)
    output_like = state.proxy_arg(2)

    if not isinstance(gluon_source, str):
        raise exc.InvalidAPIUsage(
            f"gluon_source must be a string, got {type(gluon_source)}"
        )

    formatted = _format_triton_source(
        state,
        gluon_source,
        args_obj,
        state.ast_args[1],
    )

    statements, result_expr = _parse_triton_source(
        formatted, require_expression=output_like is not None
    )
    for stmt in statements:
        state.add_statement(stmt)

    if output_like is None:
        if result_expr is not None:
            state.add_statement(create(ast.Expr, value=result_expr))
        return create(ast.Constant, value=None)

    result_name = state.device_function.new_var("inline_gluon_result")
    assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=result_name, ctx=ast.Store())],
        value=result_expr,
    )
    state.add_statement(assign)

    dtypes, output_nodes, is_multi = _collect_output_metadata(
        output_like, state.ast_args[2]
    )
    _emit_output_assertions(state, result_name, dtypes, output_nodes, is_multi)

    if is_multi:
        return [expr_from_string(f"{result_name}[{i}]") for i in range(len(dtypes))]

    return expr_from_string(result_name)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def gluon_kernel(
    gluon_source_or_fn: object,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Define (once) and call a @triton.jit function using Gluon primitives.

    Similar to ``triton_kernel()`` but intended for functions that use
    Gluon primitives (``gl.*``).  The function is emitted at module scope
    once, then invoked from the kernel body.

    Args:
        gluon_source_or_fn: Source for a single @triton.jit function
            definition using Gluon primitives, or a Python function object.
        args: Positional or keyword placeholders.
        output_like: Example tensor(s) describing the expected outputs.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(gluon_kernel)
def _(
    gluon_source_or_fn: object,
    args: object,
    output_like: object,
) -> object:
    if not (
        isinstance(gluon_source_or_fn, str)
        or inspect.isfunction(gluon_source_or_fn)
    ):
        raise exc.InvalidAPIUsage(
            f"gluon_kernel expects a string source or a function, got {type(gluon_source_or_fn)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


@_decorators.codegen(gluon_kernel, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from collections.abc import Mapping as MappingType
    from typing import cast

    gluon_source_or_fn = state.proxy_arg(0)
    args_obj = state.proxy_arg(1)
    output_like = state.proxy_arg(2)

    if not (
        isinstance(gluon_source_or_fn, str)
        or inspect.isfunction(gluon_source_or_fn)
    ):
        raise exc.InvalidAPIUsage(
            f"gluon_kernel expects a string source or a function, got {type(gluon_source_or_fn)}"
        )
    _validate_args(args_obj)

    fn_name = _get_or_add_triton_function_preamble(state, gluon_source_or_fn)

    call_args_src = ""
    if isinstance(state.ast_args[1], dict):
        kw_pairs: list[str] = []
        mapping = cast("MappingType[str, object]", args_obj)
        for key, node in state.ast_args[1].items():
            kw_pairs.append(f"{key}=" + _ensure_name(state, node, mapping[key]))
        call_args_src = ", ".join(kw_pairs)
    else:
        if not isinstance(state.ast_args[1], (ast.List, ast.Tuple, list, tuple)):
            raise exc.InvalidAPIUsage(
                "gluon_kernel expects a literal list/tuple for positional args"
            )
        arg_nodes = (
            state.ast_args[1].elts
            if isinstance(state.ast_args[1], (ast.List, ast.Tuple))
            else list(state.ast_args[1])
        )
        names = [
            _ensure_name(state, node, arg)
            for node, arg in zip(
                arg_nodes, cast("Sequence[object]", args_obj), strict=False
            )
        ]
        call_args_src = ", ".join(names)

    call_expr = expr_from_string(f"{fn_name}({call_args_src})")

    if output_like is None:
        state.add_statement(create(ast.Expr, value=call_expr))
        return create(ast.Constant, value=None)

    result_name = state.device_function.new_var("gluon_kernel_result")
    assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=result_name, ctx=ast.Store())],
        value=call_expr,
    )
    state.add_statement(assign)

    dtypes, output_nodes, is_multi = _collect_output_metadata(
        output_like, state.ast_args[2]
    )
    _emit_output_assertions(state, result_name, dtypes, output_nodes, is_multi)

    if is_multi:
        return [expr_from_string(f"{result_name}[{i}]") for i in range(len(dtypes))]
    return expr_from_string(result_name)
```

---

### Task 3: Export from `helion/language/__init__.py`

**Files:**
- Modify: `helion/language/__init__.py:23-24`

**Step 1: Add exports after the inline_triton imports**

Add after line 24:
```python
from .inline_gluon_ops import gluon_kernel as gluon_kernel
from .inline_gluon_ops import inline_gluon as inline_gluon
```

---

### Task 4: Create test file

**Files:**
- Create: `test/test_inline_gluon.py`

Tests mirror `test_inline_triton.py`. Since Gluon requires SM100+ hardware (Blackwell) which CI may not have, tests that compile/run kernels should skip gracefully. Tests for validation (invalid args, etc.) work without hardware.

The key insight: `inline_gluon` generates the same Triton code as `inline_triton` but with `gl.*` imports available. So we can test code generation (`.to_triton_code()`) without needing Blackwell hardware, and test validation errors directly.

```python
from __future__ import annotations

import torch

from helion.runtime.settings import _get_backend

if _get_backend() in ("triton", "tileir"):
    import triton

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestInlineGluon(RefEagerTestDisabled, TestCase):
    def test_inline_gluon_simple(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                result = hl.inline_gluon(
                    """
                    tmp = {lhs} + {rhs}
                    tmp
                    """,
                    args={"lhs": x_val, "rhs": y_val},
                    output_like=x_val,
                )
                out[tile] = result
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(kernel, (x, y))
        torch.testing.assert_close(result, x + y)

    def test_inline_gluon_multi_output(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sum_out = torch.empty_like(a)
            diff_out = torch.empty_like(a)
            for tile in hl.tile(a.shape):
                a_val = a[tile]
                b_val = b[tile]
                sum_val, diff_val = hl.inline_gluon(
                    """
                    sum_val = {0} + {1}
                    diff_val = {0} - {1}
                    sum_val, diff_val
                    """,
                    args=(a_val, b_val),
                    output_like=(a_val, a_val),
                )
                sum_out[tile] = sum_val
                diff_out[tile] = diff_val
            return sum_out, diff_out

        a = torch.randn(64, device=DEVICE, dtype=torch.float32)
        b = torch.randn_like(a)
        code, (sum_result, diff_result) = code_and_output(kernel, (a, b))
        torch.testing.assert_close(sum_result, a + b)
        torch.testing.assert_close(diff_result, a - b)

    def test_inline_gluon_list_args_reuse(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                out[tile] = hl.inline_gluon(
                    """
                    triple = {0} + {0} + {0}
                    triple + {1}
                    """,
                    args=[x_val, y_val],
                    output_like=x_val,
                )
            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, out = code_and_output(kernel, (x, y))
        torch.testing.assert_close(out, 3 * x + y)

    def test_inline_gluon_invalid_output_like(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_gluon(
                    "{0}\n",
                    args=(x_val,),
                    output_like="not a tensor",
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_gluon_invalid_mapping_key(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_gluon(
                    "{bad}\n",
                    args={0: x_val},
                    output_like=x_val,
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_gluon_side_effect_only(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            flag = torch.zeros(1, device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.shape):
                val = x[tile]
                _ = hl.inline_gluon(
                    "tl.store({0}, {1}[0])",
                    args=(flag, val),
                    output_like=None,
                )
            return flag

        x = torch.randn(1, device=DEVICE, dtype=torch.float32)
        bound = kernel.bind((x,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("tl.store(", code)

    def test_inline_gluon_none_output_allows_terminal_statement(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(grad_x_lock: torch.Tensor) -> torch.Tensor:
            for _ in hl.tile(grad_x_lock.shape):
                hl.inline_gluon(
                    """
                    while tl.atomic_cas({0} + {1}, 0, 1) == 1:
                        pass
                    """,
                    args=(grad_x_lock, 0),
                    output_like=None,
                )
            return grad_x_lock

        grad_x_lock = torch.ones(4, device=DEVICE, dtype=torch.int32)
        bound = kernel.bind((grad_x_lock,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("while tl.atomic_cas", code)
        self.assertNotIn("_host_tensor", code)
```

---

### Task 5: Update documentation

**Files:**
- Modify: `docs/api/language.md:144-159`

Add after the `triton_kernel()` section:

```markdown
### inline_gluon()

```{eval-rst}
.. autofunction:: inline_gluon
```

Embeds Gluon code snippets (Triton's lower-level API for Blackwell GPUs) inside a Helion kernel. Works identically to ``inline_triton()`` but intended for code using ``gl.*`` primitives for TMEM, warp specialization, and TMA descriptors. Requires ``triton.gluon`` to be available.

### gluon_kernel()

```{eval-rst}
.. autofunction:: gluon_kernel
```

Define (once) and call a ``@triton.jit`` function that uses Gluon primitives from Helion device code. Works identically to ``triton_kernel()`` but intended for Gluon-based functions.
```

---

### Task 6: Lint and verify

**Step 1:** Run `cd helion-main && ./lint.sh`
**Step 2:** Run `cd helion-main && pytest test/test_inline_gluon.py -x -vv -s`
**Step 3:** Fix any issues found

---

### Task 7: Commit and create PR

Commit all changes and create PR to pytorch/helion with `[hackathon]` prefix.
