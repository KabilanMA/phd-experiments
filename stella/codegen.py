import sys
import re
import argparse
from textwrap import dedent

def parse_einsum(spec: str):
    # spec: "i,j->ij" or "ij,jk->ik"
    lhs, rhs = spec.split("->")
    inputs = lhs.split(",")
    output= rhs
    return inputs, output

def unique_indices_in_order(output):
    seen = []
    for ch in output:
        if ch not in seen:
            seen.append(ch)
    return seen

def indices_from_operand(op):
    return list(op)

def make_strides_code(operand_idx, operand_indices, out_index_order, dims_map, var_prefix):
    """
    Build expression to compute linear offset for operand with given index order,
    assuming row-major, i.e., the rightmost index varies fastest.
    We'll compute stride for each index based on dims_map and the operand's index order.
    """
    # operand_indices is like ['i','j'] for "ij"
    # For row-major, stride for index k (within operand order) = product of sizes of indices to the right
    parts = []
    for pos, idx in enumerate(operand_indices):
        # compute multiplier (product of dims to right)
        mul_factors = []
        for right in operand_indices[pos+1:]:
            mul_factors.append(dims_map[right])
        if mul_factors:
            mul = " * ".join(mul_factors)
            parts.append(f"{idx} * ({mul})")
        else:
            # rightmost index => multiplier 1
            parts.append(f"{idx}")
    if not parts:
        return "0"
    return " + ".join(parts)

def gen_c_function(einsum_spec, in_names, out_name, dims_map, func_name="compute"):
    inputs, output = parse_einsum(einsum_spec)
    # collect indices in output order to build loops
    loop_indices = unique_indices_in_order(output)
    # validate dims_map keys
    for idx in loop_indices:
        if idx not in dims_map:
            raise ValueError(f"Missing size for index '{idx}' in dims_map")

    # build function signature
    # pointers: inputs then output
    all_ptrs = ", ".join([f"double *{n}" for n in in_names] + [f"double *{out_name}"])
    # dims are integers in order of loop_indices
    dim_args = ", ".join([f"int {idx}_len /* {idx} */" for idx in loop_indices])
    signature = f"void {func_name}({all_ptrs}, {dim_args})"

    # build loop header
    loops = []
    for idx in loop_indices:
        loops.append(f"for (int {idx} = 0; {idx} < {idx}_len; ++{idx}) {{")
    loops_close = "}\n" * len(loop_indices)

    # compute offsets for each input operand
    operand_offsets = []
    for op_idx, op in enumerate(inputs):
        idxs = indices_from_operand(op)
        # compute linear offset for this operand according to its own index order
        # row-major: rightmost index varies fastest
        offset_terms = []
        for pos, idx in enumerate(idxs):
            right_dims = idxs[pos+1:]
            if right_dims:
                right_mul = " * ".join([f"{r}_len" for r in right_dims])
                offset_terms.append(f"{idx} * ({right_mul})")
            else:
                offset_terms.append(f"{idx}")
        offset_expr = " + ".join(offset_terms) if offset_terms else "0"
        operand_offsets.append(offset_expr)

    # compute output offset expression similarly
    out_idxs = indices_from_operand(output)
    out_terms = []
    for pos, idx in enumerate(out_idxs):
        right_dims = out_idxs[pos+1:]
        if right_dims:
            right_mul = " * ".join([f"{r}_len" for r in right_dims])
            out_terms.append(f"{idx} * ({right_mul})")
        else:
            out_terms.append(f"{idx}")
    out_offset = " + ".join(out_terms) if out_terms else "0"

    # generate the body expression for the operation: here we support simple product of inputs
    # generalization: determine element-wise expression based on operands; start with product.
    multiplicands = [f"{in_names[i]}[{operand_offsets[i]}]" for i in range(len(in_names))]
    rhs_expr = " * ".join(multiplicands)

    # assemble C code
    code = dedent(f"""
    #include <stdio.h>
    #include <stdlib.h>

    {signature} {{
    """)
    # add loops
    for l in loops:
        code += "    " + l + "\n"
    # inside body: compute output offset and assign
    code += "        int out_idx = " + out_offset + ";\n"
    for i, off in enumerate(operand_offsets):
        code += f"        int off_{i} = {off};\n"
    code += f"        {out_name}[out_idx] = {rhs_expr};\n"
    # close loops
    code += "    " + loops_close
    code += "}\n"
    return code

def main():
    parser = argparse.ArgumentParser(description="Tiny einsum -> C generator (starter).")
    parser.add_argument("einsum", help="einsum spec, e.g. 'i,j->ij' or 'ij,jk->ik'")
    parser.add_argument("inputs", nargs='+', help="input variable names (space separated). Last one is output name if you prefer; we'll ask)")
    parser.add_argument("--out", default="C", help="output array name (default: C)")
    parser.add_argument("--dims", nargs='+', help="list of sizes for indices in order (e.g. M N for i j). Provide as names like M N or numeric", required=True)
    parser.add_argument("--func", default="compute", help="C function name")
    args = parser.parse_args()

    # Basic mapping: map indices in output to dims argument order
    inputs_spec, output_spec = parse_einsum(args.einsum)
    idx_order = unique_indices_in_order(output_spec)
    if len(idx_order) != len(args.dims):
        print("ERROR: number of --dims must match the number of distinct indices in the output (order matters).")
        print("Output indices:", idx_order)
        sys.exit(1)
    dims_map = {idx_order[i]: args.dims[i] for i in range(len(idx_order))}

    # generate
    ccode = gen_c_function(args.einsum, args.inputs, args.out, dims_map, func_name=args.func)
    print(ccode)

if __name__ == "__main__":
    main()