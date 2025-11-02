from stella.ir.ir import TensorOp, Storage, Tensor

import random
import string

def generate_deterministic_string(length: int, char_seed: str = None) -> str:
    """
    Generates a deterministic random string of a specified length.

    If char_seed is provided, the string is deterministic based on (length, char_seed).
    If char_seed is None, the string is deterministic based only on (length).

    Args:
        length: The desired length of the string.
        char_seed: A single character used to set the random seed (optional).

    Returns:
        A randomly generated string that is deterministic.
    """
    # 1. Determine the seed_value based on the presence of char_seed
    if char_seed is None:
        # Deterministic based only on length
        seed_value = length
    else:
        # Deterministic based on both char_seed and length
        seed_value = ord(char_seed) * length

    # 2. Set the random seed
    random.seed(seed_value)

    # 3. Define the character pool
    char_pool = string.ascii_letters + string.digits

    # 4. Generate the string
    random_string = ''.join(random.choice(char_pool) for _ in range(length))

    # 5. Reset the seed (Good practice)
    random.seed(None)

    return random_string

def _two_tensor_emit(iterating_tensor: Tensor, searching_tensor: Tensor, output_tensor: Tensor):
    found_indices = []
    search_indices = []
    loops = []
    temp_print_loop = "int nnz = 0;\n"
    for iter_axis in iterating_tensor.axis:
        if iter_axis.axis_name in found_indices:
            continue
        else:
            if (iter_axis.sparsity == Storage.DENSE):
                temp_loop = f"for (int {iter_axis.axis_name} = 0; {iter_axis.axis_name} < {iter_axis.size}; {iter_axis.axis_name}++)\n"
                found_indices.append(iter_axis.axis_name)
                loops.append(temp_loop)
                temp_print_loop += temp_loop
            elif (iter_axis.sparsity == Storage.SPARSE):
                temp_loop = f"for (int {iter_axis.axis_name}{generate_deterministic_string(5, iter_axis.axis_name)} = {iter_axis.tensor_name}.pos[{found_indices[-1]}]; {iter_axis.axis_name}{generate_deterministic_string(5, iter_axis.axis_name)} < {iter_axis.tensor_name}.pos[{found_indices[-1]}+1]; {iter_axis.axis_name}{generate_deterministic_string(5, iter_axis.axis_name)}++)\n\tint {iter_axis.axis_name} = {iter_axis.tensor_name}.idx[{iter_axis.axis_name}{generate_deterministic_string(5, iter_axis.axis_name)}];\n"
                found_indices.append(iter_axis.axis_name)
                loops.append(temp_loop)
                temp_print_loop += temp_loop
    
    for i, sear_axis in enumerate(searching_tensor.axis):
        if sear_axis.axis_name in found_indices:
            # We already have the index, then search for it
            if (sear_axis.sparsity == Storage.DENSE):
                loop_iterator_name = f"{sear_axis.axis_name}{generate_deterministic_string(len(found_indices))}"
                temp_loop = f"for (int {loop_iterator_name}= {sear_axis.tensor_name}.pos[{sear_axis.axis_name}]; {loop_iterator_name} < {sear_axis.tensor_name}.pos[{sear_axis.axis_name}+1]; {loop_iterator_name}++)\n"
                loops.append(temp_loop)
                search_indices.append(generate_deterministic_string(len(found_indices)))
                temp_print_loop += temp_loop
            elif (sear_axis.sparsity == Storage.SPARSE):
                # TODO: 

                if (i == len(searching_tensor.axis)-1):
                    temp_print_loop += f"\tif ({sear_axis.tensor_name}.idx[{search_indices[-1]}] == {sear_axis.axis_name})" + " {\n\t" + f"{output_tensor.name}.idx[nnz] = {found_indices[-1]};\n\t{output_tensor.name}.value[nnz] = {iterating_tensor.name}.val[{generate_deterministic_string(5, found_indices[-1])}] * {searching_tensor.name}.val[{search_indices[-1]}];\n\tnnz++;"
    
    print(temp_print_loop)

def emit_c(tensor_op: TensorOp, indent=0) -> str:
    space = " " * indent
    input_tensors = tensor_op.inputs
    output_tensor = tensor_op.output
    if (len(input_tensors) == 2):
        _two_tensor_emit(input_tensors[0], input_tensors[1], output_tensor)

# from stella.ir.ir import LoopNest, Store, Load, BinaryOp, TENSOR_OP
# import sys

# def emit_c_for_loop(ir_node, indent=0) -> str:
#     space = " " * indent
#     if not isinstance(ir_node, LoopNest):
#         print("[Error] Error generating code for {ir_node}")
#         sys.exit(1)
#     else:
#         inner = "\n".join([emit_c(stmt, indent+4) for stmt in ir_node.body])
#         return f"{space}for (int {ir_node.var.name}=0; {ir_node.var.name}<{ir_node.maximum_limit}; {ir_node.var.name}++) {{\n{inner}\n{space}}}"

# def emit_c_tensor_store(ir_node, indent=0) ->  str:
#     space = " " * indent
#     if not isinstance(ir_node, Store):
#         print(f"[Error] Error generating code for {ir_node}")
#         sys.exit(1)
#     else:
#         return f"{space}{ir_node.tensor}[{ir_node.index}] = {emit_c(ir_node.value)};"

# def emit_c_tensor_load(ir_node, indent=0) -> str:
#     space = " " * indent
#     if not isinstance(ir_node, Load):
#         print(f"[Error] Error generating code for {ir_node}")
#         sys.exit(1)
#     else:
#         return f"{space}{ir_node.tensor.name}[{ir_node.index}]"

# def emit_c_tensor_binary_op(ir_node, indent=0) -> str:
#     space = " " * indent
#     if not isinstance(ir_node, BinaryOp):
#         print(f"[Error] Error generating code for {ir_node}")
#         sys.exit(1)
#     else:
#         op_map = {
#             TENSOR_OP.ADD: "+",
#             TENSOR_OP.SUB: "-",
#             TENSOR_OP.MUL: "*",
#             TENSOR_OP.DIV: "/"
#         }
#         return f"({emit_c(ir_node.lhs)} {op_map[ir_node.op]} {emit_c(ir_node.rhs)})"


# def emit_c(ir_node, indent=0) -> str:
#     space = " " * indent
#     if isinstance(ir_node, LoopNest):
#         return emit_c_for_loop(ir_node, indent)
#         # inner = "\n".join([emit_c(stmt, indent + 4) for stmt in ir_node.body])
#         # return f"{space}for (int {ir_node.var.name}=0; {ir_node.var.name}<N; {ir_node.var.name}++) {{\n{inner}\n{space}}}"
#     elif isinstance(ir_node, Store):
#         print("ASASA")
#         print(type(ir_node))
#         print(ir_node.tensor)
#         print("ASASA")
#         print(ir_node)
#         print("ASASA")
#         return emit_c_tensor_store(ir_node, indent)
#         # return f"{space}{ir_node.tensor.name}[{','.join(ir_node.indices)}] = {emit_c(ir_node.value)};"
#     elif isinstance(ir_node, Load):
#         return emit_c_tensor_load(ir_node, indent)
#         # return f"{ir_node.tensor.name}[{','.join(ir_node.indices)}]"
#     elif isinstance(ir_node, BinaryOp):
#         return emit_c_tensor_binary_op(ir_node, indent)
#         # op_map = {
#         #     TENSOR_OP.ADD: "+",
#         #     TENSOR_OP.SUB: "-",
#         #     TENSOR_OP.MUL: "*",
#         #     TENSOR_OP.DIV: "/"
#         # }
#         # return f"({emit_c(ir_node.lhs)} {op_map[ir_node.op]} {emit_c(ir_node.rhs)})"
#     else:
#         return ""
