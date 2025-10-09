from stella.ir.ir import LoopNest, Store, Load, BinaryOp, TENSOR_OP
import sys

def emit_c_for_loop(ir_node, indent=0) -> str:
    space = " " * indent
    if not isinstance(ir_node, LoopNest):
        print("[Error] Error generating code for {ir_node}")
        sys.exit(1)
    else:
        inner = "\n".join([emit_c(stmt, indent+4) for stmt in ir_node.body])
        return f"{space}for (int {ir_node.var.name}=0; {ir_node.var.name}<{ir_node.maximum_limit}; {ir_node.var.name}++) {{\n{inner}\n{space}}}"

def emit_c_tensor_store(ir_node, indent=0) ->  str:
    space = " " * indent
    if not isinstance(ir_node, Store):
        print(f"[Error] Error generating code for {ir_node}")
        sys.exit(1)
    else:
        return f"{space}{ir_node.tensor}[{ir_node.index}] = {emit_c(ir_node.value)};"

def emit_c_tensor_load(ir_node, indent=0) -> str:
    space = " " * indent
    if not isinstance(ir_node, Load):
        print(f"[Error] Error generating code for {ir_node}")
        sys.exit(1)
    else:
        return f"{space}{ir_node.tensor.name}[{ir_node.index}]"

def emit_c_tensor_binary_op(ir_node, indent=0) -> str:
    space = " " * indent
    if not isinstance(ir_node, BinaryOp):
        print(f"[Error] Error generating code for {ir_node}")
        sys.exit(1)
    else:
        op_map = {
            TENSOR_OP.ADD: "+",
            TENSOR_OP.SUB: "-",
            TENSOR_OP.MUL: "*",
            TENSOR_OP.DIV: "/"
        }
        return f"({emit_c(ir_node.lhs)} {op_map[ir_node.op]} {emit_c(ir_node.rhs)})"


def emit_c(ir_node, indent=0) -> str:
    space = " " * indent
    if isinstance(ir_node, LoopNest):
        return emit_c_for_loop(ir_node, indent)
        # inner = "\n".join([emit_c(stmt, indent + 4) for stmt in ir_node.body])
        # return f"{space}for (int {ir_node.var.name}=0; {ir_node.var.name}<N; {ir_node.var.name}++) {{\n{inner}\n{space}}}"
    elif isinstance(ir_node, Store):
        print("ASASA")
        print(type(ir_node))
        print(ir_node.tensor)
        print("ASASA")
        print(ir_node)
        print("ASASA")
        return emit_c_tensor_store(ir_node, indent)
        # return f"{space}{ir_node.tensor.name}[{','.join(ir_node.indices)}] = {emit_c(ir_node.value)};"
    elif isinstance(ir_node, Load):
        return emit_c_tensor_load(ir_node, indent)
        # return f"{ir_node.tensor.name}[{','.join(ir_node.indices)}]"
    elif isinstance(ir_node, BinaryOp):
        return emit_c_tensor_binary_op(ir_node, indent)
        # op_map = {
        #     TENSOR_OP.ADD: "+",
        #     TENSOR_OP.SUB: "-",
        #     TENSOR_OP.MUL: "*",
        #     TENSOR_OP.DIV: "/"
        # }
        # return f"({emit_c(ir_node.lhs)} {op_map[ir_node.op]} {emit_c(ir_node.rhs)})"
    else:
        return ""
