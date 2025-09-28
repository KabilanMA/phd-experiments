from ir.ir import LoopNest, Store, Load, BinaryOp, TENSOR_OP

def emit_c(ir_node, indent=0) -> str:
    space = " " * indent
    if isinstance(ir_node, LoopNest):
        inner = "\n".join([emit_c(stmt, indent + 4) for stmt in ir_node.body])
        return f"{space}for (int {ir_node.var.name}=0; {ir_node.var.name}<N; {ir_node.var.name}++) {{\n{inner}\n{space}}}"
    elif isinstance(ir_node, Store):
        return f"{space}{ir_node.tensor.name}[{','.join(ir_node.indices)}] = {emit_c(ir_node.value)};"
    elif isinstance(ir_node, Load):
        return f"{ir_node.tensor.name}[{','.join(ir_node.indices)}]"
    elif isinstance(ir_node, BinaryOp):
        op_map = {
            TENSOR_OP.ADD: "+",
            TENSOR_OP.SUB: "-",
            TENSOR_OP.MUL: "*",
            TENSOR_OP.DIV: "/"
        }
        return f"({emit_c(ir_node.lhs)} {op_map[ir_node.op]} {emit_c(ir_node.rhs)})"
    else:
        return ""
