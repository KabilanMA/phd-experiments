from stella.ir.ir import EinsumOp, LoopVar, LoopNest, Load, Store, BinaryOp

def lower_to_loops(op: EinsumOp):
    """
    Convert EinsumOp (high-level IR) into nested loops (low-level IR).
    For now, this is just a stub that demonstrates the structure.
    """
    i = LoopVar("i")
    j = LoopVar("j")
    body = [
        Store(op.output.csr_values, ("nnz"), 
        BinaryOp(op.operator,
            Load(op.inputs[0], ("i", "j")),
            Load(op.inputs[1], ("j", "i"))
        )
        )
    ]
    return LoopNest(i, [LoopNest(j, body, "A")], "VC")
