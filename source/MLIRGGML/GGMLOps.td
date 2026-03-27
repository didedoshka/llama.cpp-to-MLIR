#ifndef GGML_OPS
#define GGML_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GGML_Dialect : Dialect {
  let name = "ggml";
  let cppNamespace = "::mlir::ggml";
}

class GGML_Op<string mnemonic, list<Trait> traits = []> :
    Op<GGML_Dialect, mnemonic, traits>;


def AddOp : GGML_Op<"ADD"> {
    let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
    let results = (outs F32Tensor);

  // Allow building an AddOp with from the two input operands.
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }]>
    ];
}

// Only two-dimensional for now
def GetRowsOp : GGML_Op<"GET_ROWS"> {
    let arguments = (ins F32Tensor:$values, I32Tensor:$indices);
    let results = (outs F32Tensor);
    let builders = [
        OpBuilder<(ins "Value":$values, "Value":$indices)> 
    ];
}

#endif
