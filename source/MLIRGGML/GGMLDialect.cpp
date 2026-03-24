#include "GGMLDialect.h"
#include "GGMLOps.h"

using namespace mlir;
using namespace mlir::ggml;

#include "GGMLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void GGMLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GGMLOps.cpp.inc"
      >();
}
