#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

namespace ggml {

std::unique_ptr<mlir::Pass> createLoweringPass();

}
} // namespace mlir

#endif
