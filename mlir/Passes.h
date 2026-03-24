#ifndef PASSES_H
#define PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace ggml {

std::unique_ptr<mlir::Pass> createLoweringPass();

}
} // namespace mlir

#endif
