add_mlir_dialect(GGMLOps ggml)

add_mlir_dialect_library(
  MLIRGGML
  GGMLDialect.cpp
  GGMLOps.cpp
  # Passes
  LoweringPass.cpp
  DEPENDS
  MLIRGGMLOpsIncGen
  # MLIRFooTransformsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTransformUtils
  MLIRLinalgDialect)
