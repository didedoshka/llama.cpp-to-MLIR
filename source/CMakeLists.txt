# add dialect and passes
add_subdirectory(MLIRGGML)

# add compiler
add_executable(compiler Compiler.cpp MLIRGen.cpp Utils.cpp)
llvm_update_compile_flags(compiler) # add no-rtti when needed
target_link_libraries(
  compiler
  PRIVATE MLIRGGML
          MLIRParser
          MLIRSupport
          MLIRIR
          MLIRFuncDialect
          MLIRFunctionInterfaces
          MLIRArithDialect
          MLIRPass
          MLIRBufferizationTransforms
          MLIRRegisterAllDialects
          MLIRBufferizationToMemRef
          MLIRExecutionEngine
          llama)

mlir_check_all_link_libraries(compiler)
