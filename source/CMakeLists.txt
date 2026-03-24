# add dialect and passes
add_subdirectory(MLIRGGML)


# add compiler
add_executable(compiler Compiler.cpp)
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
          llama)
