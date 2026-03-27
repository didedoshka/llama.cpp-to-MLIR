module {
  func.func @compute_graph_forward() {
    %values = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01], [1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]]> : tensor<4x4xf32>
    %indices = arith.constant dense<[1, 3]> : tensor<2xindex>
    // %n = arith.constant 2 : i32

    // %0 = "ggml.GET_ROWS"(%cst, %cst_0) : (tensor<1x1x4x4xf32>, tensor<1x1x1x2xi32>) -> tensor<1x1x2x4xf32>

    %init = tensor.empty() : tensor<2x4xf32>
    %res = affine.for %i = 0 to 2 
    iter_args(%tensor_iter = %init) -> (tensor<2x4xf32>) {
        %j = tensor.extract %indices[%i] : tensor<2xindex>
        %slice = tensor.extract_slice %values[%j, 0][1, 4][1, 1] :
            tensor<4x4xf32> to tensor<4xf32>

        %tensor = tensor.insert_slice %slice into %tensor_iter[%i, 0][1, 4][1, 1] :
            tensor<4xf32> into tensor<2x4xf32>
        affine.yield %tensor : tensor<2x4xf32>
        // %to_print = tensor.cast %slice: tensor<4xf32> to tensor<*xf32>
        // func.call @printMemrefF32(%to_print) : (tensor<*xf32>) -> ()
    }

    %to_print = tensor.cast %res: tensor<2x4xf32> to tensor<*xf32>
    call @printMemrefF32(%to_print) : (tensor<*xf32>) -> ()
    return 
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
