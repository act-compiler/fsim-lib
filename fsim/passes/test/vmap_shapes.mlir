// vmap: broadcast_in_dim, reshape, transpose and bitcast_convert.
module {
  func.func @main(%v0: tensor<2x4xf32>, %v1: tensor<4xui32>) -> (tensor<2x4xf32>, tensor<4xui32>) {
    %v2 = "stablehlo.transpose"(%v0) {permutation = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %v3 = "stablehlo.reshape"(%v2) : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %v4 = "stablehlo.bitcast_convert"(%v1) : (tensor<4xui32>) -> tensor<4xf32>
    %v5 = "stablehlo.broadcast_in_dim"(%v4) {broadcast_dimensions = array<i64: 1>} : (tensor<4xf32>) -> tensor<2x4xf32>
    %v6 = "stablehlo.add"(%v3, %v5) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    "func.return"(%v6, %v1) : (tensor<2x4xf32>, tensor<4xui32>) -> ()
  }
}
