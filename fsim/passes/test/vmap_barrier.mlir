// vmap: an optimization_barrier over a batched and an unbatched operand.
module {
  func.func @main(%v0: tensor<4xf32>, %v1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.constant"() {value = dense<2.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %v3:2 = "stablehlo.optimization_barrier"(%v0, %v2) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    %v4 = "stablehlo.multiply"(%v3#0, %v3#1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "func.return"(%v4, %v1) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }
}
