// vmap: an argument and a constant returned.
module {
  func.func @main(%v0: tensor<4xf32>, %v1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.constant"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
    "func.return"(%v0, %v2) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }
}
