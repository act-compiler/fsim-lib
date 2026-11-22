// vmap: elementwise ops and select, beside an unbatched constant.
module {
  func.func @main(%v0: tensor<4xf32>, %v1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %v3 = "stablehlo.compare"(%v0, %v2) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %v4 = "stablehlo.select"(%v3, %v1, %v0) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %v5 = "stablehlo.add"(%v0, %v1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "func.return"(%v5, %v4) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }
}
