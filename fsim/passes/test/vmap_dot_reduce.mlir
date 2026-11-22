// vmap: dot_general and reduce, with a scalar init.
module {
  func.func @main(%v0: tensor<2x4xf32>, %v1: tensor<4x2xf32>) -> (tensor<2x4xf32>, tensor<4x2xf32>) {
    %v2 = "stablehlo.dot_general"(%v0, %v1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %v3 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %v4 = "stablehlo.reduce"(%v2, %v3) ({
    ^bb0(%v5: tensor<f32>, %v6: tensor<f32>):
      %v7 = "stablehlo.add"(%v5, %v6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%v7) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %v8 = "stablehlo.broadcast_in_dim"(%v4) {broadcast_dimensions = array<i64: 0>} : (tensor<2xf32>) -> tensor<2x4xf32>
    "func.return"(%v8, %v1) : (tensor<2x4xf32>, tensor<4x2xf32>) -> ()
  }
}
