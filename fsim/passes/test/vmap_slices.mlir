// vmap: slice and dynamic_update_slice at a constant index.
module {
  func.func @main(%v0: tensor<8xf32>, %v1: tensor<4xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.slice"(%v1) {start_indices = array<i64: 1>, limit_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<4xf32>) -> tensor<2xf32>
    %v3 = "stablehlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
    %v4 = "stablehlo.dynamic_update_slice"(%v0, %v2, %v3) : (tensor<8xf32>, tensor<2xf32>, tensor<i32>) -> tensor<8xf32>
    "func.return"(%v4, %v1) : (tensor<8xf32>, tensor<4xf32>) -> ()
  }
}
