// vmap: a gather at constant indices.
module {
  func.func @main(%v0: tensor<8xf32>, %v1: tensor<4xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.constant"() {value = dense<[[1], [3], [5], [7]]> : tensor<4x1xi32>} : () -> tensor<4x1xi32>
    %v3 = "stablehlo.gather"(%v0, %v2) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>} : (tensor<8xf32>, tensor<4x1xi32>) -> tensor<4xf32>
    "func.return"(%v0, %v3) : (tensor<8xf32>, tensor<4xf32>) -> ()
  }
}
