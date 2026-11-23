// vmap: a scatter at constant indices.
module {
  func.func @main(%v0: tensor<8xf32>, %v1: tensor<4xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
    %v2 = "stablehlo.constant"() {value = dense<[[0], [2], [4], [6]]> : tensor<4x1xi32>} : () -> tensor<4x1xi32>
    %v3 = "stablehlo.scatter"(%v0, %v2, %v1) ({
    ^bb0(%v4: tensor<f32>, %v5: tensor<f32>):
      "stablehlo.return"(%v5) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
    "func.return"(%v3, %v1) : (tensor<8xf32>, tensor<4xf32>) -> ()
  }
}
