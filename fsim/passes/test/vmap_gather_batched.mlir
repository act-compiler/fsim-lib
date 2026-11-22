// vmap: a gather at batched indices.
module {
  func.func @main(%v0: tensor<2x8xf32>, %v1: tensor<1xi64>, %v2: tensor<2x4xf32>) -> (tensor<2x8xf32>, tensor<1xi64>, tensor<2x4xf32>) {
    %v3 = "stablehlo.gather"(%v0, %v1) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = array<i64: 2, 4>} : (tensor<2x8xf32>, tensor<1xi64>) -> tensor<2x4xf32>
    "func.return"(%v0, %v1, %v3) : (tensor<2x8xf32>, tensor<1xi64>, tensor<2x4xf32>) -> ()
  }
}
