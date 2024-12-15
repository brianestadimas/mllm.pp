// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
  // Keep this line indented to avoid it being pulled out of the #ifdef when the
  // sources are amalgamated.
  #include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

// Wraps the `kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed, size_t lhs_stride,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    union xnn_f32_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
      m, n, k / sizeof(float), lhs_packed, rhs_packed, dst, dst_stride_row, /*dst_stride_col=*/sizeof(float),
      minmax_params->scalar.min, minmax_params->scalar.max);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}