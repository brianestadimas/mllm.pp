// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"
#include "xnnpack/raddstoreexpminusmax.h"

void xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const struct xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const HVX_Vector vi_max = xnn_set1_f32(*max);
  const HVX_Vector vlog2e = xnn_set1_f32(0x1.715476p+0f);
  const HVX_Vector vmagic_bias = xnn_set1_f32(0x1.8000FEp23f);
  const HVX_Vector vminus_ln2_hi = xnn_set1_f32(-0x1.62E400p-1f);
  const HVX_Vector vminus_ln2_lo = xnn_set1_f32(-0x1.7F7D1Cp-20f);
  const HVX_Vector vc5 = xnn_set1_f32(0x1.0F9F9Cp-7f);
  const HVX_Vector vc4 = xnn_set1_f32(0x1.573A1Ap-5f);
  const HVX_Vector vc3 = xnn_set1_f32(0x1.555A80p-3f);
  const HVX_Vector vc2 = xnn_set1_f32(0x1.FFFDC6p-2f);
  const HVX_Vector vc1 = xnn_set1_f32(0x1.FFFFF6p-1f);
  const HVX_Vector vdenorm_cutoff = xnn_set1_f32(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);  
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  HVX_Vector vacc0 = Q6_V_vzero();
  HVX_Vector vacc1 = vacc0;
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const HVX_Vector vi0 =  xnn_loadu_f32(input);
    const HVX_Vector vi1 = xnn_loadu_f32(input + 32);
    input += 64;

    // Subtract maximum input x := i - i_max
    const HVX_Vector vx0 = xnn_sub_f32(vi0, vi_max);
    const HVX_Vector vx1 = xnn_sub_f32(vi1, vi_max);

    // n := round(x / log(2))
    HVX_Vector vn0 = xnn_fmadd_f32(vx0, vlog2e, vmagic_bias);
    HVX_Vector vn1 = xnn_fmadd_f32(vx1, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow.
    const HVX_Vector vs0 = Q6_Vw_vasl_VwR(vn0, 23);
    const HVX_Vector vs1 = Q6_Vw_vasl_VwR(vn1, 23);

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0 = xnn_sub_f32(vn0, vmagic_bias);
    vn1 = xnn_sub_f32(vn1, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    HVX_Vector vt0 = xnn_fmadd_f32(vn0, vminus_ln2_hi, vx0);
    HVX_Vector vt1 = xnn_fmadd_f32(vn1, vminus_ln2_hi, vx1);

    vt0 = xnn_fmadd_f32(vn0, vminus_ln2_lo, vt0);
    vt1 = xnn_fmadd_f32(vn1, vminus_ln2_lo, vt1);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    //  p := c5 * t + c4;
    //  p = p * t + c3;
    //  p = p * t + c2;
    //  p = p * t + c1;
    HVX_Vector vp0 = xnn_fmadd_f32(vc5, vt0, vc4);
    HVX_Vector vp1 = xnn_fmadd_f32(vc5, vt1, vc4);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc3);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc3);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc2);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc2);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc1);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 = xnn_mul_f32(vt0, vs0);
    vt1 = xnn_mul_f32(vt1, vs1);

    HVX_Vector vf0 = xnn_fmadd_f32(vt0, vp0, vs0);
    HVX_Vector vf1 = xnn_fmadd_f32(vt1, vp1, vs1);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx0), vf0);
    vf1 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx1), vf1);

    xnn_storeu_f32(output, vf0);
    xnn_storeu_f32(output + 32, vf1);
    output += 64;

    vacc0 = xnn_add_f32(vacc0, vf0);
    vacc0 = xnn_add_f32(vacc0, vf1);
  }
  vacc0 = xnn_add_f32(vacc0, vacc1);

  HVX_Vector vacc = vacc0;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector vi = xnn_loadu_f32(input);
    input += 32;

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    xnn_storeu_f32(output, vf);
    output += 32;

    vacc = xnn_add_f32(vacc, vf);
  }

  float vacc_lo = Q6_f32_vrsum_Vsf(vacc);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch < 32 * sizeof(float));

    const HVX_Vector vi = xnn_loadu_f32(input);

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    Q6_V_vstu_variable(output, batch, vf);

    vf = Q6_V_vand_QV(Q6_Q_vsetq_R(batch), vf);
    vacc_lo += Q6_f32_vrsum_Vsf(vf);
  }
  *sum = vacc_lo;
}