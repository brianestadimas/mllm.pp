// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const v128_t vscale = wasm_v128_load32_splat(&params->fp32_scalar.scale);
  XNN_FORCE_REALIZATION(vscale);

  const v128_t vmagic_bias = wasm_f32x4_const_splat(12582912.0f);
  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const v128_t vmagic_min = wasm_i32x4_splat((int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point));
  const v128_t vmagic_bias_less_output_zero_point = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  const v128_t voutput_max = wasm_i8x16_splat(params->fp32_scalar.output_max);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vmagic_min);
  XNN_FORCE_REALIZATION(vmagic_bias_less_output_zero_point);
  XNN_FORCE_REALIZATION(voutput_max);

  const v128_t vb_zero_point = wasm_i16x8_splat(params->fp32_scalar.kernel_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 8 * sizeof(uint8_t)) {
      const v128_t vxa0 = wasm_u16x8_load8x8((const v128_t*) a0);
      a0 += 8;

      const v128_t vb01 = wasm_v128_load(w);
      const v128_t vxb0 = wasm_i16x8_sub(wasm_u16x8_extend_low_u8x16(vb01), vb_zero_point);
      const v128_t vxb1 = wasm_i16x8_sub(wasm_u16x8_extend_high_u8x16(vb01), vb_zero_point);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123,
        wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 0, 0, 0, 0), vxb0));

      vacc0x0123 = wasm_i32x4_add(vacc0x0123,
        wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 1, 1, 1, 1), vxb1));
      const v128_t vb23 = wasm_v128_load((const uint8_t*) w + 16);
      const v128_t vxb2 = wasm_i16x8_sub(wasm_u16x8_extend_low_u8x16(vb23), vb_zero_point);
      const v128_t vxb3 = wasm_i16x8_sub(wasm_u16x8_extend_high_u8x16(vb23), vb_zero_point);

      vacc0x0123 = wasm_i32x4_add(vacc0x0123,
        wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 2, 2, 2, 2), vxb2));

      vacc0x0123 = wasm_i32x4_add(vacc0x0123,
        wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 3, 3, 3, 3), vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }
    if (k != 0) {
      const v128_t vxa0 = wasm_u16x8_load8x8(a0);
      a0 = (const uint8_t*) ((uintptr_t) a0 + k);

      const v128_t vxb0 = wasm_i16x8_sub(wasm_u16x8_load8x8(w), vb_zero_point);
      w = (const uint8_t*) w + 8;

      vacc0x0123 = wasm_i32x4_add(vacc0x0123,
        wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 0, 0, 0, 0), vxb0));

      if (k > 2 * sizeof(uint8_t)) {
        const v128_t vxb1 = wasm_i16x8_sub(wasm_u16x8_load8x8(w), vb_zero_point);
        w = (const uint8_t*) w + 8;

        vacc0x0123 = wasm_i32x4_add(vacc0x0123,
          wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 1, 1, 1, 1), vxb1));

        if (k > 4 * sizeof(uint8_t)) {
          const v128_t vxb2 = wasm_i16x8_sub(wasm_u16x8_load8x8(w), vb_zero_point);
          w = (const uint8_t*) w + 8;

          vacc0x0123 = wasm_i32x4_add(vacc0x0123,
            wasm_i32x4_dot_i16x8(wasm_v32x4_shuffle(vxa0, vxa0, 2, 2, 2, 2), vxb2));
        }
      }
    }

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale);

    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vacc = wasm_u8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    vacc = wasm_u8x16_min(vacc, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}