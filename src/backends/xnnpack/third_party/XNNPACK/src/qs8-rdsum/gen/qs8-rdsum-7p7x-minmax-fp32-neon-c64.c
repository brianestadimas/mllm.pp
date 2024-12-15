// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_qs8_rdsum_ukernel_7p7x__neon_c64(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    int32x4_t vacc0 = vdupq_n_s32(0);
    int32x4_t vacc4 = vdupq_n_s32(0);
    int32x4_t vacc8 = vdupq_n_s32(0);
    int32x4_t vacc12 = vdupq_n_s32(0);
    int32x4_t vacc16 = vdupq_n_s32(0);
    int32x4_t vacc20 = vdupq_n_s32(0);
    int32x4_t vacc24 = vdupq_n_s32(0);
    int32x4_t vacc28 = vdupq_n_s32(0);
    int32x4_t vacc32 = vdupq_n_s32(0);
    int32x4_t vacc36 = vdupq_n_s32(0);
    int32x4_t vacc40 = vdupq_n_s32(0);
    int32x4_t vacc44 = vdupq_n_s32(0);
    int32x4_t vacc48 = vdupq_n_s32(0);
    int32x4_t vacc52 = vdupq_n_s32(0);
    int32x4_t vacc56 = vdupq_n_s32(0);
    int32x4_t vacc60 = vdupq_n_s32(0);

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      int16x8_t vacc16_0 = vmovq_n_s16(0);
      int16x8_t vacc16_8 = vmovq_n_s16(0);
      int16x8_t vacc16_16 = vmovq_n_s16(0);
      int16x8_t vacc16_24 = vmovq_n_s16(0);
      int16x8_t vacc16_32 = vmovq_n_s16(0);
      int16x8_t vacc16_40 = vmovq_n_s16(0);
      int16x8_t vacc16_48 = vmovq_n_s16(0);
      int16x8_t vacc16_56 = vmovq_n_s16(0);
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        int8x8_t vin0;
        int8x8_t vin8;
        int8x8_t vin16;
        int8x8_t vin24;
        int8x8_t vin32;
        int8x8_t vin40;
        int8x8_t vin48;
        int8x8_t vin56;
        vin0 = vld1_s8(&i0[0]);
        vin8 = vld1_s8(&i0[8]);
        vin16 = vld1_s8(&i0[16]);
        vin24 = vld1_s8(&i0[24]);
        vin32 = vld1_s8(&i0[32]);
        vin40 = vld1_s8(&i0[40]);
        vin48 = vld1_s8(&i0[48]);
        vin56 = vld1_s8(&i0[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i1[0]);
        vin8 = vld1_s8(&i1[8]);
        vin16 = vld1_s8(&i1[16]);
        vin24 = vld1_s8(&i1[24]);
        vin32 = vld1_s8(&i1[32]);
        vin40 = vld1_s8(&i1[40]);
        vin48 = vld1_s8(&i1[48]);
        vin56 = vld1_s8(&i1[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i2[0]);
        vin8 = vld1_s8(&i2[8]);
        vin16 = vld1_s8(&i2[16]);
        vin24 = vld1_s8(&i2[24]);
        vin32 = vld1_s8(&i2[32]);
        vin40 = vld1_s8(&i2[40]);
        vin48 = vld1_s8(&i2[48]);
        vin56 = vld1_s8(&i2[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i3[0]);
        vin8 = vld1_s8(&i3[8]);
        vin16 = vld1_s8(&i3[16]);
        vin24 = vld1_s8(&i3[24]);
        vin32 = vld1_s8(&i3[32]);
        vin40 = vld1_s8(&i3[40]);
        vin48 = vld1_s8(&i3[48]);
        vin56 = vld1_s8(&i3[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i4[0]);
        vin8 = vld1_s8(&i4[8]);
        vin16 = vld1_s8(&i4[16]);
        vin24 = vld1_s8(&i4[24]);
        vin32 = vld1_s8(&i4[32]);
        vin40 = vld1_s8(&i4[40]);
        vin48 = vld1_s8(&i4[48]);
        vin56 = vld1_s8(&i4[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i5[0]);
        vin8 = vld1_s8(&i5[8]);
        vin16 = vld1_s8(&i5[16]);
        vin24 = vld1_s8(&i5[24]);
        vin32 = vld1_s8(&i5[32]);
        vin40 = vld1_s8(&i5[40]);
        vin48 = vld1_s8(&i5[48]);
        vin56 = vld1_s8(&i5[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        vin0 = vld1_s8(&i6[0]);
        vin8 = vld1_s8(&i6[8]);
        vin16 = vld1_s8(&i6[16]);
        vin24 = vld1_s8(&i6[24]);
        vin32 = vld1_s8(&i6[32]);
        vin40 = vld1_s8(&i6[40]);
        vin48 = vld1_s8(&i6[48]);
        vin56 = vld1_s8(&i6[56]);
        vacc16_0 = vaddw_s8(vacc16_0, vin0);
        vacc16_8 = vaddw_s8(vacc16_8, vin8);
        vacc16_16 = vaddw_s8(vacc16_16, vin16);
        vacc16_24 = vaddw_s8(vacc16_24, vin24);
        vacc16_32 = vaddw_s8(vacc16_32, vin32);
        vacc16_40 = vaddw_s8(vacc16_40, vin40);
        vacc16_48 = vaddw_s8(vacc16_48, vin48);
        vacc16_56 = vaddw_s8(vacc16_56, vin56);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0 = vaddw_s16(vacc0, vget_low_s16(vacc16_0));
      vacc4 = vaddw_s16(vacc4, vget_high_s16(vacc16_0));
      vacc8 = vaddw_s16(vacc8, vget_low_s16(vacc16_8));
      vacc12 = vaddw_s16(vacc12, vget_high_s16(vacc16_8));
      vacc16 = vaddw_s16(vacc16, vget_low_s16(vacc16_16));
      vacc20 = vaddw_s16(vacc20, vget_high_s16(vacc16_16));
      vacc24 = vaddw_s16(vacc24, vget_low_s16(vacc16_24));
      vacc28 = vaddw_s16(vacc28, vget_high_s16(vacc16_24));
      vacc32 = vaddw_s16(vacc32, vget_low_s16(vacc16_32));
      vacc36 = vaddw_s16(vacc36, vget_high_s16(vacc16_32));
      vacc40 = vaddw_s16(vacc40, vget_low_s16(vacc16_40));
      vacc44 = vaddw_s16(vacc44, vget_high_s16(vacc16_40));
      vacc48 = vaddw_s16(vacc48, vget_low_s16(vacc16_48));
      vacc52 = vaddw_s16(vacc52, vget_high_s16(vacc16_48));
      vacc56 = vaddw_s16(vacc56, vget_low_s16(vacc16_56));
      vacc60 = vaddw_s16(vacc60, vget_high_s16(vacc16_56));
      r = doz(r, 252);
    }

    const int32_t* o = output;
    int32x4_t vo0 = vld1q_s32(o); o += 4;
    int32x4_t vo4 = vld1q_s32(o); o += 4;
    int32x4_t vo8 = vld1q_s32(o); o += 4;
    int32x4_t vo12 = vld1q_s32(o); o += 4;
    int32x4_t vo16 = vld1q_s32(o); o += 4;
    int32x4_t vo20 = vld1q_s32(o); o += 4;
    int32x4_t vo24 = vld1q_s32(o); o += 4;
    int32x4_t vo28 = vld1q_s32(o); o += 4;
    int32x4_t vo32 = vld1q_s32(o); o += 4;
    int32x4_t vo36 = vld1q_s32(o); o += 4;
    int32x4_t vo40 = vld1q_s32(o); o += 4;
    int32x4_t vo44 = vld1q_s32(o); o += 4;
    int32x4_t vo48 = vld1q_s32(o); o += 4;
    int32x4_t vo52 = vld1q_s32(o); o += 4;
    int32x4_t vo56 = vld1q_s32(o); o += 4;
    int32x4_t vo60 = vld1q_s32(o); o += 4;
    vacc0 = vaddq_s32(vo0, vacc0);
    vacc4 = vaddq_s32(vo4, vacc4);
    vacc8 = vaddq_s32(vo8, vacc8);
    vacc12 = vaddq_s32(vo12, vacc12);
    vacc16 = vaddq_s32(vo16, vacc16);
    vacc20 = vaddq_s32(vo20, vacc20);
    vacc24 = vaddq_s32(vo24, vacc24);
    vacc28 = vaddq_s32(vo28, vacc28);
    vacc32 = vaddq_s32(vo32, vacc32);
    vacc36 = vaddq_s32(vo36, vacc36);
    vacc40 = vaddq_s32(vo40, vacc40);
    vacc44 = vaddq_s32(vo44, vacc44);
    vacc48 = vaddq_s32(vo48, vacc48);
    vacc52 = vaddq_s32(vo52, vacc52);
    vacc56 = vaddq_s32(vo56, vacc56);
    vacc60 = vaddq_s32(vo60, vacc60);
    vst1q_s32(output, vacc0); output += 4;
    vst1q_s32(output, vacc4); output += 4;
    vst1q_s32(output, vacc8); output += 4;
    vst1q_s32(output, vacc12); output += 4;
    vst1q_s32(output, vacc16); output += 4;
    vst1q_s32(output, vacc20); output += 4;
    vst1q_s32(output, vacc24); output += 4;
    vst1q_s32(output, vacc28); output += 4;
    vst1q_s32(output, vacc32); output += 4;
    vst1q_s32(output, vacc36); output += 4;
    vst1q_s32(output, vacc40); output += 4;
    vst1q_s32(output, vacc44); output += 4;
    vst1q_s32(output, vacc48); output += 4;
    vst1q_s32(output, vacc52); output += 4;
    vst1q_s32(output, vacc56); output += 4;
    vst1q_s32(output, vacc60); output += 4;

    input = (const int8_t*) ((uintptr_t) input + 64 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);

      for (; num_batches > 0; --num_batches) {
        int16x8_t vacc16 = vmovq_n_s16(0);
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          int8x8_t vin0 = vld1_s8(&i0[0]);
          int8x8_t vin1 = vld1_s8(&i1[0]);
          int8x8_t vin2 = vld1_s8(&i2[0]);
          int8x8_t vin3 = vld1_s8(&i3[0]);
          int8x8_t vin4 = vld1_s8(&i4[0]);
          int8x8_t vin5 = vld1_s8(&i5[0]);
          int8x8_t vin6 = vld1_s8(&i6[0]);
          vacc16 = vaddw_s8(vacc16, vin0);
          vacc16 = vaddw_s8(vacc16, vin1);
          vacc16 = vaddw_s8(vacc16, vin2);
          vacc16 = vaddw_s8(vacc16, vin3);
          vacc16 = vaddw_s8(vacc16, vin4);
          vacc16 = vaddw_s8(vacc16, vin5);
          vacc16 = vaddw_s8(vacc16, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = vaddw_s16(vacc0, vget_low_s16(vacc16));
        vacc1 = vaddw_s16(vacc1, vget_high_s16(vacc16));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        int32x4_t vo0 = vld1q_s32(output);
        int32x4_t vo1 = vld1q_s32(output + 4);
        vo0 = vaddq_s32(vo0, vacc0);
        vo1 = vaddq_s32(vo1, vacc1);
        vst1q_s32(output, vo0); output += 4;
        vst1q_s32(output, vo1); output += 4;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        if (channels & 4) {
          int32x4_t vo = vld1q_s32(output);
          vo = vaddq_s32(vo, vacc0);
          vst1q_s32(output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          int32x2_t vo = vld1_s32(output);
          vo = vadd_s32(vo, vget_low_s32(vacc0));
          vst1_s32(output, vo); output += 2;
          vacc0 = vextq_s32(vacc0, vacc0, 2);
        }
        if (channels & 1) {
          *output += vgetq_lane_s32(vacc0, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}