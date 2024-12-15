// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <limits>

#include "unary_operator.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/math.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


static void xnnpack_elu_f16(benchmark::State& state) {
  xnn_unary_params params;
  params.elu.alpha = 1.0f;
  benchmark_unary_operator<xnn_float16, xnn_float16>(state, xnn_unary_elu, &params);
}

static void xnnpack_elu_f32(benchmark::State& state) {
  xnn_unary_params params;
  params.elu.alpha = 1.0f;
  benchmark_unary_operator<float, float>(state, xnn_unary_elu, &params);
}

static void xnnpack_elu_qs8(benchmark::State& state) {
  xnn_unary_params params;
  params.elu.alpha = 1.0f;
  benchmark_unary_operator<int8_t, int8_t>(state, xnn_unary_elu, &params, {0, 1.0f}, {0, 1.0f});
}

BENCHMARK(xnnpack_elu_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK(xnnpack_elu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK(xnnpack_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

BENCHMARK(xnnpack_elu_f16)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
  ->UseRealTime();
BENCHMARK(xnnpack_elu_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_elu_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_elu_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, float>(state,
                                                tflite::BuiltinOperator_ELU);
}

static void tflite_elu_qs8(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, int8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      tflite::BuiltinOperator_ELU);
}

  BENCHMARK(tflite_elu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK(tflite_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif