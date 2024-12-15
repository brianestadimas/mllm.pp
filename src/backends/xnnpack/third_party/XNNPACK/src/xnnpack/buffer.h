// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_BUFFER_H_
#define __XNNPACK_TEST_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"

namespace xnnpack {

template <typename T>
class NumericLimits {
 public:
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
};

template <>
class NumericLimits<xnn_float16> {
 public:
  static xnn_float16 min() { return std::numeric_limits<float>::lowest(); }
  static xnn_float16 max() { return std::numeric_limits<float>::max(); }
};

template <typename T>
xnn_datatype datatype_of() {
  if (std::is_same<T, uint8_t>::value) {
    return xnn_datatype_quint8;
  } else if (std::is_same<T, int8_t>::value) {
    return xnn_datatype_qint8;
  } else if (std::is_same<T, int16_t>::value) {
    // TODO: We don't have this type...
    return xnn_datatype_qint8;
  } else if (std::is_same<T, xnn_float16>::value) {
    return xnn_datatype_fp16;
  } else if (std::is_same<T, float>::value) {
    return xnn_datatype_fp32;
  } else if (std::is_same<T, int32_t>::value) {
    return xnn_datatype_int32;
  } else if (std::is_same<T, uint32_t>::value) {
    // TODO: We don't have this type...
    return xnn_datatype_quint8;
  } else {
    XNN_UNREACHABLE;
  }
}

inline bool is_quantized(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      return true;
    default:
      return false;
  }
}

inline size_t datatype_size(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      return 1;
    case xnn_datatype_fp16:
      return 2;
    case xnn_datatype_fp32:
    case xnn_datatype_int32:
      return 4;
    default:
      XNN_UNREACHABLE;
  }
}

// This is a container similar to std::vector, but it leaves the memory
// uninitialized, supports alignment.
// TODO: It would be good if this also managed padding in a way that allowed
// the client code to see the unpadded data, and the padding was hidden.
template <typename T, size_t Alignment = alignof(T)>
class Buffer {
  static_assert(std::is_trivial<T>::value, "");
  T* data_;
  size_t size_;

  static void* allocate(size_t bytes) {
    size_t alignment = std::max(Alignment, sizeof(void*));
#if defined(_WIN32)
    void* memory = nullptr;
    memory = _aligned_malloc(bytes, alignment);
    if (memory == 0) {
#if !defined(__GNUC__) && !defined(_MSC_VER) || defined(__EXCEPTIONS) || \
    defined(_CPPUNWIND)
      throw std::bad_alloc();
#endif
    }
#elif defined(__ANDROID__) || defined(__CYGWIN__)
    void* memory = memalign(alignment, bytes);
    if (memory == 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc();
#endif
    }
#else
    void* memory = nullptr;
    if (posix_memalign(&memory, alignment, bytes) != 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc();
#endif
    }
#endif
    return reinterpret_cast<T*>(memory);
  }

  static void free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    ::free(p);
#endif
  }

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size)
      : data_(reinterpret_cast<T*>(allocate(size * sizeof(T)))), size_(size) {}
  Buffer(size_t size, T value) : Buffer(size) {
    std::fill(begin(), end(), value);
  }
  Buffer(std::initializer_list<T> init) : Buffer(init.size()) {
    std::copy(init.begin(), init.end(), begin());
  }
  Buffer(const Buffer& other) = delete;
  Buffer(Buffer&& other) : Buffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
  ~Buffer() {
    if (data_) free(data_);
  }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T* data() { return data_; }
  const T* data() const { return data_; }
  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  const T* cbegin() const { return data_; }
  const T* cend() const { return data_ + size_; }
  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

  bool operator==(const Buffer& other) const {
    return size_ == other.size_ && std::equal(begin(), end(), other.begin());
  }
  bool operator!=(const Buffer& other) const {
    return size_ != other.size_ || !std::equal(begin(), end(), other.begin());
  }
};

// This is a faster way of generating random numbers, by generating as many
// random values as possible for each call to rng(). Assumes that rng() returns
// entirely random bits.
template <typename T, typename Rng>
void fill_uniform_random_bits(T* data, size_t size, Rng& rng) {
  using RngT = decltype(rng());
  RngT* data_rng_t = reinterpret_cast<RngT*>(data);
  size_t size_bytes = size * sizeof(T);
  size_t i = 0;
  // Fill with as many RngT as we can.
  for (; i + sizeof(RngT) <= size_bytes; i += sizeof(RngT)) {
    *data_rng_t++ = rng();
  }
  // Fill the remaining bytes.
  char* data_char = reinterpret_cast<char*>(data_rng_t);
  for (; i < size_bytes; ++i) {
    *data_char++ = rng() & 0xff;
  }
}

};  // namespace xnnpack

#endif  // __XNNPACK_TEST_BUFFER_H_