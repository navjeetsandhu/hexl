// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/defines.hpp"
#include "ntt/ntt-internal.hpp"
#include "test/test-ntt-util.hpp"
#include "test/test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

namespace allocators {
struct CustomAllocator {
  using T = size_t;
  T* invoke_allocation(size_t size) {
    number_allocations++;
    return new T[size];
  }

  void lets_deallocate(T* ptr) {
    number_deallocations++;
    delete[] ptr;
  }
  static size_t number_allocations;
  static size_t number_deallocations;
};

size_t CustomAllocator::number_allocations = 0;
size_t CustomAllocator::number_deallocations = 0;
}  // namespace allocators

template <>
struct NTT::AllocatorAdapter<allocators::CustomAllocator>
    : public AllocatorInterface<
          NTT::AllocatorAdapter<allocators::CustomAllocator>> {
  explicit AllocatorAdapter(allocators::CustomAllocator&& a_)
      : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) {
    return a.invoke_allocation(bytes_count);
  }
  void deallocate(void* p, size_t n) {
    HEXL_UNUSED(n);
    a.lets_deallocate(static_cast<allocators::CustomAllocator::T*>(p));
  }

  allocators::CustomAllocator a;
};

template <class T>
struct NTT::AllocatorAdapter<std::allocator<T>>
    : public AllocatorInterface<NTT::AllocatorAdapter<std::allocator<T>>> {
  explicit AllocatorAdapter(std::allocator<T>&& a_) : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) { return a.allocate(bytes_count); }
  void deallocate(void* p, size_t n) { a.deallocate(static_cast<T*>(p), n); }

  std::allocator<T> a;
};

// Test different parts of the public API
TEST_P(DegreeModulusInputOutput, API) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus = std::get<1>(GetParam());

  const std::vector<uint64_t> input_copy = std::get<2>(GetParam());
  std::vector<uint64_t> exp_output = std::get<3>(GetParam());
  std::vector<uint64_t> input = input_copy;
  std::vector<uint64_t> out_buffer(input.size(), 99);

  // In-place Fwd NTT
  NTT ntt(N, modulus);

  // Test round-trip
  ntt.ComputeForward(out_buffer.data(), input.data(), 1, 1);
  ntt.ComputeInverse(input.data(), out_buffer.data(), 1, 1);
  AssertEqual(input, input_copy);

}

INSTANTIATE_TEST_SUITE_P(
    NTT, DegreeModulusInputOutput,
    ::testing::Values(
        std::make_tuple(
            32, 769,
            std::vector<uint64_t>{401, 203, 221, 352, 487, 151, 405, 356,
                                  343, 424, 635, 757, 457, 280, 624, 353,
                                  496, 353, 624, 280, 457, 757, 635, 424,
                                  343, 356, 405, 151, 487, 352, 221, 203},
            std::vector<uint64_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32})));

class NttNativeTest : public DegreeModulusBoolTest {};



}  // namespace hexl
}  // namespace intel
