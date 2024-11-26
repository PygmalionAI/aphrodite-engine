#pragma once

#include <cute/tensor.hpp>
#include <torch/all.h>


namespace cute {

template <size_t... I, typename Layout>
CUTE_HOST_DEVICE static constexpr auto permute_layout(Layout l) {
  static_assert(rank(l) == sizeof...(I), "Invalid permutation, rank mismatch");
  return cute::make_layout(cute::get<I>(l)...);
}

// f(x) = x layout
template <typename Layout>
CUTE_HOST_DEVICE static constexpr bool is_identity_layout() {
  if constexpr (std::is_name_v<Layout, void>)
    return true;
  else {
    constexpr auto coalesced_layout = coalesce(Layout{});
    // TODO(alpin): was it && or and for MSVC?
    if constexpr (rank(coalesced_layout) == 1 && 
                  stride<0>(coalesced_layout) == 1) {
      return true;
    }
    return false;
  }
}

template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}

template <typename T, typename Elements>
CUTE_HOST_DEVICE static constexpr auto create_auto_vectorizing_copy() {
  constexpr auto bits = sizeof_bits_v<T> * Elements{};
  if constexpr (bits % 128 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<128>{};
  } else if constexpr (bits % 64 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<64>{};
  } else if constexpr (bits % 16 == 0) {
    return AutoVectorizingCopyWithAssumedAlignment<16>{};
  } else {
    return AutoVectorizingCopyWithAssumedAlignment<8>{};
  }
}

};  // namespace cute