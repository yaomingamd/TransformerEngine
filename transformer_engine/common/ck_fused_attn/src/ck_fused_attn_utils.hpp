/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#ifndef CK_FUSED_ATTN_UTILS_H
#define CK_FUSED_ATTN_UTILS_H

#include <iostream>
#include<cstdint>

//forward declaration for ck fmha traits and args
class fmha_fwd_args;
class fmha_fwd_traits;
class fmha_bwd_args;
class fmha_bwd_traits;

namespace ck_fused_attn{

#define CK_FUSED_ATTN_TYPE_SWITCH_16BIT(dtype, type, ...)   \
switch (dtype) {                                            \
  case DType::kFloat16: {                                   \
    using type = ck_tile::half_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  case DType::kBFloat16: {                                  \
    using type = ck_tile::bf16_t;                           \
    __VA_ARGS__;                                            \
    break;                                                  \
  }                                                         \
  default:                                                  \
    throw std::runtime_error("Invalid type for 16 bit..");  \
}

// element-wise bias shape
enum BiasShape{
  k11SS = 0,
  k1HSS = 1,
  kB1SS = 2,
  kBHSS = 3,
  kNumBiasShapes  /*!< Number of supported bias shapes */
};

BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h);

void log_fwd_config(const char* func_name, const fmha_fwd_traits& fmha_traits, const fmha_fwd_args& fmha_args);
void log_bwd_config(const char* func_name, const fmha_bwd_traits& fmha_traits, const fmha_bwd_args& fmha_args);

}//namespace ck_fused_attn
#endif // CK_FUSED_ATTN_UTILS_H
