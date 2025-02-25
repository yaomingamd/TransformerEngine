/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include <iostream>
#include "ck_fused_attn/ck_fused_attn.hpp"
#include "bias.hpp"
#include "mask.hpp"
#include "fmha_fwd.hpp"
#include "ck_fused_attn_utils.hpp"

namespace ck_fused_attn{

__global__ void softmax_lse_from_thd(
  uint64_t b, uint64_t h, uint64_t s_q,
  const int32_t* cu_seqlen_q,
  const float* lse_thd,
  float* lse){

  int32_t total_seqlen_q = cu_seqlen_q[b];
  
  for(uint64_t bh_idx = blockIdx.x*blockDim.x + threadIdx.x; bh_idx < b*h; ss_idx += blockDim.x * gridDim.x){
    uint64_t b_idx = bh_idx/h;
    uint64_t h_idx = bh_idx%h;
    // loop over a given batch and head idx
    for(uint64_t s_idx = cu_seqlen_q[b_idx]; s_idx < cu_seqlen_q[b_idx + 1]; s_idx++){
      lse[bh_idx * s_q + s_idx - cu_seqlen_q[b_idx]] = lse_thd[h_idx * total_seqlen_q + s_idx];
    }
  }
}


hipError_t ck_attn_varlen_fwd(
  DType dtype,
  uint64_t b, uint64_t h, uint64_t hg, uint64_t s_q, uint64_t s_kv, uint64_t d,
  const void* q_ptr, 
  uint64_t stride_h_q, uint64_t stride_s_q,
  const void* k_ptr, 
  uint64_t stride_h_k, uint64_t stride_s_k,
  const void* v_ptr, 
  uint64_t stride_h_v, uint64_t stride_s_v,
  const void* cu_seqlen_q_ptr, const void* cu_seqlen_kv_ptr,
  bool is_training,
  float scaling_factor,
  float dropout_probability,
  void* philox_seed_ptr, void* philox_offset_ptr,
  MaskType attn_mask_type,
  int64_t window_size_left, int64_t window_size_right,
  void* o_ptr, 
  uint64_t stride_h_o, uint64_t stride_s_o,
  void* lse_thd_ptr,
  void* lse_ptr, 
  hipStream_t stream){

  bool has_dropout = (is_training && dropout_probability > 0.f);
  bool has_lse = (lse_ptr != nullptr);

  /* CK input parameters */
  ck_tile::index_t batch = b;
  ck_tile::index_t nhead = h;
  ck_tile::index_t hdim_q = d;
  ck_tile::index_t nhead_k = hg;
  ck_tile::index_t hdim_v = d;
  ck_tile::index_t max_seqlen_q = s_q;
  ck_tile::index_t max_seqlen_k = s_kv;
  ck_tile::index_t total_seqlen_q = *(static_cast<const int32_t *>(cu_seqlen_q_ptr) + b)
  ck_tile::index_t total_seqlen_kv = *(static_cast<const int32_t *>(cu_seqlen_kv_ptr) + b)

  float scale_s = scaling_factor;
  float scale_p = 1.f;
  float scale_o = 1.f;
  float p_drop = dropout_probability;
  bool is_group_mode = true;
  bool is_v_rowmajor = true;
  bool do_fp8_static_quant = false;

  // THD does not work with bias

  mask_enum mask_type;
  ck_tile::index_t left, right;
  if (attn_mask_type == MaskType::no_mask){
    mask_type = mask_enum::no_mask;
  }else if(attn_mask_type == MaskType::mask_top_left){
    mask_type = mask_enum::mask_top_left;
  }else if(attn_mask_type == MaskType::mask_bottom_right){
    mask_type = mask_enum::mask_bottom_right;
  }else{
    mask_type = mask_enum::window_generic;
  }
  left = window_size_left;
  right = window_size_right;
  
  ck_tile::stream_config stream_config{stream};

  std::string data_type_str;
  if(dtype==DType::kFloat16){
    data_type_str = "fp16";
  }else if(dtype==DType::kBFloat16){
    data_type_str = "bf16";
  }else{
    //TODO: better error out system
    throw std::runtime_error("Invalid dtype in ck_fused_attn.");
  }

  auto fmha_traits = fmha_fwd_traits{
    hdim_q,    hdim_v,    data_type_str, is_group_mode, is_v_rowmajor,
    mask_type, bias_type, has_lse,       has_dropout,   do_fp8_static_quant};

  auto fmha_args = [&]() {
    // setup stride_* arguments
    const ck_tile::index_t stride_q = stride_s_q;
    const ck_tile::index_t stride_k = stride_s_k;
    const ck_tile::index_t stride_v = stride_s_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t stride_bias = 0;
    // randval not used
    const ck_tile::index_t stride_randval = 0;
    const ck_tile::index_t stride_o = stride_s_o;
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = stride_h_q;
    const ck_tile::index_t nhead_stride_k = stride_h_k;
    const ck_tile::index_t nhead_stride_v = stride_h_v;
    // bias not used in THD qkv layout
    const ck_tile::index_t nhead_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t nhead_stride_randval = 0;
    // CK group mode requires softmax_lse be of shape [h, total_s_q]
    const ck_tile::index_t nhead_stride_lse = total_seqlen_q;
    const ck_tile::index_t nhead_stride_o = stride_h_o;
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = 0;
    const ck_tile::index_t batch_stride_k = 0;
    const ck_tile::index_t batch_stride_v = 0;
    // bias not used in THD qkv layout
    const ck_tile::index_t batch_stride_bias = 0;
    //TODO: randval never used, can we remove it
    const ck_tile::index_t batch_stride_randval = 0;
    const ck_tile::index_t batch_stride_lse = 0;
    const ck_tile::index_t batch_stride_o = 0;

    return fmha_fwd_args{q_ptr,
                         k_ptr,
                         v_ptr,
                         nullptr,//bias_ptr
                         nullptr,//rand_val_ptr
                         lse_thd_ptr,
                         o_ptr,
                         cu_seqlen_q_ptr,//cu_seqlen_q
                         cu_seqlen_kv_ptr,//cu_seqlen_kv
                         nullptr, /* seqlen_k_ptr */
                         total_seqlen_q,
                         total_seqlen_kv,
                         batch,
                         max_seqlen_q,
                         hdim_q,
                         hdim_v,
                         nhead,
                         nhead_k,
                         scale_s,
                         scale_p,
                         scale_o,
                         stride_q,
                         stride_k,
                         stride_v,
                         0,//stride_alibi_slopes
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         nhead_stride_bias,
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         batch_stride_bias,
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         left,
                         right,
                         static_cast<ck_tile::index_t>(mask_type),
                         p_drop,
                         false,
                         std::pair<const void*, const void*>{philox_seed_ptr, philox_offset_ptr}};
  }();

  // print ck traits and args when needed
  log_fwd_config(__FUNCTION__, fmha_traits, fmha_args); 

  float average_runtime = fmha_fwd(fmha_traits, fmha_args, stream_config);
  if(average_runtime < 0){
    //TODO: better error out system
    throw std::runtime_error("fused attn configs not supported in ck_fused_attn fwd pass.");
  }

  // convert softmax_lse from [h, total_seqlen_q] to [b, h, s_q]
  assert((lse_thd_ptr!=nullptr) && (lse_ptr!=nullptr) && (lse_thd_ptr!=lse_ptr));
  constexpr int THREADS_PER_BLOCK = 1024;
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(ceil(1.0 * b * h/THREADS_PER_BLOCK));
  hipLaunchKernelGGL(
    softmax_lse_from_thd, grid, block, 0, stream,
    b, h, s_q, static_cast<const int32_t*>(cu_seqlen_q_ptr),
    static_cast<const float*>(lse_thd_ptr),
    static_cast<float*>(lse_ptr));

  return hipSuccess;
}//namespace ck_fused_attn
