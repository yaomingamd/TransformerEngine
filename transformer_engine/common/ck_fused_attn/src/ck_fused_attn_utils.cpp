/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/

#include "ck_fused_attn_utils.hpp"
#include "fmha_fwd.hpp"

namespace ck_fused_attn{
BiasShape get_bias_shape(uint64_t b, uint64_t h, uint64_t bias_b, uint64_t bias_h){
  //identify BHSS with high priority to include scenaiors when b=1 and h=1
  //reduce the chance of dbias_expand_ptr usage
  if(bias_b==b && bias_h==h){
    // treat as 1 if b or h is 1
    return BiasShape::kBHSS;
  }else if(bias_b==1 && bias_h==h){
    return BiasShape::k1HSS;
  }else if(bias_b==b && bias_h==1){
    return BiasShape::kB1SS;
  }else if(bias_b==1 && bias_h==1){
    return BiasShape::k11SS;
  }else{
    //should not happen
    throw std::runtime_error("Invalid bias_shape in ck_fused_attn.");
  }
  return BiasShape::kNumBiasShapes;
}

void log_fwd_config(const char* func_name, const fmha_fwd_traits& fmha_traits, const fmha_fwd_args& fmha_args){
  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_fused_attn_log_config = true;
  }
  if (ck_fused_attn_log_config) {
    std::cout<<std::endl<<func_name<<std::endl;

    // debug fmha_traits
    std::cout<<"fmha_traits: "<<std::endl;
    std::cout<<"hdim_q: "<<fmha_traits.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_traits.hdim_v<<std::endl;
    std::cout<<"data_type: "<<fmha_traits.data_type<<std::endl;
    std::cout<<"is_group_mode: "<<fmha_traits.is_group_mode<<std::endl;
    std::cout<<"is_v_rowmajor: "<<fmha_traits.is_v_rowmajor<<std::endl;
    std::cout<<"mask_type: "<<static_cast<std::underlying_type<mask_enum>::type>(fmha_traits.mask_type)<<std::endl;
    std::cout<<"bias_type: "<<static_cast<std::underlying_type<bias_enum>::type>(fmha_traits.bias_type)<<std::endl;
    std::cout<<"has_lse: "<<fmha_traits.has_lse<<std::endl;
    std::cout<<"has_dropout: "<<fmha_traits.has_dropout<<std::endl;
    std::cout<<"do_fp8_static_quant: "<<fmha_traits.do_fp8_static_quant<<std::endl;

    // debug fmha_args
    std::cout<<"fmha_args: "<<std::endl;
    std::cout<<"q_ptr: "<<fmha_args.q_ptr<<std::endl;
    std::cout<<"k_ptr: "<<fmha_args.k_ptr<<std::endl;
    std::cout<<"v_ptr: "<<fmha_args.v_ptr<<std::endl;
    std::cout<<"bias_ptr: "<<fmha_args.bias_ptr<<std::endl;
    std::cout<<"rand_val_ptr: "<<fmha_args.rand_val_ptr<<std::endl;
    std::cout<<"lse_ptr: "<<fmha_args.lse_ptr<<std::endl;
    std::cout<<"o_ptr: "<<fmha_args.o_ptr<<std::endl;
    std::cout<<"seqstart_q_ptr: "<<fmha_args.seqstart_q_ptr<<std::endl;
    std::cout<<"seqstart_k_ptr: "<<fmha_args.seqstart_k_ptr<<std::endl;
    std::cout<<"seqlen_k_ptr: "<<fmha_args.seqlen_k_ptr<<std::endl;

    std::cout<<"seqlen_q: "<<fmha_args.seqlen_q<<std::endl;
    std::cout<<"seqlen_k: "<<fmha_args.seqlen_k<<std::endl;
    std::cout<<"batch: "<<fmha_args.batch<<std::endl;
    std::cout<<"max_seqlen_q: "<<fmha_args.max_seqlen_q<<std::endl;
    std::cout<<"hdim_q: "<<fmha_args.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_args.hdim_v<<std::endl;
    std::cout<<"nhead_q: "<<fmha_args.nhead_q<<std::endl;
    std::cout<<"nhead_k: "<<fmha_args.nhead_k<<std::endl;
    std::cout<<"scale_s: "<<fmha_args.scale_s<<std::endl;
    std::cout<<"scale_p: "<<fmha_args.scale_p<<std::endl;
    std::cout<<"scale_o: "<<fmha_args.scale_o<<std::endl;
    std::cout<<"stride_q: "<<fmha_args.stride_q<<std::endl;
    std::cout<<"stride_k: "<<fmha_args.stride_k<<std::endl;
    std::cout<<"stride_v: "<<fmha_args.stride_v<<std::endl;
    std::cout<<"stride_bias: "<<fmha_args.stride_bias<<std::endl;
    std::cout<<"stride_randval: "<<fmha_args.stride_randval<<std::endl;
    std::cout<<"stride_o: "<<fmha_args.stride_o<<std::endl;
    std::cout<<"nhead_stride_q: "<<fmha_args.nhead_stride_q<<std::endl;
    std::cout<<"nhead_stride_k: "<<fmha_args.nhead_stride_k<<std::endl;
    std::cout<<"nhead_stride_v: "<<fmha_args.nhead_stride_v<<std::endl;
    std::cout<<"nhead_stride_bias: "<<fmha_args.nhead_stride_bias<<std::endl;
    std::cout<<"nhead_stride_randval: "<<fmha_args.nhead_stride_randval<<std::endl;
    std::cout<<"nhead_stride_lse: "<<fmha_args.nhead_stride_lse<<std::endl;
    std::cout<<"nhead_stride_o: "<<fmha_args.nhead_stride_o<<std::endl;
    std::cout<<"batch_stride_q: "<<fmha_args.batch_stride_q<<std::endl;
    std::cout<<"batch_stride_k: "<<fmha_args.batch_stride_k<<std::endl;
    std::cout<<"batch_stride_v: "<<fmha_args.batch_stride_v<<std::endl;
    std::cout<<"batch_stride_bias: "<<fmha_args.batch_stride_bias<<std::endl;
    std::cout<<"batch_stride_randval: "<<fmha_args.batch_stride_randval<<std::endl;
    std::cout<<"batch_stride_lse: "<<fmha_args.batch_stride_lse<<std::endl;
    std::cout<<"batch_stride_o: "<<fmha_args.batch_stride_o<<std::endl;
    std::cout<<"window_size_left: "<<fmha_args.window_size_left<<std::endl;
    std::cout<<"window_size_right: "<<fmha_args.window_size_right<<std::endl;
    std::cout<<"mask_type: "<<fmha_args.mask_type<<std::endl;
    std::cout<<"p_drop: "<<fmha_args.p_drop<<std::endl;
    std::cout<<"s_randval: "<<fmha_args.s_randval<<std::endl;
    std::cout<<"dropout_seed_ptr: "<<std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
    std::cout<<"dropout_offset_ptr: "<<std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
  }
}


void log_bwd_config(const char* func_name, const fmha_bwd_traits& fmha_traits, const fmha_bwd_args& fmha_args){

  bool ck_fused_attn_log_config = false;
  if (const char* env_p = std::getenv("CK_FUSED_ATTN_LOG_CONFIG") ) {
    if (env_p != nullptr && std::string(env_p) == "1")
      ck_fused_attn_log_config = true;
  }
  if (ck_fused_attn_log_config) {
    std::cout<<std::endl<<"run ck fmha_bwd: "<<std::endl;
    // fmha_traits debug
    std::cout<<"fmha_traits: "<<std::endl;
    std::cout<<"hdim_q: "<<fmha_traits.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_traits.hdim_v<<std::endl;
    std::cout<<"data_type: "<<fmha_traits.data_type<<std::endl;
    std::cout<<"is_group_mode: "<<fmha_traits.is_group_mode<<std::endl;
    std::cout<<"mask_type: "<<static_cast<std::underlying_type<mask_enum>::type>(fmha_traits.mask_type)<<std::endl;
    std::cout<<"bias_type: "<<static_cast<std::underlying_type<bias_enum>::type>(fmha_traits.bias_type)<<std::endl;
    std::cout<<"has_dbias: "<<fmha_traits.has_dbias<<std::endl;
    std::cout<<"has_dropout: "<<fmha_traits.has_dropout<<std::endl;
    std::cout<<"is_store_randval: "<<fmha_traits.is_store_randval<<std::endl;
    std::cout<<"is_deterministic: "<<fmha_traits.is_deterministic<<std::endl;
    std::cout<<"uses_bwd_v3: "<<fmha_traits.uses_bwd_v3<<std::endl;
    std::cout<<"is_v3_atomic_fp32: "<<fmha_traits.is_v3_atomic_fp32<<std::endl;
    std::cout<<"how_v3_bf16_cvt: "<<fmha_traits.how_v3_bf16_cvt<<std::endl;

    // fmha_args debug
    std::cout<<"fmha_args: "<<std::endl;
    std::cout<<"q_ptr: "<<fmha_args.q_ptr<<std::endl;
    std::cout<<"k_ptr: "<<fmha_args.k_ptr<<std::endl;
    std::cout<<"v_ptr: "<<fmha_args.v_ptr<<std::endl;
    std::cout<<"bias_ptr: "<<fmha_args.bias_ptr<<std::endl;
    std::cout<<"o_ptr: "<<fmha_args.o_ptr<<std::endl;
    std::cout<<"lse_ptr: "<<fmha_args.lse_ptr<<std::endl;
    std::cout<<"do_ptr: "<<fmha_args.do_ptr<<std::endl;
    std::cout<<"d_ptr: "<<fmha_args.d_ptr<<std::endl;
    std::cout<<"rand_val_ptr: "<<fmha_args.rand_val_ptr<<std::endl;
    std::cout<<"dq_ptr: "<<fmha_args.dq_ptr<<std::endl;
    std::cout<<"dk_ptr: "<<fmha_args.dk_ptr<<std::endl;
    std::cout<<"dv_ptr: "<<fmha_args.dv_ptr<<std::endl;
    std::cout<<"dbias_ptr: "<<fmha_args.dbias_ptr<<std::endl;
    std::cout<<"seqstart_q_ptr: "<<fmha_args.seqstart_q_ptr<<std::endl;
    std::cout<<"seqstart_k_ptr: "<<fmha_args.seqstart_k_ptr<<std::endl;
    std::cout<<"seqlen_k_ptr: "<<fmha_args.seqlen_k_ptr<<std::endl;
    std::cout<<"seqlen_q: "<<fmha_args.seqlen_q<<std::endl;
    std::cout<<"seqlen_k: "<<fmha_args.seqlen_k<<std::endl;
    std::cout<<"batch: "<<fmha_args.batch<<std::endl;
    std::cout<<"max_seqlen_q: "<<fmha_args.max_seqlen_q<<std::endl;
    std::cout<<"max_seqlen_k: "<<fmha_args.max_seqlen_k<<std::endl;
    std::cout<<"hdim_q: "<<fmha_args.hdim_q<<std::endl;
    std::cout<<"hdim_v: "<<fmha_args.hdim_v<<std::endl;
    std::cout<<"nhead_q: "<<fmha_args.nhead_q<<std::endl;
    std::cout<<"nhead_k: "<<fmha_args.nhead_k<<std::endl;
    std::cout<<"scale: "<<fmha_args.scale<<std::endl;
    std::cout<<"stride_q: "<<fmha_args.stride_q<<std::endl;
    std::cout<<"stride_k: "<<fmha_args.stride_k<<std::endl;
    std::cout<<"stride_v: "<<fmha_args.stride_v<<std::endl;
    std::cout<<"stride_bias: "<<fmha_args.stride_bias<<std::endl;
    std::cout<<"stride_o: "<<fmha_args.stride_o<<std::endl;
    std::cout<<"stride_randval: "<<fmha_args.stride_randval<<std::endl;
    std::cout<<"stride_do: "<<fmha_args.stride_do<<std::endl;
    std::cout<<"stride_dq_acc: "<<fmha_args.stride_dq_acc<<std::endl;
    std::cout<<"stride_dq: "<<fmha_args.stride_dq<<std::endl;
    std::cout<<"stride_dk: "<<fmha_args.stride_dk<<std::endl;
    std::cout<<"stride_dv: "<<fmha_args.stride_dv<<std::endl;
    std::cout<<"stride_dbias: "<<fmha_args.stride_dbias<<std::endl;
    std::cout<<"nhead_stride_q: "<<fmha_args.nhead_stride_q<<std::endl;
    std::cout<<"nhead_stride_k: "<<fmha_args.nhead_stride_k<<std::endl;
    std::cout<<"nhead_stride_v: "<<fmha_args.nhead_stride_v<<std::endl;
    std::cout<<"nhead_stride_bias: "<<fmha_args.nhead_stride_bias<<std::endl;
    std::cout<<"nhead_stride_o: "<<fmha_args.nhead_stride_o<<std::endl;
    std::cout<<"nhead_stride_randval: "<<fmha_args.nhead_stride_randval<<std::endl;
    std::cout<<"nhead_stride_do: "<<fmha_args.nhead_stride_do<<std::endl;
    std::cout<<"nhead_stride_lsed: "<<fmha_args.nhead_stride_lsed<<std::endl;
    std::cout<<"nhead_stride_dq_acc: "<<fmha_args.nhead_stride_dq_acc<<std::endl;
    std::cout<<"nhead_stride_dq: "<<fmha_args.nhead_stride_dq<<std::endl;
    std::cout<<"nhead_stride_dk: "<<fmha_args.nhead_stride_dk<<std::endl;
    std::cout<<"nhead_stride_dv: "<<fmha_args.nhead_stride_dv<<std::endl;
    std::cout<<"nhead_stride_dbias: "<<fmha_args.nhead_stride_dbias<<std::endl;
    std::cout<<"batch_stride_q: "<<fmha_args.batch_stride_q<<std::endl;
    std::cout<<"batch_stride_k: "<<fmha_args.batch_stride_k<<std::endl;
    std::cout<<"batch_stride_v: "<<fmha_args.batch_stride_v<<std::endl;
    std::cout<<"batch_stride_bias: "<<fmha_args.batch_stride_bias<<std::endl;
    std::cout<<"batch_stride_o: "<<fmha_args.batch_stride_o<<std::endl;
    std::cout<<"batch_stride_randval: "<<fmha_args.batch_stride_randval<<std::endl;
    std::cout<<"batch_stride_do: "<<fmha_args.batch_stride_do<<std::endl;
    std::cout<<"batch_stride_lsed: "<<fmha_args.batch_stride_lsed<<std::endl;
    std::cout<<"batch_stride_dq_acc: "<<fmha_args.batch_stride_dq_acc<<std::endl;
    std::cout<<"batch_stride_dq: "<<fmha_args.batch_stride_dq<<std::endl;
    std::cout<<"batch_stride_dk: "<<fmha_args.batch_stride_dk<<std::endl;
    std::cout<<"batch_stride_dv: "<<fmha_args.batch_stride_dv<<std::endl;
    std::cout<<"batch_stride_dbias: "<<fmha_args.batch_stride_dbias<<std::endl;
    std::cout<<"window_size_left: "<<fmha_args.window_size_left<<std::endl;
    std::cout<<"window_size_right: "<<fmha_args.window_size_right<<std::endl;
    std::cout<<"mask_type: "<<fmha_args.mask_type<<std::endl;
    std::cout<<"p_drop: "<<fmha_args.p_drop<<std::endl;
    std::cout<<"p_undrop: "<<fmha_args.p_undrop<<std::endl;
    std::cout<<"dropout_seed_ptr: "<<std::get<0>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
    std::cout<<"dropout_offset_ptr: "<<std::get<1>(std::get<std::pair<const void*, const void*>>(fmha_args.drop_seed_offset))<<std::endl;
  }

}
}//namespace ck_fused_attn
