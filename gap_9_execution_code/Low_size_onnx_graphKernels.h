#ifndef Low_size__ONNX_GRAPHKERNEL_H__
#define Low_size__ONNX_GRAPHKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "at_api.h"
#include "onnx_graph.h"
#include "CNN_BasicKernels_fp32.h"
#include "CNN_BasicKernels_f16.h"
#include "CNN_BasicKernels_f16a.h"
#include "ResizeBasicKernels.h"
#include "CNN_BasicKernels_SQ8.h"
#include "Low_size_Expression_Kernels.h"
#define _Low_size_onnx_graph_L1_Memory_SIZE 115680
#define _Low_size_onnx_graph_L2_Memory_SIZE 262080
#define _Low_size_onnx_graph_L2_Memory_Dyn_SIZE 737920
extern char *Low_size_onnx_graph_L1_Memory; /* Size given for generation: 115712 bytes, used: 115680 bytes */
extern char *Low_size_onnx_graph_L2_Memory; /* Size used for generation (static): 262080 bytes */
extern char *Low_size_onnx_graph_L2_Memory_Dyn; /* Size used for generation (dynamic): 737920 bytes */
extern void S1_input_1_resizer(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S4__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S9__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S12__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S14_expr_1_multiple_1(
		f16 * __restrict__ expr_1_in_0,
		f16 * __restrict__ expr_1_in_1,
		f16 * __restrict__ expr_1_out_0);
extern void Low_size_S18__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S21__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S24__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S29__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S32__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S35__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S37_expr_7_multiple_1(
		f16 * __restrict__ expr_7_in_0,
		f16 * __restrict__ expr_7_in_1,
		f16 * __restrict__ expr_7_out_0);
extern void Low_size_S41__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S44__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S47__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S52__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S55__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S58__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S60_expr_12_multiple_1(
		f16 * __restrict__ expr_12_in_0,
		f16 * __restrict__ expr_12_in_1,
		f16 * __restrict__ expr_12_out_0);
extern void Low_size_S64__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S67__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S70__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S75__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S78__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S82__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S83__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3,
		f16 * __restrict__ Out4,
		f16 * __restrict__ Out5,
		f16 * __restrict__ Out6,
		f16 * __restrict__ Out7,
		f16 * __restrict__ Out8,
		f16 * __restrict__ Out9,
		f16 * __restrict__ Out10,
		f16 * __restrict__ Out11,
		f16 * __restrict__ Out12);
extern void Low_size_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S317__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S322__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S327__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S332__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S337__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S342__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S347__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S352__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S357__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S376__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void Low_size_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S379__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S380__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S381__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S382__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S383__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S384__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S385__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S387_expr_15_multiple_1(
		f16 * __restrict__ expr_15_in_0,
		f16 * __restrict__ expr_15_in_1,
		f16 * __restrict__ expr_15_out_0);
extern void Low_size_S388__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S390__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S391__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S393_expr_16_multiple_1(
		f16 * __restrict__ expr_16_in_0,
		f16 * __restrict__ expr_16_in_1,
		f16 * __restrict__ expr_16_out_0);
extern void Low_size_S395__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S396__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S397__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S399_expr_17_multiple_1(
		f16 * __restrict__ expr_17_in_0,
		f16 * __restrict__ expr_17_in_1,
		f16 * __restrict__ expr_17_out_0);
extern void Low_size_S401__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S404__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S405__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3,
		f16 * __restrict__ Out4,
		f16 * __restrict__ Out5,
		f16 * __restrict__ Out6,
		f16 * __restrict__ Out7,
		f16 * __restrict__ Out8,
		f16 * __restrict__ Out9,
		f16 * __restrict__ Out10,
		f16 * __restrict__ Out11,
		f16 * __restrict__ Out12);
extern void Low_size_S406__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S410__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S411__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S415__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S416__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S420__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S421__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S425__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S426__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S430__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S431__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S435__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S436__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S440__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S441__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S445__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S446__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S450__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S451__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S455__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S456__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S470__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void Low_size_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S473__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S474__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S475__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S476__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S477__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S478__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S479__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S481_expr_22_multiple_1(
		f16 * __restrict__ expr_22_in_0,
		f16 * __restrict__ expr_22_in_1,
		f16 * __restrict__ expr_22_out_0);
extern void Low_size_S482__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S484__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S485__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S487_expr_18_multiple_1(
		f16 * __restrict__ expr_18_in_0,
		f16 * __restrict__ expr_18_in_1,
		f16 * __restrict__ expr_18_out_0);
extern void Low_size_S489__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S490__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S491__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S493_expr_19_multiple_1(
		f16 * __restrict__ expr_19_in_0,
		f16 * __restrict__ expr_19_in_1,
		f16 * __restrict__ expr_19_out_0);
extern void Low_size_S495__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S496__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S497__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S500__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S503__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S504__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3,
		f16 * __restrict__ Out4,
		f16 * __restrict__ Out5,
		f16 * __restrict__ Out6,
		f16 * __restrict__ Out7,
		f16 * __restrict__ Out8,
		f16 * __restrict__ Out9,
		f16 * __restrict__ Out10,
		f16 * __restrict__ Out11,
		f16 * __restrict__ Out12,
		f16 * __restrict__ Out13,
		f16 * __restrict__ Out14,
		f16 * __restrict__ Out15,
		f16 * __restrict__ Out16,
		f16 * __restrict__ Out17,
		f16 * __restrict__ Out18,
		f16 * __restrict__ Out19,
		f16 * __restrict__ Out20,
		f16 * __restrict__ Out21,
		f16 * __restrict__ Out22,
		f16 * __restrict__ Out23,
		f16 * __restrict__ Out24);
extern void Low_size_S505__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S509__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S510__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S514__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S515__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S519__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S520__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S524__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S525__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S529__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S530__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S534__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S535__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S539__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S540__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S544__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S545__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S549__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S550__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S554__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S555__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S559__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S560__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S564__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S565__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S569__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S570__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S574__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S575__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S579__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S580__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S584__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S585__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S589__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S590__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S594__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S595__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S599__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S600__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S604__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S605__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S609__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S610__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S614__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S615__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S629__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void Low_size_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S632__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S633__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S634__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S635__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S636__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S637__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S638__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S640_expr_27_multiple_1(
		f16 * __restrict__ expr_27_in_0,
		f16 * __restrict__ expr_27_in_1,
		f16 * __restrict__ expr_27_out_0);
extern void Low_size_S641__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S643__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S644__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S646_expr_28_multiple_1(
		f16 * __restrict__ expr_28_in_0,
		f16 * __restrict__ expr_28_in_1,
		f16 * __restrict__ expr_28_out_0);
extern void Low_size_S648__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S649__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S650__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S652_expr_29_multiple_1(
		f16 * __restrict__ expr_29_in_0,
		f16 * __restrict__ expr_29_in_1,
		f16 * __restrict__ expr_29_out_0);
extern void Low_size_S654__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S657__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S658__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3,
		f16 * __restrict__ Out4,
		f16 * __restrict__ Out5,
		f16 * __restrict__ Out6,
		f16 * __restrict__ Out7,
		f16 * __restrict__ Out8,
		f16 * __restrict__ Out9,
		f16 * __restrict__ Out10,
		f16 * __restrict__ Out11,
		f16 * __restrict__ Out12,
		f16 * __restrict__ Out13,
		f16 * __restrict__ Out14,
		f16 * __restrict__ Out15,
		f16 * __restrict__ Out16,
		f16 * __restrict__ Out17,
		f16 * __restrict__ Out18,
		f16 * __restrict__ Out19,
		f16 * __restrict__ Out20,
		f16 * __restrict__ Out21,
		f16 * __restrict__ Out22,
		f16 * __restrict__ Out23,
		f16 * __restrict__ Out24);
extern void Low_size_S659__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S663__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S664__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S668__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S669__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S673__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S674__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S678__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S679__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S683__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S684__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S688__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S689__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S693__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S694__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S698__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S699__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S703__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S704__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S708__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S709__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S713__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S714__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S718__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S719__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S723__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S724__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S728__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S729__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S733__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S734__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S738__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S739__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S743__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S744__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S748__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S749__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S753__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S754__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S758__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S759__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S763__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S764__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S768__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S769__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S783__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void Low_size_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S786__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S787__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S788__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S789__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S790__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S791__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S792__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S794_expr_34_multiple_1(
		f16 * __restrict__ expr_34_in_0,
		f16 * __restrict__ expr_34_in_1,
		f16 * __restrict__ expr_34_out_0);
extern void Low_size_S795__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S797__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S798__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S800_expr_30_multiple_1(
		f16 * __restrict__ expr_30_in_0,
		f16 * __restrict__ expr_30_in_1,
		f16 * __restrict__ expr_30_out_0);
extern void Low_size_S802__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S803__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S804__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S806_expr_31_multiple_1(
		f16 * __restrict__ expr_31_in_0,
		f16 * __restrict__ expr_31_in_1,
		f16 * __restrict__ expr_31_out_0);
extern void Low_size_S808__backbone_lateral_conv0_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S811__backbone_upsample_Resize_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S813__backbone_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S815__backbone_C3_p4_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S816__backbone_C3_p4_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void Low_size_S818_expr_38_multiple_1(
		f16 * __restrict__ expr_38_in_0,
		f16 * __restrict__ expr_38_out_0);
extern void Low_size_S820__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S821__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S822__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S823__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S824__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S825__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S826__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S827__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S828__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S831_expr_48_multiple_1(
		f16 * __restrict__ expr_48_in_0,
		f16 * __restrict__ expr_48_out_0);
extern void Low_size_S832__backbone_C3_p4_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S834__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S835__backbone_reduce_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S838__backbone_upsample_1_Resize_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void Low_size_S840__backbone_Concat_1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S842__backbone_C3_p3_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S843__backbone_C3_p3_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void Low_size_S845_expr_51_multiple_1(
		f16 * __restrict__ expr_51_in_0,
		f16 * __restrict__ expr_51_out_0);
extern void Low_size_S847__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S848__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S849__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S850__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S851__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S852__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S853__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S854__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S855__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S858_expr_61_multiple_1(
		f16 * __restrict__ expr_61_in_0,
		f16 * __restrict__ expr_61_out_0);
extern void Low_size_S859__backbone_C3_p3_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S861__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S862__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S863__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S865__head_stems_0_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S866__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S867__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S868__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S869__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S870__head_cls_preds_0_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S872__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S873__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S874__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S875__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S876__head_reg_preds_0_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S878__head_obj_preds_0_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S880__head_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void Low_size_S882__backbone_Concat_2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S884__backbone_C3_n3_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S885__backbone_C3_n3_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void Low_size_S887_expr_74_multiple_1(
		f16 * __restrict__ expr_74_in_0,
		f16 * __restrict__ expr_74_out_0);
extern void Low_size_S889__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S890__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S891__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S892__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S893__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S894__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S895__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S896__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S897__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S900_expr_84_multiple_1(
		f16 * __restrict__ expr_84_in_0,
		f16 * __restrict__ expr_84_out_0);
extern void Low_size_S901__backbone_C3_n3_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S903__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S904__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S905__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S907__head_stems_1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S908__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S909__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S910__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S911__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S912__head_cls_preds_1_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S914__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S915__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S916__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S917__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S918__head_reg_preds_1_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S920__head_obj_preds_1_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S922__head_Concat_1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void Low_size_S924__backbone_Concat_3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S926__backbone_C3_n4_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S927__backbone_C3_n4_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void Low_size_S929_expr_97_multiple_1(
		f16 * __restrict__ expr_97_in_0,
		f16 * __restrict__ expr_97_out_0);
extern void Low_size_S931__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S932__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S933__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S934__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S935__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S936__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S937__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S938__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S939__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S942_expr_107_multiple_1(
		f16 * __restrict__ expr_107_in_0,
		f16 * __restrict__ expr_107_out_0);
extern void Low_size_S943__backbone_C3_n4_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void Low_size_S945__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S946__head_stems_2_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S947__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S948__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S949__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S950__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S951__head_cls_preds_2_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S953__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S954__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S955__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S956__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S957__head_reg_preds_2_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S959__head_obj_preds_2_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void Low_size_S961__head_Concat_2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void Low_size_S964__Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void Low_size_S965_expr_118_multiple_1(
		f16 * __restrict__ expr_118_in_0,
		f16 * __restrict__ expr_118_in_1,
		f16 * __restrict__ expr_118_out_0);
extern void Low_size_S966__Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern int Low_size_onnx_graphCNN_Construct(int a);
extern void Low_size_onnx_graphCNN_ConstructCluster();
extern int Low_size_onnx_graphCNN_Destruct();
extern int Low_size_onnx_graphCNN_Memory(AT_MEM_TYPE Which);
extern f16 * __restrict__ Low_size_Input_1;
extern f16 * __restrict__ Low_size_Output_1;
extern int Low_size_onnx_graphCNN(
);
extern unsigned int Low_size_AT_GraphPerf[375];
extern unsigned int Low_size_AT_GraphPerf_CNN_Total;
extern char * Low_size_AT_GraphNodeNames[375];
extern unsigned int Low_size_AT_GraphOperInfosNames[375];
#endif
