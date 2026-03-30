#ifndef __ONNX_GRAPHKERNEL_H__
#define __ONNX_GRAPHKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "at_api.h"
#include "onnx_graph.h"
#include "CNN_BasicKernels_fp32.h"
#include "CNN_BasicKernels_f16.h"
#include "CNN_BasicKernels_f16a.h"
#include "ResizeBasicKernels.h"
#include "CNN_BasicKernels_SQ8.h"
#include "Expression_Kernels.h"
#define _onnx_graph_L1_Memory_SIZE 111680
#define _onnx_graph_L2_Memory_SIZE 129600
#define _onnx_graph_L2_Memory_Dyn_SIZE 870400
extern char *onnx_graph_L1_Memory; /* Size given for generation: 111712 bytes, used: 111680 bytes */
extern char *onnx_graph_L2_Memory; /* Size used for generation (static): 129600 bytes */
extern char *onnx_graph_L2_Memory_Dyn; /* Size used for generation (dynamic): 870400 bytes */
extern void S4__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S9__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S12__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S14_expr_1_multiple_1(
		f16 * __restrict__ expr_1_in_0,
		f16 * __restrict__ expr_1_in_1,
		f16 * __restrict__ expr_1_out_0);
extern void S18__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S21__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S24__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S29__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S32__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S35__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S37_expr_7_multiple_1(
		f16 * __restrict__ expr_7_in_0,
		f16 * __restrict__ expr_7_in_1,
		f16 * __restrict__ expr_7_out_0);
extern void S41__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S44__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S47__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S52__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S55__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S58__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S60_expr_12_multiple_1(
		f16 * __restrict__ expr_12_in_0,
		f16 * __restrict__ expr_12_in_1,
		f16 * __restrict__ expr_12_out_0);
extern void S64__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S67__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S70__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S75__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S78__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S82__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S83__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
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
extern void S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S317__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S322__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S327__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S332__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S337__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S342__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S347__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S352__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S357__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S376__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S379__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S380__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S381__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S382__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S383__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S384__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S385__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S387_expr_15_multiple_1(
		f16 * __restrict__ expr_15_in_0,
		f16 * __restrict__ expr_15_in_1,
		f16 * __restrict__ expr_15_out_0);
extern void S388__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S390__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S391__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S393_expr_16_multiple_1(
		f16 * __restrict__ expr_16_in_0,
		f16 * __restrict__ expr_16_in_1,
		f16 * __restrict__ expr_16_out_0);
extern void S395__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S396__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S397__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S399_expr_17_multiple_1(
		f16 * __restrict__ expr_17_in_0,
		f16 * __restrict__ expr_17_in_1,
		f16 * __restrict__ expr_17_out_0);
extern void S401__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S404__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S405__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
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
extern void S406__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S410__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S411__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S415__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S416__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S420__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S421__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S425__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S426__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S430__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S431__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S435__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S436__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S440__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S441__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S445__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S446__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S450__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S451__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S455__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S456__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S470__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S473__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S474__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S475__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S476__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S477__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S478__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S479__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S481_expr_22_multiple_1(
		f16 * __restrict__ expr_22_in_0,
		f16 * __restrict__ expr_22_in_1,
		f16 * __restrict__ expr_22_out_0);
extern void S482__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S484__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S485__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S487_expr_18_multiple_1(
		f16 * __restrict__ expr_18_in_0,
		f16 * __restrict__ expr_18_in_1,
		f16 * __restrict__ expr_18_out_0);
extern void S489__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S490__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S491__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S493_expr_19_multiple_1(
		f16 * __restrict__ expr_19_in_0,
		f16 * __restrict__ expr_19_in_1,
		f16 * __restrict__ expr_19_out_0);
extern void S495__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S496__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S497__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S500__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S503__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S504__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
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
extern void S505__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S509__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S510__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S514__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S515__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S519__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S520__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S524__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S525__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S529__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S530__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S534__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S535__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S539__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S540__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S544__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S545__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S549__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S550__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S554__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S555__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S559__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S560__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S564__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S565__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S569__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S570__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S574__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S575__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S579__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S580__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S584__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S585__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S589__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S590__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S594__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S595__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S599__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S600__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S604__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S605__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S609__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S610__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S614__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S615__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S629__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S632__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S633__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S634__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S635__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S636__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S637__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S638__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S640_expr_27_multiple_1(
		f16 * __restrict__ expr_27_in_0,
		f16 * __restrict__ expr_27_in_1,
		f16 * __restrict__ expr_27_out_0);
extern void S641__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S643__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S644__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S646_expr_28_multiple_1(
		f16 * __restrict__ expr_28_in_0,
		f16 * __restrict__ expr_28_in_1,
		f16 * __restrict__ expr_28_out_0);
extern void S648__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S649__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S650__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S652_expr_29_multiple_1(
		f16 * __restrict__ expr_29_in_0,
		f16 * __restrict__ expr_29_in_1,
		f16 * __restrict__ expr_29_out_0);
extern void S654__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S657__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S658__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1(
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
extern void S659__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S663__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S664__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S668__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S669__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S673__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S674__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S678__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S679__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S683__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S684__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S688__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S689__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S693__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S694__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S698__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S699__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S703__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S704__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S708__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S709__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S713__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S714__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S718__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S719__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S723__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S724__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S728__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S729__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S733__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S734__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S738__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S739__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S743__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S744__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S748__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S749__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S753__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S754__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S758__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S759__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S763__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S764__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S768__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S769__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S783__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S786__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S787__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S788__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S789__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S790__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S791__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1(
		f16 * __restrict__ In2,
		f16 * __restrict__ In1,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S792__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S794_expr_34_multiple_1(
		f16 * __restrict__ expr_34_in_0,
		f16 * __restrict__ expr_34_in_1,
		f16 * __restrict__ expr_34_out_0);
extern void S795__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S797__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S798__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S800_expr_30_multiple_1(
		f16 * __restrict__ expr_30_in_0,
		f16 * __restrict__ expr_30_in_1,
		f16 * __restrict__ expr_30_out_0);
extern void S802__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S803__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S804__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S806_expr_31_multiple_1(
		f16 * __restrict__ expr_31_in_0,
		f16 * __restrict__ expr_31_in_1,
		f16 * __restrict__ expr_31_out_0);
extern void S808__backbone_lateral_conv0_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S811__backbone_upsample_Resize_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S813__backbone_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S815__backbone_C3_p4_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S816__backbone_C3_p4_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void S818_expr_38_multiple_1(
		f16 * __restrict__ expr_38_in_0,
		f16 * __restrict__ expr_38_out_0);
extern void S820__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S821__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S822__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S823__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S824__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S825__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S826__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S827__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S828__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S831_expr_48_multiple_1(
		f16 * __restrict__ expr_48_in_0,
		f16 * __restrict__ expr_48_out_0);
extern void S832__backbone_C3_p4_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S834__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S835__backbone_reduce_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S838__backbone_upsample_1_Resize_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out);
extern void S840__backbone_Concat_1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S842__backbone_C3_p3_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S843__backbone_C3_p3_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void S845_expr_51_multiple_1(
		f16 * __restrict__ expr_51_in_0,
		f16 * __restrict__ expr_51_out_0);
extern void S847__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S848__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S849__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S850__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S851__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S852__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S853__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S854__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S855__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S858_expr_61_multiple_1(
		f16 * __restrict__ expr_61_in_0,
		f16 * __restrict__ expr_61_out_0);
extern void S859__backbone_C3_p3_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S861__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S862__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S863__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S865__head_stems_0_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S866__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S867__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S868__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S869__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S870__head_cls_preds_0_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S872__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S873__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S874__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S875__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S876__head_reg_preds_0_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S878__head_obj_preds_0_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S880__head_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void S882__backbone_Concat_2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S884__backbone_C3_n3_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S885__backbone_C3_n3_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void S887_expr_74_multiple_1(
		f16 * __restrict__ expr_74_in_0,
		f16 * __restrict__ expr_74_out_0);
extern void S889__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S890__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S891__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S892__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S893__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S894__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S895__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S896__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S897__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S900_expr_84_multiple_1(
		f16 * __restrict__ expr_84_in_0,
		f16 * __restrict__ expr_84_out_0);
extern void S901__backbone_C3_n3_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S903__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S904__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S905__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S907__head_stems_1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S908__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S909__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S910__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S911__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S912__head_cls_preds_1_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S914__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S915__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S916__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S917__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S918__head_reg_preds_1_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S920__head_obj_preds_1_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S922__head_Concat_1_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void S924__backbone_Concat_3_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S926__backbone_C3_n4_conv1_conv_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S927__backbone_C3_n4_conv1_conv_Conv_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2);
extern void S929_expr_97_multiple_1(
		f16 * __restrict__ expr_97_in_0,
		f16 * __restrict__ expr_97_out_0);
extern void S931__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S932__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S933__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S934__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S935__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S936__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S937__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S938__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S939__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S942_expr_107_multiple_1(
		f16 * __restrict__ expr_107_in_0,
		f16 * __restrict__ expr_107_out_0);
extern void S943__backbone_C3_n4_Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern void S945__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S946__head_stems_2_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S947__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S948__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S949__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S950__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S951__head_cls_preds_2_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S953__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S954__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S955__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Filter,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S956__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S957__head_reg_preds_2_Conv_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S959__head_obj_preds_2_Conv_fusion_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Bias,
		f16 * __restrict__ Out);
extern void S961__head_Concat_2_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ In3,
		f16 * __restrict__ Out);
extern void S964__Slice_split_multiple_1(
		f16 * __restrict__ In,
		f16 * __restrict__ Out1,
		f16 * __restrict__ Out2,
		f16 * __restrict__ Out3);
extern void S965_expr_118_multiple_1(
		f16 * __restrict__ expr_118_in_0,
		f16 * __restrict__ expr_118_in_1,
		f16 * __restrict__ expr_118_out_0);
extern void S966__Concat_multiple_1(
		f16 * __restrict__ In1,
		f16 * __restrict__ In2,
		f16 * __restrict__ Out);
extern int onnx_graphCNN_Construct(int a);
extern void onnx_graphCNN_ConstructCluster();
extern int onnx_graphCNN_Destruct();
extern int onnx_graphCNN_Memory(AT_MEM_TYPE Which);
extern f16 * __restrict__ Input_1;
extern f16 * __restrict__ Output_1;
extern int onnx_graphCNN(
);
extern unsigned int AT_GraphPerf[375];
extern unsigned int AT_GraphPerf_CNN_Total;
extern char * AT_GraphNodeNames[375];
extern unsigned int AT_GraphOperInfosNames[375];
#endif
