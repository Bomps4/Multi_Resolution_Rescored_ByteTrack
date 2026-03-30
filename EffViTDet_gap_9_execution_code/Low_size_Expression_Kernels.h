#ifndef Low_size_ONNX_GRAPH_BASIC_KERNELS_H
#define Low_size_ONNX_GRAPH_BASIC_KERNELS_H
#include "at_api.h"
#include "DspLib.h"
#include "FloatDefines.h"
#include "FastFloatApprox16.h"
#include "FastFloatApprox.h"


typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_1_in_0;
    f16 *__restrict__  Low_size_expr_1_in_1;
    f16 *__restrict__  Low_size_expr_1_out_0;
} Low_size_s14_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_7_in_0;
    f16 *__restrict__  Low_size_expr_7_in_1;
    f16 *__restrict__  Low_size_expr_7_out_0;
} Low_size_s37_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_12_in_0;
    f16 *__restrict__  Low_size_expr_12_in_1;
    f16 *__restrict__  Low_size_expr_12_out_0;
} Low_size_s60_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  Low_size_expr_15_in_0;
    f16 *__restrict__  Low_size_expr_15_in_1;
    f16 *__restrict__  Low_size_expr_15_out_0;
} Low_size_s387_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_16_in_0;
    f16 *__restrict__  Low_size_expr_16_in_1;
    f16 *__restrict__  Low_size_expr_16_out_0;
} Low_size_s393_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_17_in_0;
    f16 *__restrict__  Low_size_expr_17_in_1;
    f16 *__restrict__  Low_size_expr_17_out_0;
} Low_size_s399_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_18_in_0;
    f16 *__restrict__  Low_size_expr_18_in_1;
    f16 *__restrict__  Low_size_expr_18_out_0;
} Low_size_s487_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_19_in_0;
    f16 *__restrict__  Low_size_expr_19_in_1;
    f16 *__restrict__  Low_size_expr_19_out_0;
} Low_size_s493_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  Low_size_expr_22_in_0;
    f16 *__restrict__  Low_size_expr_22_in_1;
    f16 *__restrict__  Low_size_expr_22_out_0;
} Low_size_s481_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  Low_size_expr_27_in_0;
    f16 *__restrict__  Low_size_expr_27_in_1;
    f16 *__restrict__  Low_size_expr_27_out_0;
} Low_size_s640_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_28_in_0;
    f16 *__restrict__  Low_size_expr_28_in_1;
    f16 *__restrict__  Low_size_expr_28_out_0;
} Low_size_s646_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_29_in_0;
    f16 *__restrict__  Low_size_expr_29_in_1;
    f16 *__restrict__  Low_size_expr_29_out_0;
} Low_size_s652_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_30_in_0;
    f16 *__restrict__  Low_size_expr_30_in_1;
    f16 *__restrict__  Low_size_expr_30_out_0;
} Low_size_s800_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_31_in_0;
    f16 *__restrict__  Low_size_expr_31_in_1;
    f16 *__restrict__  Low_size_expr_31_out_0;
} Low_size_s806_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  Low_size_expr_34_in_0;
    f16 *__restrict__  Low_size_expr_34_in_1;
    f16 *__restrict__  Low_size_expr_34_out_0;
} Low_size_s794_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_38_in_0;
    f16 *__restrict__  Low_size_expr_38_out_0;
} Low_size_s818_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_48_in_0;
    f16 *__restrict__  Low_size_expr_48_out_0;
} Low_size_s831_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_51_in_0;
    f16 *__restrict__  Low_size_expr_51_out_0;
} Low_size_s845_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_61_in_0;
    f16 *__restrict__  Low_size_expr_61_out_0;
} Low_size_s858_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_74_in_0;
    f16 *__restrict__  Low_size_expr_74_out_0;
} Low_size_s887_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_84_in_0;
    f16 *__restrict__  Low_size_expr_84_out_0;
} Low_size_s900_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_97_in_0;
    f16 *__restrict__  Low_size_expr_97_out_0;
} Low_size_s929_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  Low_size_expr_107_in_0;
    f16 *__restrict__  Low_size_expr_107_out_0;
} Low_size_s942_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  Low_size_expr_118_in_0;
    f16 *__restrict__  Low_size_expr_118_in_1;
    f16 *__restrict__  Low_size_expr_118_out_0;
} Low_size_s965_multiple_1_kernel_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_0_in_0;
    f16 *__restrict__  Low_size_expr_0_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_0multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_2_in_0;
    f16 *__restrict__  Low_size_expr_2_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_2multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_3_in_0;
    f16 *__restrict__  Low_size_expr_3_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_3multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_4_in_0;
    f16 *__restrict__  Low_size_expr_4_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_4multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_5_in_0;
    f16 *__restrict__  Low_size_expr_5_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_5multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_6_in_0;
    f16 *__restrict__  Low_size_expr_6_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_6multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_8_in_0;
    f16 *__restrict__  Low_size_expr_8_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_8multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_9_in_0;
    f16 *__restrict__  Low_size_expr_9_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_9multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_10_in_0;
    f16 *__restrict__  Low_size_expr_10_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_10multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_11_in_0;
    f16 *__restrict__  Low_size_expr_11_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_11multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_13_in_0;
    f16 *__restrict__  Low_size_expr_13_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_13multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_14_in_0;
    f16 *__restrict__  Low_size_expr_14_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_14multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_20_in_0;
    f16 *__restrict__  Low_size_expr_20_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_20multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_21_in_0;
    f16 *__restrict__  Low_size_expr_21_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_21multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_23_in_0;
    f16 *__restrict__  Low_size_expr_23_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_23multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_24_in_0;
    f16 *__restrict__  Low_size_expr_24_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_24multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_25_in_0;
    f16 *__restrict__  Low_size_expr_25_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_25multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_26_in_0;
    f16 *__restrict__  Low_size_expr_26_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_26multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_32_in_0;
    f16 *__restrict__  Low_size_expr_32_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_32multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_33_in_0;
    f16 *__restrict__  Low_size_expr_33_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_33multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_35_in_0;
    f16 *__restrict__  Low_size_expr_35_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_35multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_36_in_0;
    f16 *__restrict__  Low_size_expr_36_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_36multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_37_in_0;
    f16 *__restrict__  Low_size_expr_37_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_37multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_39_in_0;
    f16 *__restrict__  Low_size_expr_39_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_39multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_40_in_0;
    f16 *__restrict__  Low_size_expr_40_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_40multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_41_in_0;
    f16 *__restrict__  Low_size_expr_41_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_41multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_42_in_0;
    f16 *__restrict__  Low_size_expr_42_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_42multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_43_in_0;
    f16 *__restrict__  Low_size_expr_43_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_43multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_44_in_0;
    f16 *__restrict__  Low_size_expr_44_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_44multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_45_in_0;
    f16 *__restrict__  Low_size_expr_45_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_45multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_46_in_0;
    f16 *__restrict__  Low_size_expr_46_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_46multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_47_in_0;
    f16 *__restrict__  Low_size_expr_47_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_47multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_49_in_0;
    f16 *__restrict__  Low_size_expr_49_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_49multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_50_in_0;
    f16 *__restrict__  Low_size_expr_50_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_50multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_52_in_0;
    f16 *__restrict__  Low_size_expr_52_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_52multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_53_in_0;
    f16 *__restrict__  Low_size_expr_53_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_53multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_54_in_0;
    f16 *__restrict__  Low_size_expr_54_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_54multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_55_in_0;
    f16 *__restrict__  Low_size_expr_55_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_55multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_56_in_0;
    f16 *__restrict__  Low_size_expr_56_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_56multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_57_in_0;
    f16 *__restrict__  Low_size_expr_57_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_57multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_58_in_0;
    f16 *__restrict__  Low_size_expr_58_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_58multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_59_in_0;
    f16 *__restrict__  Low_size_expr_59_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_59multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_60_in_0;
    f16 *__restrict__  Low_size_expr_60_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_60multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_62_in_0;
    f16 *__restrict__  Low_size_expr_62_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_62multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_63_in_0;
    f16 *__restrict__  Low_size_expr_63_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_63multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_64_in_0;
    f16 *__restrict__  Low_size_expr_64_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_64multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_75_in_0;
    f16 *__restrict__  Low_size_expr_75_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_75multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_76_in_0;
    f16 *__restrict__  Low_size_expr_76_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_76multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_77_in_0;
    f16 *__restrict__  Low_size_expr_77_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_77multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_78_in_0;
    f16 *__restrict__  Low_size_expr_78_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_78multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_79_in_0;
    f16 *__restrict__  Low_size_expr_79_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_79multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_80_in_0;
    f16 *__restrict__  Low_size_expr_80_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_80multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_81_in_0;
    f16 *__restrict__  Low_size_expr_81_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_81multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_82_in_0;
    f16 *__restrict__  Low_size_expr_82_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_82multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_83_in_0;
    f16 *__restrict__  Low_size_expr_83_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_83multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_85_in_0;
    f16 *__restrict__  Low_size_expr_85_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_85multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_86_in_0;
    f16 *__restrict__  Low_size_expr_86_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_86multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_87_in_0;
    f16 *__restrict__  Low_size_expr_87_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_87multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_98_in_0;
    f16 *__restrict__  Low_size_expr_98_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_98multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_99_in_0;
    f16 *__restrict__  Low_size_expr_99_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_99multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_100_in_0;
    f16 *__restrict__  Low_size_expr_100_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_100multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_101_in_0;
    f16 *__restrict__  Low_size_expr_101_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_101multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_102_in_0;
    f16 *__restrict__  Low_size_expr_102_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_102multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_103_in_0;
    f16 *__restrict__  Low_size_expr_103_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_103multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_104_in_0;
    f16 *__restrict__  Low_size_expr_104_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_104multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_105_in_0;
    f16 *__restrict__  Low_size_expr_105_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_105multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_106_in_0;
    f16 *__restrict__  Low_size_expr_106_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_106multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_108_in_0;
    f16 *__restrict__  Low_size_expr_108_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_108multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_65_in_0;
    f16 *__restrict__  Low_size_expr_65_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_65multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_66_in_0;
    f16 *__restrict__  Low_size_expr_66_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_66multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_67_in_0;
    f16 *__restrict__  Low_size_expr_67_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_67multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_68_in_0;
    f16 *__restrict__  Low_size_expr_68_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_68multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_69_in_0;
    f16 *__restrict__  Low_size_expr_69_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_69multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_70_in_0;
    f16 *__restrict__  Low_size_expr_70_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_70multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_71_in_0;
    f16 *__restrict__  Low_size_expr_71_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_71multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_72_in_0;
    f16 *__restrict__  Low_size_expr_72_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_72multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_73_in_0;
    f16 *__restrict__  Low_size_expr_73_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_73multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_88_in_0;
    f16 *__restrict__  Low_size_expr_88_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_88multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_89_in_0;
    f16 *__restrict__  Low_size_expr_89_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_89multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_90_in_0;
    f16 *__restrict__  Low_size_expr_90_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_90multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_91_in_0;
    f16 *__restrict__  Low_size_expr_91_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_91multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_92_in_0;
    f16 *__restrict__  Low_size_expr_92_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_92multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_93_in_0;
    f16 *__restrict__  Low_size_expr_93_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_93multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_94_in_0;
    f16 *__restrict__  Low_size_expr_94_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_94multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_95_in_0;
    f16 *__restrict__  Low_size_expr_95_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_95multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_96_in_0;
    f16 *__restrict__  Low_size_expr_96_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_96multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_109_in_0;
    f16 *__restrict__  Low_size_expr_109_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_109multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_110_in_0;
    f16 *__restrict__  Low_size_expr_110_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_110multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_111_in_0;
    f16 *__restrict__  Low_size_expr_111_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_111multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_112_in_0;
    f16 *__restrict__  Low_size_expr_112_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_112multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_113_in_0;
    f16 *__restrict__  Low_size_expr_113_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_113multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_114_in_0;
    f16 *__restrict__  Low_size_expr_114_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_114multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_115_in_0;
    f16 *__restrict__  Low_size_expr_115_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_115multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_116_in_0;
    f16 *__restrict__  Low_size_expr_116_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_116multiple_1_args_t;

typedef struct {
    f16 *__restrict__  Low_size_expr_117_in_0;
    f16 *__restrict__  Low_size_expr_117_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} Low_size_expr_117multiple_1_args_t;


void Low_size_s14_multiple_1_kernel(Low_size_s14_multiple_1_kernel_args_t *Args);

void Low_size_s37_multiple_1_kernel(Low_size_s37_multiple_1_kernel_args_t *Args);

void Low_size_s60_multiple_1_kernel(Low_size_s60_multiple_1_kernel_args_t *Args);

void Low_size_s387_multiple_1_kernel(Low_size_s387_multiple_1_kernel_args_t *Args);

void Low_size_s393_multiple_1_kernel(Low_size_s393_multiple_1_kernel_args_t *Args);

void Low_size_s399_multiple_1_kernel(Low_size_s399_multiple_1_kernel_args_t *Args);

void Low_size_s487_multiple_1_kernel(Low_size_s487_multiple_1_kernel_args_t *Args);

void Low_size_s493_multiple_1_kernel(Low_size_s493_multiple_1_kernel_args_t *Args);

void Low_size_s481_multiple_1_kernel(Low_size_s481_multiple_1_kernel_args_t *Args);

void Low_size_s640_multiple_1_kernel(Low_size_s640_multiple_1_kernel_args_t *Args);

void Low_size_s646_multiple_1_kernel(Low_size_s646_multiple_1_kernel_args_t *Args);

void Low_size_s652_multiple_1_kernel(Low_size_s652_multiple_1_kernel_args_t *Args);

void Low_size_s800_multiple_1_kernel(Low_size_s800_multiple_1_kernel_args_t *Args);

void Low_size_s806_multiple_1_kernel(Low_size_s806_multiple_1_kernel_args_t *Args);

void Low_size_s794_multiple_1_kernel(Low_size_s794_multiple_1_kernel_args_t *Args);

void Low_size_s818_multiple_1_kernel(Low_size_s818_multiple_1_kernel_args_t *Args);

void Low_size_s831_multiple_1_kernel(Low_size_s831_multiple_1_kernel_args_t *Args);

void Low_size_s845_multiple_1_kernel(Low_size_s845_multiple_1_kernel_args_t *Args);

void Low_size_s858_multiple_1_kernel(Low_size_s858_multiple_1_kernel_args_t *Args);

void Low_size_s887_multiple_1_kernel(Low_size_s887_multiple_1_kernel_args_t *Args);

void Low_size_s900_multiple_1_kernel(Low_size_s900_multiple_1_kernel_args_t *Args);

void Low_size_s929_multiple_1_kernel(Low_size_s929_multiple_1_kernel_args_t *Args);

void Low_size_s942_multiple_1_kernel(Low_size_s942_multiple_1_kernel_args_t *Args);

void Low_size_s965_multiple_1_kernel(Low_size_s965_multiple_1_kernel_args_t *Args);

void Low_size_expr_0multiple_1(Low_size_expr_0multiple_1_args_t *Args);

void Low_size_expr_2multiple_1(Low_size_expr_2multiple_1_args_t *Args);

void Low_size_expr_3multiple_1(Low_size_expr_3multiple_1_args_t *Args);

void Low_size_expr_4multiple_1(Low_size_expr_4multiple_1_args_t *Args);

void Low_size_expr_5multiple_1(Low_size_expr_5multiple_1_args_t *Args);

void Low_size_expr_6multiple_1(Low_size_expr_6multiple_1_args_t *Args);

void Low_size_expr_8multiple_1(Low_size_expr_8multiple_1_args_t *Args);

void Low_size_expr_9multiple_1(Low_size_expr_9multiple_1_args_t *Args);

void Low_size_expr_10multiple_1(Low_size_expr_10multiple_1_args_t *Args);

void Low_size_expr_11multiple_1(Low_size_expr_11multiple_1_args_t *Args);

void Low_size_expr_13multiple_1(Low_size_expr_13multiple_1_args_t *Args);

void Low_size_expr_14multiple_1(Low_size_expr_14multiple_1_args_t *Args);

void Low_size_expr_20multiple_1(Low_size_expr_20multiple_1_args_t *Args);

void Low_size_expr_21multiple_1(Low_size_expr_21multiple_1_args_t *Args);

void Low_size_expr_23multiple_1(Low_size_expr_23multiple_1_args_t *Args);

void Low_size_expr_24multiple_1(Low_size_expr_24multiple_1_args_t *Args);

void Low_size_expr_25multiple_1(Low_size_expr_25multiple_1_args_t *Args);

void Low_size_expr_26multiple_1(Low_size_expr_26multiple_1_args_t *Args);

void Low_size_expr_32multiple_1(Low_size_expr_32multiple_1_args_t *Args);

void Low_size_expr_33multiple_1(Low_size_expr_33multiple_1_args_t *Args);

void Low_size_expr_35multiple_1(Low_size_expr_35multiple_1_args_t *Args);

void Low_size_expr_36multiple_1(Low_size_expr_36multiple_1_args_t *Args);

void Low_size_expr_37multiple_1(Low_size_expr_37multiple_1_args_t *Args);

void Low_size_expr_39multiple_1(Low_size_expr_39multiple_1_args_t *Args);

void Low_size_expr_40multiple_1(Low_size_expr_40multiple_1_args_t *Args);

void Low_size_expr_41multiple_1(Low_size_expr_41multiple_1_args_t *Args);

void Low_size_expr_42multiple_1(Low_size_expr_42multiple_1_args_t *Args);

void Low_size_expr_43multiple_1(Low_size_expr_43multiple_1_args_t *Args);

void Low_size_expr_44multiple_1(Low_size_expr_44multiple_1_args_t *Args);

void Low_size_expr_45multiple_1(Low_size_expr_45multiple_1_args_t *Args);

void Low_size_expr_46multiple_1(Low_size_expr_46multiple_1_args_t *Args);

void Low_size_expr_47multiple_1(Low_size_expr_47multiple_1_args_t *Args);

void Low_size_expr_49multiple_1(Low_size_expr_49multiple_1_args_t *Args);

void Low_size_expr_50multiple_1(Low_size_expr_50multiple_1_args_t *Args);

void Low_size_expr_52multiple_1(Low_size_expr_52multiple_1_args_t *Args);

void Low_size_expr_53multiple_1(Low_size_expr_53multiple_1_args_t *Args);

void Low_size_expr_54multiple_1(Low_size_expr_54multiple_1_args_t *Args);

void Low_size_expr_55multiple_1(Low_size_expr_55multiple_1_args_t *Args);

void Low_size_expr_56multiple_1(Low_size_expr_56multiple_1_args_t *Args);

void Low_size_expr_57multiple_1(Low_size_expr_57multiple_1_args_t *Args);

void Low_size_expr_58multiple_1(Low_size_expr_58multiple_1_args_t *Args);

void Low_size_expr_59multiple_1(Low_size_expr_59multiple_1_args_t *Args);

void Low_size_expr_60multiple_1(Low_size_expr_60multiple_1_args_t *Args);

void Low_size_expr_62multiple_1(Low_size_expr_62multiple_1_args_t *Args);

void Low_size_expr_63multiple_1(Low_size_expr_63multiple_1_args_t *Args);

void Low_size_expr_64multiple_1(Low_size_expr_64multiple_1_args_t *Args);

void Low_size_expr_75multiple_1(Low_size_expr_75multiple_1_args_t *Args);

void Low_size_expr_76multiple_1(Low_size_expr_76multiple_1_args_t *Args);

void Low_size_expr_77multiple_1(Low_size_expr_77multiple_1_args_t *Args);

void Low_size_expr_78multiple_1(Low_size_expr_78multiple_1_args_t *Args);

void Low_size_expr_79multiple_1(Low_size_expr_79multiple_1_args_t *Args);

void Low_size_expr_80multiple_1(Low_size_expr_80multiple_1_args_t *Args);

void Low_size_expr_81multiple_1(Low_size_expr_81multiple_1_args_t *Args);

void Low_size_expr_82multiple_1(Low_size_expr_82multiple_1_args_t *Args);

void Low_size_expr_83multiple_1(Low_size_expr_83multiple_1_args_t *Args);

void Low_size_expr_85multiple_1(Low_size_expr_85multiple_1_args_t *Args);

void Low_size_expr_86multiple_1(Low_size_expr_86multiple_1_args_t *Args);

void Low_size_expr_87multiple_1(Low_size_expr_87multiple_1_args_t *Args);

void Low_size_expr_98multiple_1(Low_size_expr_98multiple_1_args_t *Args);

void Low_size_expr_99multiple_1(Low_size_expr_99multiple_1_args_t *Args);

void Low_size_expr_100multiple_1(Low_size_expr_100multiple_1_args_t *Args);

void Low_size_expr_101multiple_1(Low_size_expr_101multiple_1_args_t *Args);

void Low_size_expr_102multiple_1(Low_size_expr_102multiple_1_args_t *Args);

void Low_size_expr_103multiple_1(Low_size_expr_103multiple_1_args_t *Args);

void Low_size_expr_104multiple_1(Low_size_expr_104multiple_1_args_t *Args);

void Low_size_expr_105multiple_1(Low_size_expr_105multiple_1_args_t *Args);

void Low_size_expr_106multiple_1(Low_size_expr_106multiple_1_args_t *Args);

void Low_size_expr_108multiple_1(Low_size_expr_108multiple_1_args_t *Args);

void Low_size_expr_65multiple_1(Low_size_expr_65multiple_1_args_t *Args);

void Low_size_expr_66multiple_1(Low_size_expr_66multiple_1_args_t *Args);

void Low_size_expr_67multiple_1(Low_size_expr_67multiple_1_args_t *Args);

void Low_size_expr_68multiple_1(Low_size_expr_68multiple_1_args_t *Args);

void Low_size_expr_69multiple_1(Low_size_expr_69multiple_1_args_t *Args);

void Low_size_expr_70multiple_1(Low_size_expr_70multiple_1_args_t *Args);

void Low_size_expr_71multiple_1(Low_size_expr_71multiple_1_args_t *Args);

void Low_size_expr_72multiple_1(Low_size_expr_72multiple_1_args_t *Args);

void Low_size_expr_73multiple_1(Low_size_expr_73multiple_1_args_t *Args);

void Low_size_expr_88multiple_1(Low_size_expr_88multiple_1_args_t *Args);

void Low_size_expr_89multiple_1(Low_size_expr_89multiple_1_args_t *Args);

void Low_size_expr_90multiple_1(Low_size_expr_90multiple_1_args_t *Args);

void Low_size_expr_91multiple_1(Low_size_expr_91multiple_1_args_t *Args);

void Low_size_expr_92multiple_1(Low_size_expr_92multiple_1_args_t *Args);

void Low_size_expr_93multiple_1(Low_size_expr_93multiple_1_args_t *Args);

void Low_size_expr_94multiple_1(Low_size_expr_94multiple_1_args_t *Args);

void Low_size_expr_95multiple_1(Low_size_expr_95multiple_1_args_t *Args);

void Low_size_expr_96multiple_1(Low_size_expr_96multiple_1_args_t *Args);

void Low_size_expr_109multiple_1(Low_size_expr_109multiple_1_args_t *Args);

void Low_size_expr_110multiple_1(Low_size_expr_110multiple_1_args_t *Args);

void Low_size_expr_111multiple_1(Low_size_expr_111multiple_1_args_t *Args);

void Low_size_expr_112multiple_1(Low_size_expr_112multiple_1_args_t *Args);

void Low_size_expr_113multiple_1(Low_size_expr_113multiple_1_args_t *Args);

void Low_size_expr_114multiple_1(Low_size_expr_114multiple_1_args_t *Args);

void Low_size_expr_115multiple_1(Low_size_expr_115multiple_1_args_t *Args);

void Low_size_expr_116multiple_1(Low_size_expr_116multiple_1_args_t *Args);

void Low_size_expr_117multiple_1(Low_size_expr_117multiple_1_args_t *Args);


#endif // ONNX_GRAPH_BASIC_KERNELS_H