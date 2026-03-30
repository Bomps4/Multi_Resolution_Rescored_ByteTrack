#ifndef ONNX_GRAPH_BASIC_KERNELS_H
#define ONNX_GRAPH_BASIC_KERNELS_H
#include "at_api.h"
#include "DspLib.h"
#include "FloatDefines.h"
#include "FastFloatApprox.h"
#include "FastFloatApprox16.h"

typedef struct SplitWidthIn12Arg_S {
    void * In;
    unsigned char DataSize;
    unsigned short int H;
    unsigned short int InWidth;
    unsigned short int S1;
    unsigned short int W1;
    void * Out1;
    unsigned short int S2;
    unsigned short int W2;
    void * Out2;
    unsigned short int S3;
    unsigned short int W3;
    void * Out3;
    unsigned short int S4;
    unsigned short int W4;
    void * Out4;
    unsigned short int S5;
    unsigned short int W5;
    void * Out5;
    unsigned short int S6;
    unsigned short int W6;
    void * Out6;
    unsigned short int S7;
    unsigned short int W7;
    void * Out7;
    unsigned short int S8;
    unsigned short int W8;
    void * Out8;
    unsigned short int S9;
    unsigned short int W9;
    void * Out9;
    unsigned short int S10;
    unsigned short int W10;
    void * Out10;
    unsigned short int S11;
    unsigned short int W11;
    void * Out11;
    unsigned short int S12;
    unsigned short int W12;
    void * Out12;
} SplitWidthIn12Arg_T;


void CNN_Split_Width_In12(SplitWidthIn12Arg_T *Arg);
void CNN_ParSplit_Width_In12(SplitWidthIn12Arg_T *Arg);

typedef struct ConcatWidthIn13Arg_S {
    void * Out;
    int DataSize;
    int H;
    int W1;
    void * In1;
    int W2;
    void * In2;
    int W3;
    void * In3;
    int W4;
    void * In4;
    int W5;
    void * In5;
    int W6;
    void * In6;
    int W7;
    void * In7;
    int W8;
    void * In8;
    int W9;
    void * In9;
    int W10;
    void * In10;
    int W11;
    void * In11;
    int W12;
    void * In12;
    int W13;
    void * In13;
} ConcatWidthIn13Arg_T;

void CNN_Concat_Width_In13(ConcatWidthIn13Arg_T *Arg);

typedef struct SplitWidthIn24Arg_S {
    void * In;
    unsigned char DataSize;
    unsigned short int H;
    unsigned short int InWidth;
    unsigned short int S1;
    unsigned short int W1;
    void * Out1;
    unsigned short int S2;
    unsigned short int W2;
    void * Out2;
    unsigned short int S3;
    unsigned short int W3;
    void * Out3;
    unsigned short int S4;
    unsigned short int W4;
    void * Out4;
    unsigned short int S5;
    unsigned short int W5;
    void * Out5;
    unsigned short int S6;
    unsigned short int W6;
    void * Out6;
    unsigned short int S7;
    unsigned short int W7;
    void * Out7;
    unsigned short int S8;
    unsigned short int W8;
    void * Out8;
    unsigned short int S9;
    unsigned short int W9;
    void * Out9;
    unsigned short int S10;
    unsigned short int W10;
    void * Out10;
    unsigned short int S11;
    unsigned short int W11;
    void * Out11;
    unsigned short int S12;
    unsigned short int W12;
    void * Out12;
    unsigned short int S13;
    unsigned short int W13;
    void * Out13;
    unsigned short int S14;
    unsigned short int W14;
    void * Out14;
    unsigned short int S15;
    unsigned short int W15;
    void * Out15;
    unsigned short int S16;
    unsigned short int W16;
    void * Out16;
    unsigned short int S17;
    unsigned short int W17;
    void * Out17;
    unsigned short int S18;
    unsigned short int W18;
    void * Out18;
    unsigned short int S19;
    unsigned short int W19;
    void * Out19;
    unsigned short int S20;
    unsigned short int W20;
    void * Out20;
    unsigned short int S21;
    unsigned short int W21;
    void * Out21;
    unsigned short int S22;
    unsigned short int W22;
    void * Out22;
    unsigned short int S23;
    unsigned short int W23;
    void * Out23;
    unsigned short int S24;
    unsigned short int W24;
    void * Out24;
} SplitWidthIn24Arg_T;


void CNN_Split_Width_In24(SplitWidthIn24Arg_T *Arg);
void CNN_ParSplit_Width_In24(SplitWidthIn24Arg_T *Arg);

typedef struct ConcatWidthIn25Arg_S {
    void * Out;
    int DataSize;
    int H;
    int W1;
    void * In1;
    int W2;
    void * In2;
    int W3;
    void * In3;
    int W4;
    void * In4;
    int W5;
    void * In5;
    int W6;
    void * In6;
    int W7;
    void * In7;
    int W8;
    void * In8;
    int W9;
    void * In9;
    int W10;
    void * In10;
    int W11;
    void * In11;
    int W12;
    void * In12;
    int W13;
    void * In13;
    int W14;
    void * In14;
    int W15;
    void * In15;
    int W16;
    void * In16;
    int W17;
    void * In17;
    int W18;
    void * In18;
    int W19;
    void * In19;
    int W20;
    void * In20;
    int W21;
    void * In21;
    int W22;
    void * In22;
    int W23;
    void * In23;
    int W24;
    void * In24;
    int W25;
    void * In25;
} ConcatWidthIn25Arg_T;

void CNN_Concat_Width_In25(ConcatWidthIn25Arg_T *Arg);

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_1_in_0;
    f16 *__restrict__  expr_1_in_1;
    f16 *__restrict__  expr_1_out_0;
} s14_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_7_in_0;
    f16 *__restrict__  expr_7_in_1;
    f16 *__restrict__  expr_7_out_0;
} s37_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_12_in_0;
    f16 *__restrict__  expr_12_in_1;
    f16 *__restrict__  expr_12_out_0;
} s60_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  expr_15_in_0;
    f16 *__restrict__  expr_15_in_1;
    f16 *__restrict__  expr_15_out_0;
} s387_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_16_in_0;
    f16 *__restrict__  expr_16_in_1;
    f16 *__restrict__  expr_16_out_0;
} s393_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_17_in_0;
    f16 *__restrict__  expr_17_in_1;
    f16 *__restrict__  expr_17_out_0;
} s399_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_18_in_0;
    f16 *__restrict__  expr_18_in_1;
    f16 *__restrict__  expr_18_out_0;
} s487_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_19_in_0;
    f16 *__restrict__  expr_19_in_1;
    f16 *__restrict__  expr_19_out_0;
} s493_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  expr_22_in_0;
    f16 *__restrict__  expr_22_in_1;
    f16 *__restrict__  expr_22_out_0;
} s481_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  expr_27_in_0;
    f16 *__restrict__  expr_27_in_1;
    f16 *__restrict__  expr_27_out_0;
} s640_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_28_in_0;
    f16 *__restrict__  expr_28_in_1;
    f16 *__restrict__  expr_28_out_0;
} s646_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_29_in_0;
    f16 *__restrict__  expr_29_in_1;
    f16 *__restrict__  expr_29_out_0;
} s652_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_30_in_0;
    f16 *__restrict__  expr_30_in_1;
    f16 *__restrict__  expr_30_out_0;
} s800_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_31_in_0;
    f16 *__restrict__  expr_31_in_1;
    f16 *__restrict__  expr_31_out_0;
} s806_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  expr_34_in_0;
    f16 *__restrict__  expr_34_in_1;
    f16 *__restrict__  expr_34_out_0;
} s794_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_38_in_0;
    f16 *__restrict__  expr_38_out_0;
} s818_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_48_in_0;
    f16 *__restrict__  expr_48_out_0;
} s831_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_51_in_0;
    f16 *__restrict__  expr_51_out_0;
} s845_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_61_in_0;
    f16 *__restrict__  expr_61_out_0;
} s858_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_74_in_0;
    f16 *__restrict__  expr_74_out_0;
} s887_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_84_in_0;
    f16 *__restrict__  expr_84_out_0;
} s900_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_97_in_0;
    f16 *__restrict__  expr_97_out_0;
} s929_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    f16 *__restrict__  expr_107_in_0;
    f16 *__restrict__  expr_107_out_0;
} s942_multiple_1_kernel_args_t;

typedef struct {
    unsigned int I0;
    unsigned int I1;
    f16 *__restrict__  expr_118_in_0;
    f16 *__restrict__  expr_118_in_1;
    f16 *__restrict__  expr_118_out_0;
} s965_multiple_1_kernel_args_t;

typedef struct {
    f16 *__restrict__  expr_0_in_0;
    f16 *__restrict__  expr_0_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_0multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_2_in_0;
    f16 *__restrict__  expr_2_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_2multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_3_in_0;
    f16 *__restrict__  expr_3_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_3multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_4_in_0;
    f16 *__restrict__  expr_4_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_4multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_5_in_0;
    f16 *__restrict__  expr_5_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_5multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_6_in_0;
    f16 *__restrict__  expr_6_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_6multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_8_in_0;
    f16 *__restrict__  expr_8_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_8multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_9_in_0;
    f16 *__restrict__  expr_9_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_9multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_10_in_0;
    f16 *__restrict__  expr_10_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_10multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_11_in_0;
    f16 *__restrict__  expr_11_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_11multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_13_in_0;
    f16 *__restrict__  expr_13_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_13multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_14_in_0;
    f16 *__restrict__  expr_14_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_14multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_20_in_0;
    f16 *__restrict__  expr_20_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_20multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_21_in_0;
    f16 *__restrict__  expr_21_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_21multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_23_in_0;
    f16 *__restrict__  expr_23_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_23multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_24_in_0;
    f16 *__restrict__  expr_24_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_24multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_25_in_0;
    f16 *__restrict__  expr_25_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_25multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_26_in_0;
    f16 *__restrict__  expr_26_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_26multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_32_in_0;
    f16 *__restrict__  expr_32_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_32multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_33_in_0;
    f16 *__restrict__  expr_33_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_33multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_35_in_0;
    f16 *__restrict__  expr_35_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_35multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_36_in_0;
    f16 *__restrict__  expr_36_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_36multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_37_in_0;
    f16 *__restrict__  expr_37_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_37multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_39_in_0;
    f16 *__restrict__  expr_39_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_39multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_40_in_0;
    f16 *__restrict__  expr_40_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_40multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_41_in_0;
    f16 *__restrict__  expr_41_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_41multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_42_in_0;
    f16 *__restrict__  expr_42_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_42multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_43_in_0;
    f16 *__restrict__  expr_43_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_43multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_44_in_0;
    f16 *__restrict__  expr_44_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_44multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_45_in_0;
    f16 *__restrict__  expr_45_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_45multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_46_in_0;
    f16 *__restrict__  expr_46_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_46multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_47_in_0;
    f16 *__restrict__  expr_47_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_47multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_49_in_0;
    f16 *__restrict__  expr_49_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_49multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_50_in_0;
    f16 *__restrict__  expr_50_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_50multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_52_in_0;
    f16 *__restrict__  expr_52_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_52multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_53_in_0;
    f16 *__restrict__  expr_53_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_53multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_54_in_0;
    f16 *__restrict__  expr_54_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_54multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_55_in_0;
    f16 *__restrict__  expr_55_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_55multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_56_in_0;
    f16 *__restrict__  expr_56_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_56multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_57_in_0;
    f16 *__restrict__  expr_57_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_57multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_58_in_0;
    f16 *__restrict__  expr_58_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_58multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_59_in_0;
    f16 *__restrict__  expr_59_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_59multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_60_in_0;
    f16 *__restrict__  expr_60_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_60multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_62_in_0;
    f16 *__restrict__  expr_62_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_62multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_63_in_0;
    f16 *__restrict__  expr_63_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_63multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_64_in_0;
    f16 *__restrict__  expr_64_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_64multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_75_in_0;
    f16 *__restrict__  expr_75_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_75multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_76_in_0;
    f16 *__restrict__  expr_76_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_76multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_77_in_0;
    f16 *__restrict__  expr_77_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_77multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_78_in_0;
    f16 *__restrict__  expr_78_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_78multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_79_in_0;
    f16 *__restrict__  expr_79_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_79multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_80_in_0;
    f16 *__restrict__  expr_80_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_80multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_81_in_0;
    f16 *__restrict__  expr_81_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_81multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_82_in_0;
    f16 *__restrict__  expr_82_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_82multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_83_in_0;
    f16 *__restrict__  expr_83_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_83multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_85_in_0;
    f16 *__restrict__  expr_85_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_85multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_86_in_0;
    f16 *__restrict__  expr_86_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_86multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_87_in_0;
    f16 *__restrict__  expr_87_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_87multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_98_in_0;
    f16 *__restrict__  expr_98_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_98multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_99_in_0;
    f16 *__restrict__  expr_99_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_99multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_100_in_0;
    f16 *__restrict__  expr_100_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_100multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_101_in_0;
    f16 *__restrict__  expr_101_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_101multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_102_in_0;
    f16 *__restrict__  expr_102_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_102multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_103_in_0;
    f16 *__restrict__  expr_103_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_103multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_104_in_0;
    f16 *__restrict__  expr_104_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_104multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_105_in_0;
    f16 *__restrict__  expr_105_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_105multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_106_in_0;
    f16 *__restrict__  expr_106_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_106multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_108_in_0;
    f16 *__restrict__  expr_108_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_108multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_65_in_0;
    f16 *__restrict__  expr_65_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_65multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_66_in_0;
    f16 *__restrict__  expr_66_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_66multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_67_in_0;
    f16 *__restrict__  expr_67_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_67multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_68_in_0;
    f16 *__restrict__  expr_68_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_68multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_69_in_0;
    f16 *__restrict__  expr_69_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_69multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_70_in_0;
    f16 *__restrict__  expr_70_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_70multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_71_in_0;
    f16 *__restrict__  expr_71_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_71multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_72_in_0;
    f16 *__restrict__  expr_72_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_72multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_73_in_0;
    f16 *__restrict__  expr_73_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_73multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_88_in_0;
    f16 *__restrict__  expr_88_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_88multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_89_in_0;
    f16 *__restrict__  expr_89_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_89multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_90_in_0;
    f16 *__restrict__  expr_90_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_90multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_91_in_0;
    f16 *__restrict__  expr_91_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_91multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_92_in_0;
    f16 *__restrict__  expr_92_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_92multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_93_in_0;
    f16 *__restrict__  expr_93_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_93multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_94_in_0;
    f16 *__restrict__  expr_94_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_94multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_95_in_0;
    f16 *__restrict__  expr_95_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_95multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_96_in_0;
    f16 *__restrict__  expr_96_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_96multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_109_in_0;
    f16 *__restrict__  expr_109_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_109multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_110_in_0;
    f16 *__restrict__  expr_110_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_110multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_111_in_0;
    f16 *__restrict__  expr_111_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_111multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_112_in_0;
    f16 *__restrict__  expr_112_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_112multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_113_in_0;
    f16 *__restrict__  expr_113_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_113multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_114_in_0;
    f16 *__restrict__  expr_114_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_114multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_115_in_0;
    f16 *__restrict__  expr_115_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_115multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_116_in_0;
    f16 *__restrict__  expr_116_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_116multiple_1_args_t;

typedef struct {
    f16 *__restrict__  expr_117_in_0;
    f16 *__restrict__  expr_117_out_0;
    unsigned short int W;
    unsigned short int H;
    unsigned short int Feat;
} expr_117multiple_1_args_t;


void s14_multiple_1_kernel(s14_multiple_1_kernel_args_t *Args);

void s37_multiple_1_kernel(s37_multiple_1_kernel_args_t *Args);

void s60_multiple_1_kernel(s60_multiple_1_kernel_args_t *Args);

void s387_multiple_1_kernel(s387_multiple_1_kernel_args_t *Args);

void s393_multiple_1_kernel(s393_multiple_1_kernel_args_t *Args);

void s399_multiple_1_kernel(s399_multiple_1_kernel_args_t *Args);

void s487_multiple_1_kernel(s487_multiple_1_kernel_args_t *Args);

void s493_multiple_1_kernel(s493_multiple_1_kernel_args_t *Args);

void s481_multiple_1_kernel(s481_multiple_1_kernel_args_t *Args);

void s640_multiple_1_kernel(s640_multiple_1_kernel_args_t *Args);

void s646_multiple_1_kernel(s646_multiple_1_kernel_args_t *Args);

void s652_multiple_1_kernel(s652_multiple_1_kernel_args_t *Args);

void s800_multiple_1_kernel(s800_multiple_1_kernel_args_t *Args);

void s806_multiple_1_kernel(s806_multiple_1_kernel_args_t *Args);

void s794_multiple_1_kernel(s794_multiple_1_kernel_args_t *Args);

void s818_multiple_1_kernel(s818_multiple_1_kernel_args_t *Args);

void s831_multiple_1_kernel(s831_multiple_1_kernel_args_t *Args);

void s845_multiple_1_kernel(s845_multiple_1_kernel_args_t *Args);

void s858_multiple_1_kernel(s858_multiple_1_kernel_args_t *Args);

void s887_multiple_1_kernel(s887_multiple_1_kernel_args_t *Args);

void s900_multiple_1_kernel(s900_multiple_1_kernel_args_t *Args);

void s929_multiple_1_kernel(s929_multiple_1_kernel_args_t *Args);

void s942_multiple_1_kernel(s942_multiple_1_kernel_args_t *Args);

void s965_multiple_1_kernel(s965_multiple_1_kernel_args_t *Args);

void expr_0multiple_1(expr_0multiple_1_args_t *Args);

void expr_2multiple_1(expr_2multiple_1_args_t *Args);

void expr_3multiple_1(expr_3multiple_1_args_t *Args);

void expr_4multiple_1(expr_4multiple_1_args_t *Args);

void expr_5multiple_1(expr_5multiple_1_args_t *Args);

void expr_6multiple_1(expr_6multiple_1_args_t *Args);

void expr_8multiple_1(expr_8multiple_1_args_t *Args);

void expr_9multiple_1(expr_9multiple_1_args_t *Args);

void expr_10multiple_1(expr_10multiple_1_args_t *Args);

void expr_11multiple_1(expr_11multiple_1_args_t *Args);

void expr_13multiple_1(expr_13multiple_1_args_t *Args);

void expr_14multiple_1(expr_14multiple_1_args_t *Args);

void expr_20multiple_1(expr_20multiple_1_args_t *Args);

void expr_21multiple_1(expr_21multiple_1_args_t *Args);

void expr_23multiple_1(expr_23multiple_1_args_t *Args);

void expr_24multiple_1(expr_24multiple_1_args_t *Args);

void expr_25multiple_1(expr_25multiple_1_args_t *Args);

void expr_26multiple_1(expr_26multiple_1_args_t *Args);

void expr_32multiple_1(expr_32multiple_1_args_t *Args);

void expr_33multiple_1(expr_33multiple_1_args_t *Args);

void expr_35multiple_1(expr_35multiple_1_args_t *Args);

void expr_36multiple_1(expr_36multiple_1_args_t *Args);

void expr_37multiple_1(expr_37multiple_1_args_t *Args);

void expr_39multiple_1(expr_39multiple_1_args_t *Args);

void expr_40multiple_1(expr_40multiple_1_args_t *Args);

void expr_41multiple_1(expr_41multiple_1_args_t *Args);

void expr_42multiple_1(expr_42multiple_1_args_t *Args);

void expr_43multiple_1(expr_43multiple_1_args_t *Args);

void expr_44multiple_1(expr_44multiple_1_args_t *Args);

void expr_45multiple_1(expr_45multiple_1_args_t *Args);

void expr_46multiple_1(expr_46multiple_1_args_t *Args);

void expr_47multiple_1(expr_47multiple_1_args_t *Args);

void expr_49multiple_1(expr_49multiple_1_args_t *Args);

void expr_50multiple_1(expr_50multiple_1_args_t *Args);

void expr_52multiple_1(expr_52multiple_1_args_t *Args);

void expr_53multiple_1(expr_53multiple_1_args_t *Args);

void expr_54multiple_1(expr_54multiple_1_args_t *Args);

void expr_55multiple_1(expr_55multiple_1_args_t *Args);

void expr_56multiple_1(expr_56multiple_1_args_t *Args);

void expr_57multiple_1(expr_57multiple_1_args_t *Args);

void expr_58multiple_1(expr_58multiple_1_args_t *Args);

void expr_59multiple_1(expr_59multiple_1_args_t *Args);

void expr_60multiple_1(expr_60multiple_1_args_t *Args);

void expr_62multiple_1(expr_62multiple_1_args_t *Args);

void expr_63multiple_1(expr_63multiple_1_args_t *Args);

void expr_64multiple_1(expr_64multiple_1_args_t *Args);

void expr_75multiple_1(expr_75multiple_1_args_t *Args);

void expr_76multiple_1(expr_76multiple_1_args_t *Args);

void expr_77multiple_1(expr_77multiple_1_args_t *Args);

void expr_78multiple_1(expr_78multiple_1_args_t *Args);

void expr_79multiple_1(expr_79multiple_1_args_t *Args);

void expr_80multiple_1(expr_80multiple_1_args_t *Args);

void expr_81multiple_1(expr_81multiple_1_args_t *Args);

void expr_82multiple_1(expr_82multiple_1_args_t *Args);

void expr_83multiple_1(expr_83multiple_1_args_t *Args);

void expr_85multiple_1(expr_85multiple_1_args_t *Args);

void expr_86multiple_1(expr_86multiple_1_args_t *Args);

void expr_87multiple_1(expr_87multiple_1_args_t *Args);

void expr_98multiple_1(expr_98multiple_1_args_t *Args);

void expr_99multiple_1(expr_99multiple_1_args_t *Args);

void expr_100multiple_1(expr_100multiple_1_args_t *Args);

void expr_101multiple_1(expr_101multiple_1_args_t *Args);

void expr_102multiple_1(expr_102multiple_1_args_t *Args);

void expr_103multiple_1(expr_103multiple_1_args_t *Args);

void expr_104multiple_1(expr_104multiple_1_args_t *Args);

void expr_105multiple_1(expr_105multiple_1_args_t *Args);

void expr_106multiple_1(expr_106multiple_1_args_t *Args);

void expr_108multiple_1(expr_108multiple_1_args_t *Args);

void expr_65multiple_1(expr_65multiple_1_args_t *Args);

void expr_66multiple_1(expr_66multiple_1_args_t *Args);

void expr_67multiple_1(expr_67multiple_1_args_t *Args);

void expr_68multiple_1(expr_68multiple_1_args_t *Args);

void expr_69multiple_1(expr_69multiple_1_args_t *Args);

void expr_70multiple_1(expr_70multiple_1_args_t *Args);

void expr_71multiple_1(expr_71multiple_1_args_t *Args);

void expr_72multiple_1(expr_72multiple_1_args_t *Args);

void expr_73multiple_1(expr_73multiple_1_args_t *Args);

void expr_88multiple_1(expr_88multiple_1_args_t *Args);

void expr_89multiple_1(expr_89multiple_1_args_t *Args);

void expr_90multiple_1(expr_90multiple_1_args_t *Args);

void expr_91multiple_1(expr_91multiple_1_args_t *Args);

void expr_92multiple_1(expr_92multiple_1_args_t *Args);

void expr_93multiple_1(expr_93multiple_1_args_t *Args);

void expr_94multiple_1(expr_94multiple_1_args_t *Args);

void expr_95multiple_1(expr_95multiple_1_args_t *Args);

void expr_96multiple_1(expr_96multiple_1_args_t *Args);

void expr_109multiple_1(expr_109multiple_1_args_t *Args);

void expr_110multiple_1(expr_110multiple_1_args_t *Args);

void expr_111multiple_1(expr_111multiple_1_args_t *Args);

void expr_112multiple_1(expr_112multiple_1_args_t *Args);

void expr_113multiple_1(expr_113multiple_1_args_t *Args);

void expr_114multiple_1(expr_114multiple_1_args_t *Args);

void expr_115multiple_1(expr_115multiple_1_args_t *Args);

void expr_116multiple_1(expr_116multiple_1_args_t *Args);

void expr_117multiple_1(expr_117multiple_1_args_t *Args);


#endif // ONNX_GRAPH_BASIC_KERNELS_H