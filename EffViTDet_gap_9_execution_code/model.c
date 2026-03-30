#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_fp16.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"

void load_expressions_kernels() {
    LibKernelTemplate("SplitWidthIn12Arg_T",
        CArgs(27,
        TCArg("void * __restrict__", "In"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "Out1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "Out2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "Out3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "Out4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "Out5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "Out6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "Out7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "Out8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "Out9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "Out10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "Out11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "Out12"),
        TCArg("int", "W12"),
        TCArg("int", "DataSize")
        )
    );

    LibKernel("CNN_Split_Width_In12", CALL_PARALLEL,
        CArgs(27,
        TCArg("void * __restrict__", "In"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "Out1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "Out2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "Out3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "Out4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "Out5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "Out6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "Out7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "Out8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "Out9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "Out10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "Out11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "Out12"),
        TCArg("int", "W12"),
        TCArg("int", "DataSize")
        ),
        "SplitWidthIn12Arg_T", NULL
    );

    LibKernelTemplate("ConcatWidthIn13Arg_T",
        CArgs(29,
        TCArg("void * __restrict__", "Out"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "In1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "In2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "In3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "In4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "In5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "In6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "In7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "In8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "In9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "In10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "In11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "In12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "In13"),
        TCArg("int", "W13"),
        TCArg("int", "DataSize")
        )
    );

    LibKernel("CNN_Concat_Width_In13", CALL_PARALLEL,
        CArgs(29,
        TCArg("void * __restrict__", "Out"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "In1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "In2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "In3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "In4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "In5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "In6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "In7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "In8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "In9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "In10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "In11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "In12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "In13"),
        TCArg("int", "W13"),
        TCArg("int", "DataSize")
        ),
        "ConcatWidthIn13Arg_T", NULL
    );

    LibKernelTemplate("SplitWidthIn24Arg_T",
        CArgs(51,
        TCArg("void * __restrict__", "In"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "Out1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "Out2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "Out3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "Out4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "Out5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "Out6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "Out7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "Out8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "Out9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "Out10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "Out11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "Out12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "Out13"),
        TCArg("int", "W13"),
        TCArg("void * __restrict__", "Out14"),
        TCArg("int", "W14"),
        TCArg("void * __restrict__", "Out15"),
        TCArg("int", "W15"),
        TCArg("void * __restrict__", "Out16"),
        TCArg("int", "W16"),
        TCArg("void * __restrict__", "Out17"),
        TCArg("int", "W17"),
        TCArg("void * __restrict__", "Out18"),
        TCArg("int", "W18"),
        TCArg("void * __restrict__", "Out19"),
        TCArg("int", "W19"),
        TCArg("void * __restrict__", "Out20"),
        TCArg("int", "W20"),
        TCArg("void * __restrict__", "Out21"),
        TCArg("int", "W21"),
        TCArg("void * __restrict__", "Out22"),
        TCArg("int", "W22"),
        TCArg("void * __restrict__", "Out23"),
        TCArg("int", "W23"),
        TCArg("void * __restrict__", "Out24"),
        TCArg("int", "W24"),
        TCArg("int", "DataSize")
        )
    );

    LibKernel("CNN_Split_Width_In24", CALL_PARALLEL,
        CArgs(51,
        TCArg("void * __restrict__", "In"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "Out1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "Out2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "Out3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "Out4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "Out5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "Out6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "Out7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "Out8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "Out9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "Out10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "Out11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "Out12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "Out13"),
        TCArg("int", "W13"),
        TCArg("void * __restrict__", "Out14"),
        TCArg("int", "W14"),
        TCArg("void * __restrict__", "Out15"),
        TCArg("int", "W15"),
        TCArg("void * __restrict__", "Out16"),
        TCArg("int", "W16"),
        TCArg("void * __restrict__", "Out17"),
        TCArg("int", "W17"),
        TCArg("void * __restrict__", "Out18"),
        TCArg("int", "W18"),
        TCArg("void * __restrict__", "Out19"),
        TCArg("int", "W19"),
        TCArg("void * __restrict__", "Out20"),
        TCArg("int", "W20"),
        TCArg("void * __restrict__", "Out21"),
        TCArg("int", "W21"),
        TCArg("void * __restrict__", "Out22"),
        TCArg("int", "W22"),
        TCArg("void * __restrict__", "Out23"),
        TCArg("int", "W23"),
        TCArg("void * __restrict__", "Out24"),
        TCArg("int", "W24"),
        TCArg("int", "DataSize")
        ),
        "SplitWidthIn24Arg_T", NULL
    );

    LibKernelTemplate("ConcatWidthIn25Arg_T",
        CArgs(53,
        TCArg("void * __restrict__", "Out"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "In1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "In2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "In3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "In4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "In5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "In6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "In7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "In8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "In9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "In10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "In11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "In12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "In13"),
        TCArg("int", "W13"),
        TCArg("void * __restrict__", "In14"),
        TCArg("int", "W14"),
        TCArg("void * __restrict__", "In15"),
        TCArg("int", "W15"),
        TCArg("void * __restrict__", "In16"),
        TCArg("int", "W16"),
        TCArg("void * __restrict__", "In17"),
        TCArg("int", "W17"),
        TCArg("void * __restrict__", "In18"),
        TCArg("int", "W18"),
        TCArg("void * __restrict__", "In19"),
        TCArg("int", "W19"),
        TCArg("void * __restrict__", "In20"),
        TCArg("int", "W20"),
        TCArg("void * __restrict__", "In21"),
        TCArg("int", "W21"),
        TCArg("void * __restrict__", "In22"),
        TCArg("int", "W22"),
        TCArg("void * __restrict__", "In23"),
        TCArg("int", "W23"),
        TCArg("void * __restrict__", "In24"),
        TCArg("int", "W24"),
        TCArg("void * __restrict__", "In25"),
        TCArg("int", "W25"),
        TCArg("int", "DataSize")
        )
    );

    LibKernel("CNN_Concat_Width_In25", CALL_PARALLEL,
        CArgs(53,
        TCArg("void * __restrict__", "Out"),
        TCArg("int", "H"),
        TCArg("void * __restrict__", "In1"),
        TCArg("int", "W1"),
        TCArg("void * __restrict__", "In2"),
        TCArg("int", "W2"),
        TCArg("void * __restrict__", "In3"),
        TCArg("int", "W3"),
        TCArg("void * __restrict__", "In4"),
        TCArg("int", "W4"),
        TCArg("void * __restrict__", "In5"),
        TCArg("int", "W5"),
        TCArg("void * __restrict__", "In6"),
        TCArg("int", "W6"),
        TCArg("void * __restrict__", "In7"),
        TCArg("int", "W7"),
        TCArg("void * __restrict__", "In8"),
        TCArg("int", "W8"),
        TCArg("void * __restrict__", "In9"),
        TCArg("int", "W9"),
        TCArg("void * __restrict__", "In10"),
        TCArg("int", "W10"),
        TCArg("void * __restrict__", "In11"),
        TCArg("int", "W11"),
        TCArg("void * __restrict__", "In12"),
        TCArg("int", "W12"),
        TCArg("void * __restrict__", "In13"),
        TCArg("int", "W13"),
        TCArg("void * __restrict__", "In14"),
        TCArg("int", "W14"),
        TCArg("void * __restrict__", "In15"),
        TCArg("int", "W15"),
        TCArg("void * __restrict__", "In16"),
        TCArg("int", "W16"),
        TCArg("void * __restrict__", "In17"),
        TCArg("int", "W17"),
        TCArg("void * __restrict__", "In18"),
        TCArg("int", "W18"),
        TCArg("void * __restrict__", "In19"),
        TCArg("int", "W19"),
        TCArg("void * __restrict__", "In20"),
        TCArg("int", "W20"),
        TCArg("void * __restrict__", "In21"),
        TCArg("int", "W21"),
        TCArg("void * __restrict__", "In22"),
        TCArg("int", "W22"),
        TCArg("void * __restrict__", "In23"),
        TCArg("int", "W23"),
        TCArg("void * __restrict__", "In24"),
        TCArg("int", "W24"),
        TCArg("void * __restrict__", "In25"),
        TCArg("int", "W25"),
        TCArg("int", "DataSize")
        ),
        "ConcatWidthIn25Arg_T", NULL
    );

    LibKernelTemplate(
        "s10_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_1_in_0"),
            TCArg("f16 *__restrict__ ", "expr_1_in_1"),
            TCArg("f16 *__restrict__ ", "expr_1_out_0")
        )
    );
    
    LibKernel(
        "s10_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s10_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s29_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_7_in_0"),
            TCArg("f16 *__restrict__ ", "expr_7_in_1"),
            TCArg("f16 *__restrict__ ", "expr_7_out_0")
        )
    );
    
    LibKernel(
        "s29_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s29_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s48_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_12_in_0"),
            TCArg("f16 *__restrict__ ", "expr_12_in_1"),
            TCArg("f16 *__restrict__ ", "expr_12_out_0")
        )
    );
    
    LibKernel(
        "s48_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s48_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s374_multiple_1_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("f16 *__restrict__ ", "expr_15_in_0"),
            TCArg("f16 *__restrict__ ", "expr_15_in_1"),
            TCArg("f16 *__restrict__ ", "expr_15_out_0")
        )
    );
    
    LibKernel(
        "s374_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s374_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s380_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_16_in_0"),
            TCArg("f16 *__restrict__ ", "expr_16_in_1"),
            TCArg("f16 *__restrict__ ", "expr_16_out_0")
        )
    );
    
    LibKernel(
        "s380_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s380_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s386_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_17_in_0"),
            TCArg("f16 *__restrict__ ", "expr_17_in_1"),
            TCArg("f16 *__restrict__ ", "expr_17_out_0")
        )
    );
    
    LibKernel(
        "s386_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s386_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s474_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_18_in_0"),
            TCArg("f16 *__restrict__ ", "expr_18_in_1"),
            TCArg("f16 *__restrict__ ", "expr_18_out_0")
        )
    );
    
    LibKernel(
        "s474_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s474_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s480_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_19_in_0"),
            TCArg("f16 *__restrict__ ", "expr_19_in_1"),
            TCArg("f16 *__restrict__ ", "expr_19_out_0")
        )
    );
    
    LibKernel(
        "s480_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s480_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s468_multiple_1_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("f16 *__restrict__ ", "expr_22_in_0"),
            TCArg("f16 *__restrict__ ", "expr_22_in_1"),
            TCArg("f16 *__restrict__ ", "expr_22_out_0")
        )
    );
    
    LibKernel(
        "s468_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s468_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s627_multiple_1_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("f16 *__restrict__ ", "expr_27_in_0"),
            TCArg("f16 *__restrict__ ", "expr_27_in_1"),
            TCArg("f16 *__restrict__ ", "expr_27_out_0")
        )
    );
    
    LibKernel(
        "s627_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s627_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s633_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_28_in_0"),
            TCArg("f16 *__restrict__ ", "expr_28_in_1"),
            TCArg("f16 *__restrict__ ", "expr_28_out_0")
        )
    );
    
    LibKernel(
        "s633_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s633_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s639_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_29_in_0"),
            TCArg("f16 *__restrict__ ", "expr_29_in_1"),
            TCArg("f16 *__restrict__ ", "expr_29_out_0")
        )
    );
    
    LibKernel(
        "s639_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s639_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s787_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_30_in_0"),
            TCArg("f16 *__restrict__ ", "expr_30_in_1"),
            TCArg("f16 *__restrict__ ", "expr_30_out_0")
        )
    );
    
    LibKernel(
        "s787_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s787_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s793_multiple_1_kernel_args_t",
        CArgs(4,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_31_in_0"),
            TCArg("f16 *__restrict__ ", "expr_31_in_1"),
            TCArg("f16 *__restrict__ ", "expr_31_out_0")
        )
    );
    
    LibKernel(
        "s793_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s793_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s781_multiple_1_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("f16 *__restrict__ ", "expr_34_in_0"),
            TCArg("f16 *__restrict__ ", "expr_34_in_1"),
            TCArg("f16 *__restrict__ ", "expr_34_out_0")
        )
    );
    
    LibKernel(
        "s781_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s781_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s803_multiple_1_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_38_in_0"),
            TCArg("f16 *__restrict__ ", "expr_38_out_0")
        )
    );
    
    LibKernel(
        "s803_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s803_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s816_multiple_1_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_48_in_0"),
            TCArg("f16 *__restrict__ ", "expr_48_out_0")
        )
    );
    
    LibKernel(
        "s816_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s816_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s843_multiple_1_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_67_in_0"),
            TCArg("f16 *__restrict__ ", "expr_67_out_0")
        )
    );
    
    LibKernel(
        "s843_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s843_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s874_multiple_1_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_90_in_0"),
            TCArg("f16 *__restrict__ ", "expr_90_out_0")
        )
    );
    
    LibKernel(
        "s874_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s874_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s903_multiple_1_kernel_args_t",
        CArgs(3,
            TCArg("unsigned int", "I0"),
            TCArg("f16 *__restrict__ ", "expr_113_in_0"),
            TCArg("f16 *__restrict__ ", "expr_113_out_0")
        )
    );
    
    LibKernel(
        "s903_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s903_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "s915_multiple_1_kernel_args_t",
        CArgs(5,
            TCArg("unsigned int", "I0"),
            TCArg("unsigned int", "I1"),
            TCArg("f16 *__restrict__ ", "expr_118_in_0"),
            TCArg("f16 *__restrict__ ", "expr_118_in_1"),
            TCArg("f16 *__restrict__ ", "expr_118_out_0")
        )
    );
    
    LibKernel(
        "s915_multiple_1_kernel",
        CALL_PARALLEL,
        0,
        "s915_multiple_1_kernel_args_t",
        0
    );
    LibKernelTemplate(
        "expr_0_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_0_in_0"),
            TCArg("f16 *__restrict__ ", "expr_0_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_0",
        CALL_PARALLEL,
        0,
        "expr_0_args_t",
        0
    );
    LibKernelTemplate(
        "expr_2_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_2_in_0"),
            TCArg("f16 *__restrict__ ", "expr_2_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_2",
        CALL_PARALLEL,
        0,
        "expr_2_args_t",
        0
    );
    LibKernelTemplate(
        "expr_3_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_3_in_0"),
            TCArg("f16 *__restrict__ ", "expr_3_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_3",
        CALL_PARALLEL,
        0,
        "expr_3_args_t",
        0
    );
    LibKernelTemplate(
        "expr_4_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_4_in_0"),
            TCArg("f16 *__restrict__ ", "expr_4_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_4",
        CALL_PARALLEL,
        0,
        "expr_4_args_t",
        0
    );
    LibKernelTemplate(
        "expr_5_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_5_in_0"),
            TCArg("f16 *__restrict__ ", "expr_5_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_5",
        CALL_PARALLEL,
        0,
        "expr_5_args_t",
        0
    );
    LibKernelTemplate(
        "expr_6_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_6_in_0"),
            TCArg("f16 *__restrict__ ", "expr_6_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_6",
        CALL_PARALLEL,
        0,
        "expr_6_args_t",
        0
    );
    LibKernelTemplate(
        "expr_8_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_8_in_0"),
            TCArg("f16 *__restrict__ ", "expr_8_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_8",
        CALL_PARALLEL,
        0,
        "expr_8_args_t",
        0
    );
    LibKernelTemplate(
        "expr_9_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_9_in_0"),
            TCArg("f16 *__restrict__ ", "expr_9_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_9",
        CALL_PARALLEL,
        0,
        "expr_9_args_t",
        0
    );
    LibKernelTemplate(
        "expr_10_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_10_in_0"),
            TCArg("f16 *__restrict__ ", "expr_10_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_10",
        CALL_PARALLEL,
        0,
        "expr_10_args_t",
        0
    );
    LibKernelTemplate(
        "expr_11_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_11_in_0"),
            TCArg("f16 *__restrict__ ", "expr_11_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_11",
        CALL_PARALLEL,
        0,
        "expr_11_args_t",
        0
    );
    LibKernelTemplate(
        "expr_13_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_13_in_0"),
            TCArg("f16 *__restrict__ ", "expr_13_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_13",
        CALL_PARALLEL,
        0,
        "expr_13_args_t",
        0
    );
    LibKernelTemplate(
        "expr_14_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_14_in_0"),
            TCArg("f16 *__restrict__ ", "expr_14_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_14",
        CALL_PARALLEL,
        0,
        "expr_14_args_t",
        0
    );
    LibKernelTemplate(
        "expr_20_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_20_in_0"),
            TCArg("f16 *__restrict__ ", "expr_20_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_20",
        CALL_PARALLEL,
        0,
        "expr_20_args_t",
        0
    );
    LibKernelTemplate(
        "expr_21_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_21_in_0"),
            TCArg("f16 *__restrict__ ", "expr_21_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_21",
        CALL_PARALLEL,
        0,
        "expr_21_args_t",
        0
    );
    LibKernelTemplate(
        "expr_23_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_23_in_0"),
            TCArg("f16 *__restrict__ ", "expr_23_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_23",
        CALL_PARALLEL,
        0,
        "expr_23_args_t",
        0
    );
    LibKernelTemplate(
        "expr_24_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_24_in_0"),
            TCArg("f16 *__restrict__ ", "expr_24_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_24",
        CALL_PARALLEL,
        0,
        "expr_24_args_t",
        0
    );
    LibKernelTemplate(
        "expr_25_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_25_in_0"),
            TCArg("f16 *__restrict__ ", "expr_25_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_25",
        CALL_PARALLEL,
        0,
        "expr_25_args_t",
        0
    );
    LibKernelTemplate(
        "expr_26_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_26_in_0"),
            TCArg("f16 *__restrict__ ", "expr_26_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_26",
        CALL_PARALLEL,
        0,
        "expr_26_args_t",
        0
    );
    LibKernelTemplate(
        "expr_32_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_32_in_0"),
            TCArg("f16 *__restrict__ ", "expr_32_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_32",
        CALL_PARALLEL,
        0,
        "expr_32_args_t",
        0
    );
    LibKernelTemplate(
        "expr_33_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_33_in_0"),
            TCArg("f16 *__restrict__ ", "expr_33_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_33",
        CALL_PARALLEL,
        0,
        "expr_33_args_t",
        0
    );
    LibKernelTemplate(
        "expr_35_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_35_in_0"),
            TCArg("f16 *__restrict__ ", "expr_35_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_35",
        CALL_PARALLEL,
        0,
        "expr_35_args_t",
        0
    );
    LibKernelTemplate(
        "expr_36_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_36_in_0"),
            TCArg("f16 *__restrict__ ", "expr_36_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_36",
        CALL_PARALLEL,
        0,
        "expr_36_args_t",
        0
    );
    LibKernelTemplate(
        "expr_37_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_37_in_0"),
            TCArg("f16 *__restrict__ ", "expr_37_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_37",
        CALL_PARALLEL,
        0,
        "expr_37_args_t",
        0
    );
    LibKernelTemplate(
        "expr_39_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_39_in_0"),
            TCArg("f16 *__restrict__ ", "expr_39_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_39",
        CALL_PARALLEL,
        0,
        "expr_39_args_t",
        0
    );
    LibKernelTemplate(
        "expr_40_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_40_in_0"),
            TCArg("f16 *__restrict__ ", "expr_40_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_40",
        CALL_PARALLEL,
        0,
        "expr_40_args_t",
        0
    );
    LibKernelTemplate(
        "expr_41_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_41_in_0"),
            TCArg("f16 *__restrict__ ", "expr_41_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_41",
        CALL_PARALLEL,
        0,
        "expr_41_args_t",
        0
    );
    LibKernelTemplate(
        "expr_42_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_42_in_0"),
            TCArg("f16 *__restrict__ ", "expr_42_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_42",
        CALL_PARALLEL,
        0,
        "expr_42_args_t",
        0
    );
    LibKernelTemplate(
        "expr_43_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_43_in_0"),
            TCArg("f16 *__restrict__ ", "expr_43_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_43",
        CALL_PARALLEL,
        0,
        "expr_43_args_t",
        0
    );
    LibKernelTemplate(
        "expr_44_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_44_in_0"),
            TCArg("f16 *__restrict__ ", "expr_44_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_44",
        CALL_PARALLEL,
        0,
        "expr_44_args_t",
        0
    );
    LibKernelTemplate(
        "expr_45_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_45_in_0"),
            TCArg("f16 *__restrict__ ", "expr_45_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_45",
        CALL_PARALLEL,
        0,
        "expr_45_args_t",
        0
    );
    LibKernelTemplate(
        "expr_46_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_46_in_0"),
            TCArg("f16 *__restrict__ ", "expr_46_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_46",
        CALL_PARALLEL,
        0,
        "expr_46_args_t",
        0
    );
    LibKernelTemplate(
        "expr_47_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_47_in_0"),
            TCArg("f16 *__restrict__ ", "expr_47_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_47",
        CALL_PARALLEL,
        0,
        "expr_47_args_t",
        0
    );
    LibKernelTemplate(
        "expr_49_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_49_in_0"),
            TCArg("f16 *__restrict__ ", "expr_49_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_49",
        CALL_PARALLEL,
        0,
        "expr_49_args_t",
        0
    );
    LibKernelTemplate(
        "expr_50_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_50_in_0"),
            TCArg("f16 *__restrict__ ", "expr_50_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_50",
        CALL_PARALLEL,
        0,
        "expr_50_args_t",
        0
    );
    LibKernelTemplate(
        "expr_51_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_51_in_0"),
            TCArg("f16 *__restrict__ ", "expr_51_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_51",
        CALL_PARALLEL,
        0,
        "expr_51_args_t",
        0
    );
    LibKernelTemplate(
        "expr_52_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_52_in_0"),
            TCArg("f16 *__restrict__ ", "expr_52_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_52",
        CALL_PARALLEL,
        0,
        "expr_52_args_t",
        0
    );
    LibKernelTemplate(
        "expr_53_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_53_in_0"),
            TCArg("f16 *__restrict__ ", "expr_53_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_53",
        CALL_PARALLEL,
        0,
        "expr_53_args_t",
        0
    );
    LibKernelTemplate(
        "expr_54_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_54_in_0"),
            TCArg("f16 *__restrict__ ", "expr_54_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_54",
        CALL_PARALLEL,
        0,
        "expr_54_args_t",
        0
    );
    LibKernelTemplate(
        "expr_55_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_55_in_0"),
            TCArg("f16 *__restrict__ ", "expr_55_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_55",
        CALL_PARALLEL,
        0,
        "expr_55_args_t",
        0
    );
    LibKernelTemplate(
        "expr_56_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_56_in_0"),
            TCArg("f16 *__restrict__ ", "expr_56_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_56",
        CALL_PARALLEL,
        0,
        "expr_56_args_t",
        0
    );
    LibKernelTemplate(
        "expr_57_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_57_in_0"),
            TCArg("f16 *__restrict__ ", "expr_57_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_57",
        CALL_PARALLEL,
        0,
        "expr_57_args_t",
        0
    );
    LibKernelTemplate(
        "expr_58_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_58_in_0"),
            TCArg("f16 *__restrict__ ", "expr_58_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_58",
        CALL_PARALLEL,
        0,
        "expr_58_args_t",
        0
    );
    LibKernelTemplate(
        "expr_59_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_59_in_0"),
            TCArg("f16 *__restrict__ ", "expr_59_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_59",
        CALL_PARALLEL,
        0,
        "expr_59_args_t",
        0
    );
    LibKernelTemplate(
        "expr_60_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_60_in_0"),
            TCArg("f16 *__restrict__ ", "expr_60_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_60",
        CALL_PARALLEL,
        0,
        "expr_60_args_t",
        0
    );
    LibKernelTemplate(
        "expr_62_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_62_in_0"),
            TCArg("f16 *__restrict__ ", "expr_62_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_62",
        CALL_PARALLEL,
        0,
        "expr_62_args_t",
        0
    );
    LibKernelTemplate(
        "expr_72_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_72_in_0"),
            TCArg("f16 *__restrict__ ", "expr_72_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_72",
        CALL_PARALLEL,
        0,
        "expr_72_args_t",
        0
    );
    LibKernelTemplate(
        "expr_73_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_73_in_0"),
            TCArg("f16 *__restrict__ ", "expr_73_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_73",
        CALL_PARALLEL,
        0,
        "expr_73_args_t",
        0
    );
    LibKernelTemplate(
        "expr_74_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_74_in_0"),
            TCArg("f16 *__restrict__ ", "expr_74_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_74",
        CALL_PARALLEL,
        0,
        "expr_74_args_t",
        0
    );
    LibKernelTemplate(
        "expr_75_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_75_in_0"),
            TCArg("f16 *__restrict__ ", "expr_75_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_75",
        CALL_PARALLEL,
        0,
        "expr_75_args_t",
        0
    );
    LibKernelTemplate(
        "expr_76_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_76_in_0"),
            TCArg("f16 *__restrict__ ", "expr_76_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_76",
        CALL_PARALLEL,
        0,
        "expr_76_args_t",
        0
    );
    LibKernelTemplate(
        "expr_77_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_77_in_0"),
            TCArg("f16 *__restrict__ ", "expr_77_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_77",
        CALL_PARALLEL,
        0,
        "expr_77_args_t",
        0
    );
    LibKernelTemplate(
        "expr_78_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_78_in_0"),
            TCArg("f16 *__restrict__ ", "expr_78_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_78",
        CALL_PARALLEL,
        0,
        "expr_78_args_t",
        0
    );
    LibKernelTemplate(
        "expr_79_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_79_in_0"),
            TCArg("f16 *__restrict__ ", "expr_79_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_79",
        CALL_PARALLEL,
        0,
        "expr_79_args_t",
        0
    );
    LibKernelTemplate(
        "expr_80_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_80_in_0"),
            TCArg("f16 *__restrict__ ", "expr_80_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_80",
        CALL_PARALLEL,
        0,
        "expr_80_args_t",
        0
    );
    LibKernelTemplate(
        "expr_81_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_81_in_0"),
            TCArg("f16 *__restrict__ ", "expr_81_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_81",
        CALL_PARALLEL,
        0,
        "expr_81_args_t",
        0
    );
    LibKernelTemplate(
        "expr_82_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_82_in_0"),
            TCArg("f16 *__restrict__ ", "expr_82_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_82",
        CALL_PARALLEL,
        0,
        "expr_82_args_t",
        0
    );
    LibKernelTemplate(
        "expr_83_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_83_in_0"),
            TCArg("f16 *__restrict__ ", "expr_83_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_83",
        CALL_PARALLEL,
        0,
        "expr_83_args_t",
        0
    );
    LibKernelTemplate(
        "expr_85_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_85_in_0"),
            TCArg("f16 *__restrict__ ", "expr_85_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_85",
        CALL_PARALLEL,
        0,
        "expr_85_args_t",
        0
    );
    LibKernelTemplate(
        "expr_95_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_95_in_0"),
            TCArg("f16 *__restrict__ ", "expr_95_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_95",
        CALL_PARALLEL,
        0,
        "expr_95_args_t",
        0
    );
    LibKernelTemplate(
        "expr_96_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_96_in_0"),
            TCArg("f16 *__restrict__ ", "expr_96_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_96",
        CALL_PARALLEL,
        0,
        "expr_96_args_t",
        0
    );
    LibKernelTemplate(
        "expr_107_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_107_in_0"),
            TCArg("f16 *__restrict__ ", "expr_107_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_107",
        CALL_PARALLEL,
        0,
        "expr_107_args_t",
        0
    );
    LibKernelTemplate(
        "expr_98_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_98_in_0"),
            TCArg("f16 *__restrict__ ", "expr_98_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_98",
        CALL_PARALLEL,
        0,
        "expr_98_args_t",
        0
    );
    LibKernelTemplate(
        "expr_99_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_99_in_0"),
            TCArg("f16 *__restrict__ ", "expr_99_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_99",
        CALL_PARALLEL,
        0,
        "expr_99_args_t",
        0
    );
    LibKernelTemplate(
        "expr_100_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_100_in_0"),
            TCArg("f16 *__restrict__ ", "expr_100_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_100",
        CALL_PARALLEL,
        0,
        "expr_100_args_t",
        0
    );
    LibKernelTemplate(
        "expr_101_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_101_in_0"),
            TCArg("f16 *__restrict__ ", "expr_101_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_101",
        CALL_PARALLEL,
        0,
        "expr_101_args_t",
        0
    );
    LibKernelTemplate(
        "expr_102_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_102_in_0"),
            TCArg("f16 *__restrict__ ", "expr_102_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_102",
        CALL_PARALLEL,
        0,
        "expr_102_args_t",
        0
    );
    LibKernelTemplate(
        "expr_103_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_103_in_0"),
            TCArg("f16 *__restrict__ ", "expr_103_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_103",
        CALL_PARALLEL,
        0,
        "expr_103_args_t",
        0
    );
    LibKernelTemplate(
        "expr_104_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_104_in_0"),
            TCArg("f16 *__restrict__ ", "expr_104_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_104",
        CALL_PARALLEL,
        0,
        "expr_104_args_t",
        0
    );
    LibKernelTemplate(
        "expr_105_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_105_in_0"),
            TCArg("f16 *__restrict__ ", "expr_105_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_105",
        CALL_PARALLEL,
        0,
        "expr_105_args_t",
        0
    );
    LibKernelTemplate(
        "expr_106_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_106_in_0"),
            TCArg("f16 *__restrict__ ", "expr_106_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_106",
        CALL_PARALLEL,
        0,
        "expr_106_args_t",
        0
    );
    LibKernelTemplate(
        "expr_108_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_108_in_0"),
            TCArg("f16 *__restrict__ ", "expr_108_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_108",
        CALL_PARALLEL,
        0,
        "expr_108_args_t",
        0
    );
    LibKernelTemplate(
        "expr_63_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_63_in_0"),
            TCArg("f16 *__restrict__ ", "expr_63_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_63",
        CALL_PARALLEL,
        0,
        "expr_63_args_t",
        0
    );
    LibKernelTemplate(
        "expr_64_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_64_in_0"),
            TCArg("f16 *__restrict__ ", "expr_64_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_64",
        CALL_PARALLEL,
        0,
        "expr_64_args_t",
        0
    );
    LibKernelTemplate(
        "expr_65_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_65_in_0"),
            TCArg("f16 *__restrict__ ", "expr_65_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_65",
        CALL_PARALLEL,
        0,
        "expr_65_args_t",
        0
    );
    LibKernelTemplate(
        "expr_66_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_66_in_0"),
            TCArg("f16 *__restrict__ ", "expr_66_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_66",
        CALL_PARALLEL,
        0,
        "expr_66_args_t",
        0
    );
    LibKernelTemplate(
        "expr_68_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_68_in_0"),
            TCArg("f16 *__restrict__ ", "expr_68_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_68",
        CALL_PARALLEL,
        0,
        "expr_68_args_t",
        0
    );
    LibKernelTemplate(
        "expr_69_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_69_in_0"),
            TCArg("f16 *__restrict__ ", "expr_69_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_69",
        CALL_PARALLEL,
        0,
        "expr_69_args_t",
        0
    );
    LibKernelTemplate(
        "expr_70_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_70_in_0"),
            TCArg("f16 *__restrict__ ", "expr_70_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_70",
        CALL_PARALLEL,
        0,
        "expr_70_args_t",
        0
    );
    LibKernelTemplate(
        "expr_71_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_71_in_0"),
            TCArg("f16 *__restrict__ ", "expr_71_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_71",
        CALL_PARALLEL,
        0,
        "expr_71_args_t",
        0
    );
    LibKernelTemplate(
        "expr_86_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_86_in_0"),
            TCArg("f16 *__restrict__ ", "expr_86_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_86",
        CALL_PARALLEL,
        0,
        "expr_86_args_t",
        0
    );
    LibKernelTemplate(
        "expr_87_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_87_in_0"),
            TCArg("f16 *__restrict__ ", "expr_87_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_87",
        CALL_PARALLEL,
        0,
        "expr_87_args_t",
        0
    );
    LibKernelTemplate(
        "expr_88_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_88_in_0"),
            TCArg("f16 *__restrict__ ", "expr_88_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_88",
        CALL_PARALLEL,
        0,
        "expr_88_args_t",
        0
    );
    LibKernelTemplate(
        "expr_89_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_89_in_0"),
            TCArg("f16 *__restrict__ ", "expr_89_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_89",
        CALL_PARALLEL,
        0,
        "expr_89_args_t",
        0
    );
    LibKernelTemplate(
        "expr_91_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_91_in_0"),
            TCArg("f16 *__restrict__ ", "expr_91_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_91",
        CALL_PARALLEL,
        0,
        "expr_91_args_t",
        0
    );
    LibKernelTemplate(
        "expr_92_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_92_in_0"),
            TCArg("f16 *__restrict__ ", "expr_92_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_92",
        CALL_PARALLEL,
        0,
        "expr_92_args_t",
        0
    );
    LibKernelTemplate(
        "expr_93_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_93_in_0"),
            TCArg("f16 *__restrict__ ", "expr_93_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_93",
        CALL_PARALLEL,
        0,
        "expr_93_args_t",
        0
    );
    LibKernelTemplate(
        "expr_94_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_94_in_0"),
            TCArg("f16 *__restrict__ ", "expr_94_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_94",
        CALL_PARALLEL,
        0,
        "expr_94_args_t",
        0
    );
    LibKernelTemplate(
        "expr_109_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_109_in_0"),
            TCArg("f16 *__restrict__ ", "expr_109_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_109",
        CALL_PARALLEL,
        0,
        "expr_109_args_t",
        0
    );
    LibKernelTemplate(
        "expr_110_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_110_in_0"),
            TCArg("f16 *__restrict__ ", "expr_110_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_110",
        CALL_PARALLEL,
        0,
        "expr_110_args_t",
        0
    );
    LibKernelTemplate(
        "expr_111_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_111_in_0"),
            TCArg("f16 *__restrict__ ", "expr_111_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_111",
        CALL_PARALLEL,
        0,
        "expr_111_args_t",
        0
    );
    LibKernelTemplate(
        "expr_112_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_112_in_0"),
            TCArg("f16 *__restrict__ ", "expr_112_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_112",
        CALL_PARALLEL,
        0,
        "expr_112_args_t",
        0
    );
    LibKernelTemplate(
        "expr_114_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_114_in_0"),
            TCArg("f16 *__restrict__ ", "expr_114_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_114",
        CALL_PARALLEL,
        0,
        "expr_114_args_t",
        0
    );
    LibKernelTemplate(
        "expr_115_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_115_in_0"),
            TCArg("f16 *__restrict__ ", "expr_115_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_115",
        CALL_PARALLEL,
        0,
        "expr_115_args_t",
        0
    );
    LibKernelTemplate(
        "expr_116_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_116_in_0"),
            TCArg("f16 *__restrict__ ", "expr_116_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_116",
        CALL_PARALLEL,
        0,
        "expr_116_args_t",
        0
    );
    LibKernelTemplate(
        "expr_117_args_t",
        CArgs(5,
            TCArg("f16 *__restrict__ ", "expr_117_in_0"),
            TCArg("f16 *__restrict__ ", "expr_117_out_0"),
            TCArg("unsigned short int", "W"),
            TCArg("unsigned short int", "H"),
            TCArg("unsigned short int", "Feat")
        )
    );
    
    LibKernel(
        "expr_117",
        CALL_PARALLEL,
        0,
        "expr_117_args_t",
        0
    );
}



int s10_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (160, 160, 8) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (204800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 204800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_1_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_1_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_1_out_0")
        ),
        Calls(1,
            Call("s10_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_1_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_1_in_0", KER_ARG_TILE),
                    K_Arg("expr_1_in_1", KER_ARG_TILE),
                    K_Arg("expr_1_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_1_out_0 axes: (0,)
        // var: expr_1_in_0 axes: (0,)
        // var: expr_1_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_1_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_1_out_0"),
            KerArg("expr_1_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_1_in_0"),
            KerArg("expr_1_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_1_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 204800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 614400, 0);
        AddKernelArgDimExplicit(Name, "expr_1_in_0", ARG_IEEE16,  4, 160, 160, 8, 2);
        AddKernelArgDimExplicit(Name, "expr_1_in_1", ARG_IEEE16,  4, 160, 160, 8, 2);
        AddKernelArgDimExplicit(Name, "expr_1_out_0", ARG_IEEE16, 4, 160, 160, 8, 2);
    }
    return (Kernel!=0);
}
int s29_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (80, 80, 16) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (102400.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 102400, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_7_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_7_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_7_out_0")
        ),
        Calls(1,
            Call("s29_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_7_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_7_in_0", KER_ARG_TILE),
                    K_Arg("expr_7_in_1", KER_ARG_TILE),
                    K_Arg("expr_7_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_7_out_0 axes: (0,)
        // var: expr_7_in_0 axes: (0,)
        // var: expr_7_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_7_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_7_out_0"),
            KerArg("expr_7_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_7_in_0"),
            KerArg("expr_7_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_7_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 102400, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 307200, 0);
        AddKernelArgDimExplicit(Name, "expr_7_in_0", ARG_IEEE16,  4, 80, 80, 16, 2);
        AddKernelArgDimExplicit(Name, "expr_7_in_1", ARG_IEEE16,  4, 80, 80, 16, 2);
        AddKernelArgDimExplicit(Name, "expr_7_out_0", ARG_IEEE16, 4, 80, 80, 16, 2);
    }
    return (Kernel!=0);
}
int s48_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (40, 40, 32) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (51200.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 51200, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_12_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_12_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_12_out_0")
        ),
        Calls(1,
            Call("s48_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_12_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_12_in_0", KER_ARG_TILE),
                    K_Arg("expr_12_in_1", KER_ARG_TILE),
                    K_Arg("expr_12_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_12_out_0 axes: (0,)
        // var: expr_12_in_0 axes: (0,)
        // var: expr_12_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_12_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_12_out_0"),
            KerArg("expr_12_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_12_in_0"),
            KerArg("expr_12_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_12_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 51200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 153600, 0);
        AddKernelArgDimExplicit(Name, "expr_12_in_0", ARG_IEEE16,  4, 40, 40, 32, 2);
        AddKernelArgDimExplicit(Name, "expr_12_in_1", ARG_IEEE16,  4, 40, 40, 32, 2);
        AddKernelArgDimExplicit(Name, "expr_12_out_0", ARG_IEEE16, 4, 40, 40, 32, 2);
    }
    return (Kernel!=0);
}
int s374_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 1, 8, 400) spaces: ((0,), (2, 3)) 
        // parametric_spaces: ((0,), (2, 3)) 
        // exterior_shape: (16, 3200.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 16, 8), IterParSpace(KER_ITER_D1, 3200, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_15_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_15_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_15_out_0")
        ),
        Calls(1,
            Call("s374_multiple_1_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_15_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_15_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_15_in_0", KER_ARG_TILE),
                    K_Arg("expr_15_in_1", KER_ARG_TILE),
                    K_Arg("expr_15_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_15_out_0 axes: (0, 1)
        // var: expr_15_in_1 axes: (0, 1)
        // var: expr_15_in_0 axes: (1,)
        KerArgs(3,
            KerArg("expr_15_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_15_out_0"),
            KerArg("expr_15_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_15_in_1"),
            KerArg("expr_15_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_15_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 153600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 105600, 0);
        AddKernelArgDimExplicit(Name, "expr_15_in_0", ARG_IEEE16,  3, 8, 400,        2);
        AddKernelArgDimExplicit(Name, "expr_15_in_1", ARG_IEEE16,  5, 16, 1, 8, 400, 2);
        AddKernelArgDimExplicit(Name, "expr_15_out_0", ARG_IEEE16, 5, 16, 1, 8, 400, 2);
    }
    return (Kernel!=0);
}
int s380_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 64, 1) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (25600.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 25600, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_16_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_16_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_16_out_0")
        ),
        Calls(1,
            Call("s380_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_16_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_16_in_0", KER_ARG_TILE),
                    K_Arg("expr_16_in_1", KER_ARG_TILE),
                    K_Arg("expr_16_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_16_out_0 axes: (0,)
        // var: expr_16_in_0 axes: (0,)
        // var: expr_16_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_16_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_16_out_0"),
            KerArg("expr_16_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_16_in_0"),
            KerArg("expr_16_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_16_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 76800, 0);
        AddKernelArgDimExplicit(Name, "expr_16_in_0", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_16_in_1", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_16_out_0", ARG_IEEE16, 5, 20, 20, 64, 1, 2);
    }
    return (Kernel!=0);
}
int s386_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 64, 1) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (25600.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 25600, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_17_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_17_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_17_out_0")
        ),
        Calls(1,
            Call("s386_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_17_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_17_in_0", KER_ARG_TILE),
                    K_Arg("expr_17_in_1", KER_ARG_TILE),
                    K_Arg("expr_17_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_17_out_0 axes: (0,)
        // var: expr_17_in_0 axes: (0,)
        // var: expr_17_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_17_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_17_out_0"),
            KerArg("expr_17_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_17_in_0"),
            KerArg("expr_17_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_17_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 76800, 0);
        AddKernelArgDimExplicit(Name, "expr_17_in_0", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_17_in_1", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_17_out_0", ARG_IEEE16, 5, 20, 20, 64, 1, 2);
    }
    return (Kernel!=0);
}
int s474_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 64, 1) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (25600.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 25600, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_18_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_18_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_18_out_0")
        ),
        Calls(1,
            Call("s474_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_18_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_18_in_0", KER_ARG_TILE),
                    K_Arg("expr_18_in_1", KER_ARG_TILE),
                    K_Arg("expr_18_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_18_out_0 axes: (0,)
        // var: expr_18_in_0 axes: (0,)
        // var: expr_18_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_18_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_18_out_0"),
            KerArg("expr_18_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_18_in_0"),
            KerArg("expr_18_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_18_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 76800, 0);
        AddKernelArgDimExplicit(Name, "expr_18_in_0", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_18_in_1", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_18_out_0", ARG_IEEE16, 5, 20, 20, 64, 1, 2);
    }
    return (Kernel!=0);
}
int s480_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 64, 1) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (25600.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 25600, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_19_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_19_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_19_out_0")
        ),
        Calls(1,
            Call("s480_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_19_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_19_in_0", KER_ARG_TILE),
                    K_Arg("expr_19_in_1", KER_ARG_TILE),
                    K_Arg("expr_19_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_19_out_0 axes: (0,)
        // var: expr_19_in_0 axes: (0,)
        // var: expr_19_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_19_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_19_out_0"),
            KerArg("expr_19_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_19_in_0"),
            KerArg("expr_19_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_19_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 76800, 0);
        AddKernelArgDimExplicit(Name, "expr_19_in_0", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_19_in_1", ARG_IEEE16,  5, 20, 20, 64, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_19_out_0", ARG_IEEE16, 5, 20, 20, 64, 1, 2);
    }
    return (Kernel!=0);
}
int s468_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 1, 8, 400) spaces: ((0,), (2, 3)) 
        // parametric_spaces: ((0,), (2, 3)) 
        // exterior_shape: (16, 3200.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 16, 8), IterParSpace(KER_ITER_D1, 3200, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_22_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_22_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_22_out_0")
        ),
        Calls(1,
            Call("s468_multiple_1_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_22_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_22_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_22_in_0", KER_ARG_TILE),
                    K_Arg("expr_22_in_1", KER_ARG_TILE),
                    K_Arg("expr_22_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_22_out_0 axes: (0, 1)
        // var: expr_22_in_1 axes: (0, 1)
        // var: expr_22_in_0 axes: (1,)
        KerArgs(3,
            KerArg("expr_22_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_22_out_0"),
            KerArg("expr_22_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_22_in_1"),
            KerArg("expr_22_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_22_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 153600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 105600, 0);
        AddKernelArgDimExplicit(Name, "expr_22_in_0", ARG_IEEE16,  3, 8, 400,        2);
        AddKernelArgDimExplicit(Name, "expr_22_in_1", ARG_IEEE16,  5, 16, 1, 8, 400, 2);
        AddKernelArgDimExplicit(Name, "expr_22_out_0", ARG_IEEE16, 5, 16, 1, 8, 400, 2);
    }
    return (Kernel!=0);
}
int s627_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 1, 16, 100) spaces: ((0,), (2, 3)) 
        // parametric_spaces: ((0,), (2, 3)) 
        // exterior_shape: (16, 1600.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 16, 8), IterParSpace(KER_ITER_D1, 1600, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_27_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_27_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_27_out_0")
        ),
        Calls(1,
            Call("s627_multiple_1_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_27_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_27_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_27_in_0", KER_ARG_TILE),
                    K_Arg("expr_27_in_1", KER_ARG_TILE),
                    K_Arg("expr_27_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_27_out_0 axes: (0, 1)
        // var: expr_27_in_1 axes: (0, 1)
        // var: expr_27_in_0 axes: (1,)
        KerArgs(3,
            KerArg("expr_27_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_27_out_0"),
            KerArg("expr_27_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_27_in_1"),
            KerArg("expr_27_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_27_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 76800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 52800, 0);
        AddKernelArgDimExplicit(Name, "expr_27_in_0", ARG_IEEE16,  3, 16, 100,        2);
        AddKernelArgDimExplicit(Name, "expr_27_in_1", ARG_IEEE16,  5, 16, 1, 16, 100, 2);
        AddKernelArgDimExplicit(Name, "expr_27_out_0", ARG_IEEE16, 5, 16, 1, 16, 100, 2);
    }
    return (Kernel!=0);
}
int s633_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 10, 10, 128) spaces: ((1, 2, 3),) 
        // parametric_spaces: ((1, 2, 3),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_28_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_28_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_28_out_0")
        ),
        Calls(1,
            Call("s633_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_28_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_28_in_0", KER_ARG_TILE),
                    K_Arg("expr_28_in_1", KER_ARG_TILE),
                    K_Arg("expr_28_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_28_out_0 axes: (0,)
        // var: expr_28_in_0 axes: (0,)
        // var: expr_28_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_28_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_28_out_0"),
            KerArg("expr_28_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_28_in_0"),
            KerArg("expr_28_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_28_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 12800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 38400, 0);
        AddKernelArgDimExplicit(Name, "expr_28_in_0", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_28_in_1", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_28_out_0", ARG_IEEE16, 4, 10, 10, 128, 2);
    }
    return (Kernel!=0);
}
int s639_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 10, 10, 128) spaces: ((1, 2, 3),) 
        // parametric_spaces: ((1, 2, 3),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_29_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_29_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_29_out_0")
        ),
        Calls(1,
            Call("s639_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_29_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_29_in_0", KER_ARG_TILE),
                    K_Arg("expr_29_in_1", KER_ARG_TILE),
                    K_Arg("expr_29_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_29_out_0 axes: (0,)
        // var: expr_29_in_0 axes: (0,)
        // var: expr_29_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_29_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_29_out_0"),
            KerArg("expr_29_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_29_in_0"),
            KerArg("expr_29_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_29_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 12800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 38400, 0);
        AddKernelArgDimExplicit(Name, "expr_29_in_0", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_29_in_1", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_29_out_0", ARG_IEEE16, 4, 10, 10, 128, 2);
    }
    return (Kernel!=0);
}
int s787_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 10, 10, 128) spaces: ((1, 2, 3),) 
        // parametric_spaces: ((1, 2, 3),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_30_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_30_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_30_out_0")
        ),
        Calls(1,
            Call("s787_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_30_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_30_in_0", KER_ARG_TILE),
                    K_Arg("expr_30_in_1", KER_ARG_TILE),
                    K_Arg("expr_30_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_30_out_0 axes: (0,)
        // var: expr_30_in_0 axes: (0,)
        // var: expr_30_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_30_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_30_out_0"),
            KerArg("expr_30_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_30_in_0"),
            KerArg("expr_30_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_30_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 12800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 38400, 0);
        AddKernelArgDimExplicit(Name, "expr_30_in_0", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_30_in_1", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_30_out_0", ARG_IEEE16, 4, 10, 10, 128, 2);
    }
    return (Kernel!=0);
}
int s793_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 10, 10, 128) spaces: ((1, 2, 3),) 
        // parametric_spaces: ((1, 2, 3),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_31_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_31_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_31_out_0")
        ),
        Calls(1,
            Call("s793_multiple_1_kernel", LOC_D0,
                Bindings(4,
                    K_ArgPar("expr_31_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_31_in_0", KER_ARG_TILE),
                    K_Arg("expr_31_in_1", KER_ARG_TILE),
                    K_Arg("expr_31_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_31_out_0 axes: (0,)
        // var: expr_31_in_0 axes: (0,)
        // var: expr_31_in_1 axes: (0,)
        KerArgs(3,
            KerArg("expr_31_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_31_out_0"),
            KerArg("expr_31_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_31_in_0"),
            KerArg("expr_31_in_1",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_31_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 12800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 38400, 0);
        AddKernelArgDimExplicit(Name, "expr_31_in_0", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_31_in_1", ARG_IEEE16,  4, 10, 10, 128, 2);
        AddKernelArgDimExplicit(Name, "expr_31_out_0", ARG_IEEE16, 4, 10, 10, 128, 2);
    }
    return (Kernel!=0);
}
int s781_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (16, 1, 16, 100) spaces: ((0,), (2, 3)) 
        // parametric_spaces: ((0,), (2, 3)) 
        // exterior_shape: (16, 1600.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 16, 8), IterParSpace(KER_ITER_D1, 1600, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_34_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_34_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_34_out_0")
        ),
        Calls(1,
            Call("s781_multiple_1_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_34_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_34_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_34_in_0", KER_ARG_TILE),
                    K_Arg("expr_34_in_1", KER_ARG_TILE),
                    K_Arg("expr_34_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_34_out_0 axes: (0, 1)
        // var: expr_34_in_1 axes: (0, 1)
        // var: expr_34_in_0 axes: (1,)
        KerArgs(3,
            KerArg("expr_34_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_34_out_0"),
            KerArg("expr_34_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_34_in_1"),
            KerArg("expr_34_in_0",  KerArgSpace(1, KER_ITER_D1),              O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_34_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 76800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 52800, 0);
        AddKernelArgDimExplicit(Name, "expr_34_in_0", ARG_IEEE16,  3, 16, 100,        2);
        AddKernelArgDimExplicit(Name, "expr_34_in_1", ARG_IEEE16,  5, 16, 1, 16, 100, 2);
        AddKernelArgDimExplicit(Name, "expr_34_out_0", ARG_IEEE16, 5, 16, 1, 16, 100, 2);
    }
    return (Kernel!=0);
}
int s803_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (1, 20, 20, 32) spaces: ((1, 2, 3),) 
        // parametric_spaces: ((1, 2, 3),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_38_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_38_out_0")
        ),
        Calls(1,
            Call("s803_multiple_1_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_38_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_38_in_0", KER_ARG_TILE),
                    K_Arg("expr_38_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_38_out_0 axes: (0,)
        // var: expr_38_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_38_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_38_out_0"),
            KerArg("expr_38_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_38_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 25600, 0);
        AddKernelArgDimExplicit(Name, "expr_38_in_0", ARG_IEEE16,  4, 20, 20, 32, 2);
        AddKernelArgDimExplicit(Name, "expr_38_out_0", ARG_IEEE16, 4, 20, 20, 32, 2);
    }
    return (Kernel!=0);
}
int s816_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 32, 1) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (12800.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 12800, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_48_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_48_out_0")
        ),
        Calls(1,
            Call("s816_multiple_1_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_48_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_48_in_0", KER_ARG_TILE),
                    K_Arg("expr_48_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_48_out_0 axes: (0,)
        // var: expr_48_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_48_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_48_out_0"),
            KerArg("expr_48_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_48_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 25600, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 25600, 0);
        AddKernelArgDimExplicit(Name, "expr_48_in_0", ARG_IEEE16,  5, 20, 20, 32, 1, 2);
        AddKernelArgDimExplicit(Name, "expr_48_out_0", ARG_IEEE16, 5, 20, 20, 32, 1, 2);
    }
    return (Kernel!=0);
}
int s843_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (40, 40, 64) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (102400.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 102400, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_67_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_67_out_0")
        ),
        Calls(1,
            Call("s843_multiple_1_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_67_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_67_in_0", KER_ARG_TILE),
                    K_Arg("expr_67_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_67_out_0 axes: (0,)
        // var: expr_67_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_67_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_67_out_0"),
            KerArg("expr_67_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_67_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 204800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 204800, 0);
        AddKernelArgDimExplicit(Name, "expr_67_in_0", ARG_IEEE16,  4, 40, 40, 64, 2);
        AddKernelArgDimExplicit(Name, "expr_67_out_0", ARG_IEEE16, 4, 40, 40, 64, 2);
    }
    return (Kernel!=0);
}
int s874_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (20, 20, 64) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (25600.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 25600, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_90_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_90_out_0")
        ),
        Calls(1,
            Call("s874_multiple_1_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_90_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_90_in_0", KER_ARG_TILE),
                    K_Arg("expr_90_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_90_out_0 axes: (0,)
        // var: expr_90_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_90_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_90_out_0"),
            KerArg("expr_90_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_90_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 51200, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 51200, 0);
        AddKernelArgDimExplicit(Name, "expr_90_in_0", ARG_IEEE16,  4, 20, 20, 64, 2);
        AddKernelArgDimExplicit(Name, "expr_90_out_0", ARG_IEEE16, 4, 20, 20, 64, 2);
    }
    return (Kernel!=0);
}
int s903_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (10, 10, 64) spaces: ((0, 1, 2),) 
        // parametric_spaces: ((0, 1, 2),) 
        // exterior_shape: (6400.0,) 
        KernelIterSpace(2, IterParSpace(KER_ITER_D0, 6400, 8), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(2,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_113_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_113_out_0")
        ),
        Calls(1,
            Call("s903_multiple_1_kernel", LOC_D0,
                Bindings(3,
                    K_ArgPar("expr_113_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_Arg("expr_113_in_0", KER_ARG_TILE),
                    K_Arg("expr_113_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_113_out_0 axes: (0,)
        // var: expr_113_in_0 axes: (0,)
        KerArgs(2,
            KerArg("expr_113_out_0", KerArgSpace(1, KER_ITER_D0), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_113_out_0"),
            KerArg("expr_113_in_0",  KerArgSpace(1, KER_ITER_D0), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_113_in_0")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 12800, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 12800, 0);
        AddKernelArgDimExplicit(Name, "expr_113_in_0", ARG_IEEE16,  4, 10, 10, 64, 2);
        AddKernelArgDimExplicit(Name, "expr_113_out_0", ARG_IEEE16, 4, 10, 10, 64, 2);
    }
    return (Kernel!=0);
}
int s915_multiple_1_kernel_gen(char *Name) {
    Kernel_T *Kernel = UserKernel(
        Name,
        // shape: (2100, 80) spaces: ((0,), (1,)) 
        // parametric_spaces: ((0,), (1,)) 
        // exterior_shape: (2100, 80.0) 
        KernelIterSpace(3, IterParSpace(KER_ITER_D0, 2100, 8), IterParSpace(KER_ITER_D1, 80, 1), IterTiledSpace(KER_ITER_TILE0)),
        TILE_VER,
        CArgs(3,
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_118_in_0"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_118_in_1"),
            TCArg(CNN_ArgDataTypeExplicit(2, 1, 1, ARG_IEEE16), "expr_118_out_0")
        ),
        Calls(1,
            Call("s915_multiple_1_kernel", LOC_D1,
                Bindings(5,
                    K_ArgPar("expr_118_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D0),
                    K_ArgPar("expr_118_out_0", KER_ARG_PARTILE_SIZE, KER_ITER_D1),
                    K_Arg("expr_118_in_0", KER_ARG_TILE),
                    K_Arg("expr_118_in_1", KER_ARG_TILE),
                    K_Arg("expr_118_out_0", KER_ARG_TILE)
                )
            )
        ),
        // var: expr_118_out_0 axes: (0, 1)
        // var: expr_118_in_0 axes: (0,)
        // var: expr_118_in_1 axes: (0, 1)
        KerArgs(3,
            KerArg("expr_118_out_0", KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_OUT|O_DB, 1, 1, 2, 0, 0, 0, "expr_118_out_0"),
            KerArg("expr_118_in_0",  KerArgSpace(1, KER_ITER_D0),              O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_118_in_0"),
            KerArg("expr_118_in_1",  KerArgSpace(2, KER_ITER_D0, KER_ITER_D1), O_IN|O_DB,  1, 1, 2, 0, 0, 0, "expr_118_in_1")
        )
    );
    if (Kernel) {
        AddKernelInfos(Name, AT_KERINFO_OPER, 168000, 0);
        AddKernelInfos(Name, AT_KERINFO_BANDWIDTH, 338100, 0);
        AddKernelArgDimExplicit(Name, "expr_118_in_0", ARG_IEEE16,  3, 2100, 1,  2);
        AddKernelArgDimExplicit(Name, "expr_118_in_1", ARG_IEEE16,  3, 2100, 80, 2);
        AddKernelArgDimExplicit(Name, "expr_118_out_0", ARG_IEEE16, 3, 2100, 80, 2);
    }
    return (Kernel!=0);
}

void onnx_graphModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 8, "at_api.h", "onnx_graph.h", "CNN_BasicKernels_fp32.h", "CNN_BasicKernels_f16.h", "CNN_BasicKernels_f16a.h", "ResizeBasicKernels.h", "CNN_BasicKernels_SQ8.h", "Expression_Kernels.h");
    SetGeneratedFilesNames("onnx_graphKernels.c", "onnx_graphKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_CONST_EXEC_FROM_FLASH, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "onnx_graph_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "onnx_graph_L2_Memory", 0, 1,
        AT_MEM_L3_DEFAULTRAM, L3Memory, "onnx_graph_L3_Memory", 0, 0,
        AT_MEM_L3_DEFAULTFLASH, L3Flash, "onnx_graph_L3_Flash", "onnx_graph_L3_Flash_Const.dat", 0
    );

    LoadCNNLibrary_fp16();
    LoadResizeLibrary();
    LoadCNN_Copy_Library();
    load_expressions_kernels();

    CNN_GenControl_T gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_0");
    // generator for _backbone_backbone_stem_in_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 3, 8, 320, 320,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_2");
    // generator for _backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 160, 160,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stem_res0_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 160, 160,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_1
    s10_multiple_1_kernel_gen("S10_expr_1_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_3");
    // generator for _backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 32, 160, 160,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_4");
    // generator for _backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 160, 160,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 16, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_5");
    // generator for _backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 64, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_6");
    // generator for _backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 16, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_7
    s29_multiple_1_kernel_gen("S29_expr_7_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_8");
    // generator for _backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 64, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_9");
    // generator for _backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 80, 80,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 32, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_10");
    // generator for _backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 128, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_11");
    // generator for _backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 128, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 32, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_12
    s48_multiple_1_kernel_gen("S48_expr_12_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_13");
    // generator for _backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 128, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_14");
    // generator for _backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 128, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1", &gen_ctrl_S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 192, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans Transpose 20x20x192 -> 192x20x20 ((1, 0))
    CNN_MatTranspose("S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1", &gen_ctrl_S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, 2,
                      1, 192, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv
    CNN_ConvolutionPoolAct_fp16("S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1", &gen_ctrl_S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 192, 192, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1", &gen_ctrl_S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, 2, 400, 12, KOP_SPLIT, 16,16,16,16,16,16,16,16,16,16,16,16);
    
    CNN_GenControl_T gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0
    CNN_ConvolutionPoolAct_fp16("S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1", &gen_ctrl_S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1", &gen_ctrl_S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1
    CNN_ConvolutionPoolAct_fp16("S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1", &gen_ctrl_S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1", &gen_ctrl_S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2
    CNN_ConvolutionPoolAct_fp16("S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1", &gen_ctrl_S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1", &gen_ctrl_S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3
    CNN_ConvolutionPoolAct_fp16("S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1", &gen_ctrl_S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1", &gen_ctrl_S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4
    CNN_ConvolutionPoolAct_fp16("S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1", &gen_ctrl_S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1", &gen_ctrl_S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5
    CNN_ConvolutionPoolAct_fp16("S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1", &gen_ctrl_S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1", &gen_ctrl_S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6
    CNN_ConvolutionPoolAct_fp16("S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1", &gen_ctrl_S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1", &gen_ctrl_S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7
    CNN_ConvolutionPoolAct_fp16("S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1", &gen_ctrl_S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1", &gen_ctrl_S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8
    CNN_ConvolutionPoolAct_fp16("S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1", &gen_ctrl_S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1", &gen_ctrl_S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9
    CNN_ConvolutionPoolAct_fp16("S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1", &gen_ctrl_S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1", &gen_ctrl_S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10
    CNN_ConvolutionPoolAct_fp16("S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1", &gen_ctrl_S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1", &gen_ctrl_S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11
    CNN_ConvolutionPoolAct_fp16("S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1", &gen_ctrl_S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1", &gen_ctrl_S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin Transpose 8x48x400 -> 8x400x48 ((0, 2, 1))
    CNN_3DTensorPermute("S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1", &gen_ctrl_S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, 2,
                         8, 400, 48, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1", &gen_ctrl_S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1, 2, 3200, 3, KOP_SPLIT, 16,16,16);
    
    CNN_GenControl_T gen_ctrl_S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu
    CNN_ConvolutionPoolAct_fp16("S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1", &gen_ctrl_S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 16, 400,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu
    CNN_ConvolutionPoolAct_fp16("S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1", &gen_ctrl_S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 16, 400,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans Transpose 1x8x400x16 -> 1x8x16x400 ((0, 2, 1))
    CNN_3DTensorPermute("S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1", &gen_ctrl_S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, 2,
                         8, 16, 400, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad
    CNN_Padding_Generator(
        "S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1", &gen_ctrl_S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1, 0, 2,
        1,                // ParSpaceDim (>0 if first, <0 if last, 0 or 1 if no)
        3200, 16,                 // Dim1, Dim2
        0, 0,    // PadBefore1, PadAfter1
        0, 1     // PadBefore2, PadAfter2
    );
    
    CNN_GenControl_T gen_ctrl_S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul
    CNN_BatchedMatMulAct_fp16("S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1", &gen_ctrl_S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1, 
                              8, /* NBatches */
                              400, 16, /* W1, H1 */
                              17, 400, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3 Transpose 1x8x16x17 -> 1x8x17x16 ((0, 2, 1))
    CNN_3DTensorPermute("S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1", &gen_ctrl_S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, 2,
                         8, 17, 16, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1
    CNN_BatchedMatMulAct_fp16("S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1", &gen_ctrl_S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1, 
                              8, /* NBatches */
                              16, 17, /* W1, H1 */
                              400, 16, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL_TRANSPOSED, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin Transpose 1x8x17x400 -> 17x1x8x400 ((1, 0, 2))
    CNN_3DTensorPermute("S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1", &gen_ctrl_S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, 2,
                         8, 400, 17, KOP_MATPERM_CHW2HCW);
    
    
    // generator for expr_15
    s374_multiple_1_kernel_gen("S374_expr_15_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans Transpose 16x1x8x400 -> 1x8x16x400 ((1, 0, 2))
    CNN_3DTensorPermute("S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1", &gen_ctrl_S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, 2,
                         16, 400, 8, KOP_MATPERM_CHW2HCW);
    
    CNN_GenControl_T gen_ctrl_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0 Transpose 128x20x20 -> 20x20x128 ((1, 0))
    CNN_MatTranspose("S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1", &gen_ctrl_S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, 2,
                      1, 400, 128, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1", &gen_ctrl_S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_16
    s380_multiple_1_kernel_gen("S380_expr_16_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_20");
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_21");
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_17
    s386_multiple_1_kernel_gen("S386_expr_17_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1", &gen_ctrl_S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 192, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv
    CNN_ConvolutionPoolAct_fp16("S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1", &gen_ctrl_S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 192, 192, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1", &gen_ctrl_S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, 2, 400, 12, KOP_SPLIT, 16,16,16,16,16,16,16,16,16,16,16,16);
    
    CNN_GenControl_T gen_ctrl_S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans Transpose 20x20x192x1 -> 192x1x20x20 ((1, 0))
    CNN_MatTranspose("S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1", &gen_ctrl_S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, 2,
                      1, 192, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0
    CNN_ConvolutionPoolAct_fp16("S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1", &gen_ctrl_S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1", &gen_ctrl_S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1
    CNN_ConvolutionPoolAct_fp16("S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1", &gen_ctrl_S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1", &gen_ctrl_S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2
    CNN_ConvolutionPoolAct_fp16("S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1", &gen_ctrl_S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1", &gen_ctrl_S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3
    CNN_ConvolutionPoolAct_fp16("S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1", &gen_ctrl_S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1", &gen_ctrl_S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4
    CNN_ConvolutionPoolAct_fp16("S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1", &gen_ctrl_S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1", &gen_ctrl_S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5
    CNN_ConvolutionPoolAct_fp16("S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1", &gen_ctrl_S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1", &gen_ctrl_S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6
    CNN_ConvolutionPoolAct_fp16("S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1", &gen_ctrl_S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1", &gen_ctrl_S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7
    CNN_ConvolutionPoolAct_fp16("S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1", &gen_ctrl_S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1", &gen_ctrl_S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8
    CNN_ConvolutionPoolAct_fp16("S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1", &gen_ctrl_S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1", &gen_ctrl_S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9
    CNN_ConvolutionPoolAct_fp16("S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1", &gen_ctrl_S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1", &gen_ctrl_S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10
    CNN_ConvolutionPoolAct_fp16("S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1", &gen_ctrl_S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1", &gen_ctrl_S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11
    CNN_ConvolutionPoolAct_fp16("S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1", &gen_ctrl_S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 Transpose 20x20x16 -> 16x20x20 ((1, 0))
    CNN_MatTranspose("S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1", &gen_ctrl_S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, 2,
                      1, 16, 400, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin Transpose 1x8x48x400 -> 1x8x400x48 ((0, 2, 1))
    CNN_3DTensorPermute("S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1", &gen_ctrl_S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, 2,
                         8, 400, 48, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1", &gen_ctrl_S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1, 2, 3200, 3, KOP_SPLIT, 16,16,16);
    
    CNN_GenControl_T gen_ctrl_S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu
    CNN_ConvolutionPoolAct_fp16("S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1", &gen_ctrl_S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 16, 400,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu
    CNN_ConvolutionPoolAct_fp16("S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1", &gen_ctrl_S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 8, 8, 16, 400,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans Transpose 1x8x400x16 -> 1x8x16x400 ((0, 2, 1))
    CNN_3DTensorPermute("S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1", &gen_ctrl_S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, 2,
                         8, 16, 400, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad
    CNN_Padding_Generator(
        "S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1", &gen_ctrl_S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1, 0, 2,
        1,                // ParSpaceDim (>0 if first, <0 if last, 0 or 1 if no)
        3200, 16,                 // Dim1, Dim2
        0, 0,    // PadBefore1, PadAfter1
        0, 1     // PadBefore2, PadAfter2
    );
    
    CNN_GenControl_T gen_ctrl_S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul
    CNN_BatchedMatMulAct_fp16("S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1", &gen_ctrl_S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1, 
                              8, /* NBatches */
                              400, 16, /* W1, H1 */
                              17, 400, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3 Transpose 1x8x16x17 -> 1x8x17x16 ((0, 2, 1))
    CNN_3DTensorPermute("S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1", &gen_ctrl_S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, 2,
                         8, 17, 16, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1
    CNN_BatchedMatMulAct_fp16("S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1", &gen_ctrl_S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1, 
                              8, /* NBatches */
                              16, 17, /* W1, H1 */
                              400, 16, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL_TRANSPOSED, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin Transpose 1x8x17x400 -> 17x1x8x400 ((1, 0, 2))
    CNN_3DTensorPermute("S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1", &gen_ctrl_S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, 2,
                         8, 400, 17, KOP_MATPERM_CHW2HCW);
    
    
    // generator for expr_22
    s468_multiple_1_kernel_gen("S468_expr_22_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans Transpose 16x1x8x400 -> 1x8x16x400 ((1, 0, 2))
    CNN_3DTensorPermute("S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1", &gen_ctrl_S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, 2,
                         16, 400, 8, KOP_MATPERM_CHW2HCW);
    
    CNN_GenControl_T gen_ctrl_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0 Transpose 128x20x20 -> 20x20x128 ((1, 0))
    CNN_MatTranspose("S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1", &gen_ctrl_S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, 2,
                      1, 400, 128, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1", &gen_ctrl_S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_18
    s474_multiple_1_kernel_gen("S474_expr_18_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_23");
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_24");
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_19
    s480_multiple_1_kernel_gen("S480_expr_19_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_25");
    // generator for _backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_26");
    // generator for _backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 256, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1", &gen_ctrl_S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 384, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv
    CNN_ConvolutionPoolAct_fp16("S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1", &gen_ctrl_S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 384, 384, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1", &gen_ctrl_S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, 2, 100, 24, KOP_SPLIT, 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16);
    
    CNN_GenControl_T gen_ctrl_S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans Transpose 10x10x384x1 -> 384x1x10x10 ((1, 0))
    CNN_MatTranspose("S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1", &gen_ctrl_S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1, 2,
                      1, 384, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0
    CNN_ConvolutionPoolAct_fp16("S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1", &gen_ctrl_S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1", &gen_ctrl_S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1
    CNN_ConvolutionPoolAct_fp16("S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1", &gen_ctrl_S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1", &gen_ctrl_S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2
    CNN_ConvolutionPoolAct_fp16("S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1", &gen_ctrl_S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1", &gen_ctrl_S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3
    CNN_ConvolutionPoolAct_fp16("S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1", &gen_ctrl_S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1", &gen_ctrl_S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4
    CNN_ConvolutionPoolAct_fp16("S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1", &gen_ctrl_S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1", &gen_ctrl_S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5
    CNN_ConvolutionPoolAct_fp16("S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1", &gen_ctrl_S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1", &gen_ctrl_S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6
    CNN_ConvolutionPoolAct_fp16("S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1", &gen_ctrl_S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1", &gen_ctrl_S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7
    CNN_ConvolutionPoolAct_fp16("S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1", &gen_ctrl_S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1", &gen_ctrl_S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8
    CNN_ConvolutionPoolAct_fp16("S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1", &gen_ctrl_S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1", &gen_ctrl_S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9
    CNN_ConvolutionPoolAct_fp16("S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1", &gen_ctrl_S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1", &gen_ctrl_S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10
    CNN_ConvolutionPoolAct_fp16("S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1", &gen_ctrl_S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1", &gen_ctrl_S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11
    CNN_ConvolutionPoolAct_fp16("S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1", &gen_ctrl_S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1", &gen_ctrl_S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12
    CNN_ConvolutionPoolAct_fp16("S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1", &gen_ctrl_S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1", &gen_ctrl_S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13
    CNN_ConvolutionPoolAct_fp16("S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1", &gen_ctrl_S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1", &gen_ctrl_S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14
    CNN_ConvolutionPoolAct_fp16("S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1", &gen_ctrl_S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1", &gen_ctrl_S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15
    CNN_ConvolutionPoolAct_fp16("S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1", &gen_ctrl_S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1", &gen_ctrl_S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16
    CNN_ConvolutionPoolAct_fp16("S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1", &gen_ctrl_S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1", &gen_ctrl_S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17
    CNN_ConvolutionPoolAct_fp16("S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1", &gen_ctrl_S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1", &gen_ctrl_S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18
    CNN_ConvolutionPoolAct_fp16("S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1", &gen_ctrl_S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1", &gen_ctrl_S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19
    CNN_ConvolutionPoolAct_fp16("S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1", &gen_ctrl_S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1", &gen_ctrl_S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20
    CNN_ConvolutionPoolAct_fp16("S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1", &gen_ctrl_S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1", &gen_ctrl_S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21
    CNN_ConvolutionPoolAct_fp16("S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1", &gen_ctrl_S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1", &gen_ctrl_S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22
    CNN_ConvolutionPoolAct_fp16("S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1", &gen_ctrl_S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1", &gen_ctrl_S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23
    CNN_ConvolutionPoolAct_fp16("S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1", &gen_ctrl_S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1", &gen_ctrl_S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin Transpose 1x16x48x100 -> 1x16x100x48 ((0, 2, 1))
    CNN_3DTensorPermute("S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1", &gen_ctrl_S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1, 2,
                         16, 100, 48, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1", &gen_ctrl_S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1, 2, 1600, 3, KOP_SPLIT, 16,16,16);
    
    CNN_GenControl_T gen_ctrl_S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu
    CNN_ConvolutionPoolAct_fp16("S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1", &gen_ctrl_S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 16, 100,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu
    CNN_ConvolutionPoolAct_fp16("S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1", &gen_ctrl_S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 16, 100,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans Transpose 1x16x100x16 -> 1x16x16x100 ((0, 2, 1))
    CNN_3DTensorPermute("S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1", &gen_ctrl_S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1, 2,
                         16, 16, 100, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad
    CNN_Padding_Generator(
        "S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1", &gen_ctrl_S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1, 0, 2,
        1,                // ParSpaceDim (>0 if first, <0 if last, 0 or 1 if no)
        1600, 16,                 // Dim1, Dim2
        0, 0,    // PadBefore1, PadAfter1
        0, 1     // PadBefore2, PadAfter2
    );
    
    CNN_GenControl_T gen_ctrl_S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul
    CNN_BatchedMatMulAct_fp16("S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1", &gen_ctrl_S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1, 
                              16, /* NBatches */
                              100, 16, /* W1, H1 */
                              17, 100, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3 Transpose 1x16x16x17 -> 1x16x17x16 ((0, 2, 1))
    CNN_3DTensorPermute("S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1", &gen_ctrl_S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1, 2,
                         16, 17, 16, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1
    CNN_BatchedMatMulAct_fp16("S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1", &gen_ctrl_S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1, 
                              16, /* NBatches */
                              16, 17, /* W1, H1 */
                              100, 16, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL_TRANSPOSED, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin Transpose 1x16x17x100 -> 17x1x16x100 ((1, 0, 2))
    CNN_3DTensorPermute("S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1", &gen_ctrl_S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1, 2,
                         16, 100, 17, KOP_MATPERM_CHW2HCW);
    
    
    // generator for expr_27
    s627_multiple_1_kernel_gen("S627_expr_27_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans Transpose 16x1x16x100 -> 1x16x16x100 ((1, 0, 2))
    CNN_3DTensorPermute("S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1", &gen_ctrl_S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1, 2,
                         16, 100, 16, KOP_MATPERM_CHW2HCW);
    
    CNN_GenControl_T gen_ctrl_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0 Transpose 256x10x10 -> 10x10x256 ((1, 0))
    CNN_MatTranspose("S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1", &gen_ctrl_S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1, 2,
                      1, 100, 256, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1", &gen_ctrl_S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_28
    s633_multiple_1_kernel_gen("S633_expr_28_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_32");
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 512, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_33");
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 512, 512, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 512, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_29
    s639_multiple_1_kernel_gen("S639_expr_29_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1", &gen_ctrl_S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 384, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv
    CNN_ConvolutionPoolAct_fp16("S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1", &gen_ctrl_S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 384, 384, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 5, 5, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1", &gen_ctrl_S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1, 2, 100, 24, KOP_SPLIT, 16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16);
    
    CNN_GenControl_T gen_ctrl_S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans Transpose 10x10x384x1 -> 384x1x10x10 ((1, 0))
    CNN_MatTranspose("S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1", &gen_ctrl_S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1, 2,
                      1, 384, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0
    CNN_ConvolutionPoolAct_fp16("S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1", &gen_ctrl_S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1", &gen_ctrl_S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1
    CNN_ConvolutionPoolAct_fp16("S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1", &gen_ctrl_S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1", &gen_ctrl_S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2
    CNN_ConvolutionPoolAct_fp16("S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1", &gen_ctrl_S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1", &gen_ctrl_S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3
    CNN_ConvolutionPoolAct_fp16("S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1", &gen_ctrl_S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1", &gen_ctrl_S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4
    CNN_ConvolutionPoolAct_fp16("S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1", &gen_ctrl_S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1", &gen_ctrl_S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5
    CNN_ConvolutionPoolAct_fp16("S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1", &gen_ctrl_S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1", &gen_ctrl_S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6
    CNN_ConvolutionPoolAct_fp16("S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1", &gen_ctrl_S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1", &gen_ctrl_S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7
    CNN_ConvolutionPoolAct_fp16("S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1", &gen_ctrl_S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1", &gen_ctrl_S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8
    CNN_ConvolutionPoolAct_fp16("S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1", &gen_ctrl_S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1", &gen_ctrl_S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9
    CNN_ConvolutionPoolAct_fp16("S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1", &gen_ctrl_S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1", &gen_ctrl_S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10
    CNN_ConvolutionPoolAct_fp16("S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1", &gen_ctrl_S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1", &gen_ctrl_S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11
    CNN_ConvolutionPoolAct_fp16("S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1", &gen_ctrl_S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1", &gen_ctrl_S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12
    CNN_ConvolutionPoolAct_fp16("S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1", &gen_ctrl_S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1", &gen_ctrl_S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13
    CNN_ConvolutionPoolAct_fp16("S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1", &gen_ctrl_S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1", &gen_ctrl_S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14
    CNN_ConvolutionPoolAct_fp16("S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1", &gen_ctrl_S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1", &gen_ctrl_S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15
    CNN_ConvolutionPoolAct_fp16("S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1", &gen_ctrl_S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1", &gen_ctrl_S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16
    CNN_ConvolutionPoolAct_fp16("S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1", &gen_ctrl_S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1", &gen_ctrl_S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17
    CNN_ConvolutionPoolAct_fp16("S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1", &gen_ctrl_S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1", &gen_ctrl_S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18
    CNN_ConvolutionPoolAct_fp16("S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1", &gen_ctrl_S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1", &gen_ctrl_S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19
    CNN_ConvolutionPoolAct_fp16("S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1", &gen_ctrl_S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1", &gen_ctrl_S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20
    CNN_ConvolutionPoolAct_fp16("S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1", &gen_ctrl_S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1", &gen_ctrl_S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21
    CNN_ConvolutionPoolAct_fp16("S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1", &gen_ctrl_S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1", &gen_ctrl_S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22
    CNN_ConvolutionPoolAct_fp16("S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1", &gen_ctrl_S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1", &gen_ctrl_S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23
    CNN_ConvolutionPoolAct_fp16("S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1", &gen_ctrl_S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0 Transpose 10x10x16 -> 16x10x10 ((1, 0))
    CNN_MatTranspose("S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1", &gen_ctrl_S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1, 2,
                      1, 16, 100, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin Transpose 1x16x48x100 -> 1x16x100x48 ((0, 2, 1))
    CNN_3DTensorPermute("S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1", &gen_ctrl_S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1, 2,
                         16, 100, 48, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1", &gen_ctrl_S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1, 2, 1600, 3, KOP_SPLIT, 16,16,16);
    
    CNN_GenControl_T gen_ctrl_S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu
    CNN_ConvolutionPoolAct_fp16("S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1", &gen_ctrl_S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 16, 100,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu
    CNN_ConvolutionPoolAct_fp16("S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1", &gen_ctrl_S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 16, 100,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans Transpose 1x16x100x16 -> 1x16x16x100 ((0, 2, 1))
    CNN_3DTensorPermute("S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1", &gen_ctrl_S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1, 2,
                         16, 16, 100, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad
    CNN_Padding_Generator(
        "S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1", &gen_ctrl_S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1, 0, 2,
        1,                // ParSpaceDim (>0 if first, <0 if last, 0 or 1 if no)
        1600, 16,                 // Dim1, Dim2
        0, 0,    // PadBefore1, PadAfter1
        0, 1     // PadBefore2, PadAfter2
    );
    
    CNN_GenControl_T gen_ctrl_S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul
    CNN_BatchedMatMulAct_fp16("S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1", &gen_ctrl_S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1, 
                              16, /* NBatches */
                              100, 16, /* W1, H1 */
                              17, 100, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3 Transpose 1x16x16x17 -> 1x16x17x16 ((0, 2, 1))
    CNN_3DTensorPermute("S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1", &gen_ctrl_S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1, 2,
                         16, 17, 16, KOP_MATPERM_CHW2CWH);
    
    CNN_GenControl_T gen_ctrl_S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1
    CNN_BatchedMatMulAct_fp16("S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1", &gen_ctrl_S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1, 
                              16, /* NBatches */
                              16, 17, /* W1, H1 */
                              100, 16, /* W2, H2 */
                              0, 0, 1, 1, /* w, h, strides (used only for im2col convs) */
                              KOP_MATMUL_TRANSPOSED, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin Transpose 1x16x17x100 -> 17x1x16x100 ((1, 0, 2))
    CNN_3DTensorPermute("S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1", &gen_ctrl_S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1, 2,
                         16, 100, 17, KOP_MATPERM_CHW2HCW);
    
    
    // generator for expr_34
    s781_multiple_1_kernel_gen("S781_expr_34_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans Transpose 16x1x16x100 -> 1x16x16x100 ((1, 0, 2))
    CNN_3DTensorPermute("S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1", &gen_ctrl_S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1, 2,
                         16, 100, 16, KOP_MATPERM_CHW2HCW);
    
    CNN_GenControl_T gen_ctrl_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0 Transpose 256x10x10 -> 10x10x256 ((1, 0))
    CNN_MatTranspose("S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1", &gen_ctrl_S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1, 2,
                      1, 100, 256, KOP_MATTRANSP);
    
    CNN_GenControl_T gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1", &gen_ctrl_S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 256, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_30
    s787_multiple_1_kernel_gen("S787_expr_30_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_35");
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 512, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_36");
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1", &gen_ctrl_S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 512, 512, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1", &gen_ctrl_S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 512, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_31
    s793_multiple_1_kernel_gen("S793_expr_31_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_37");
    // generator for _backbone_lateral_conv0_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1", &gen_ctrl_S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S796__backbone_upsample_Resize_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S796__backbone_upsample_Resize_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S796__backbone_upsample_Resize_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S796__backbone_upsample_Resize_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_upsample_Resize
    GenerateResizeMultiChannel_fp16("S796__backbone_upsample_Resize_multiple_1", &gen_ctrl_S796__backbone_upsample_Resize_multiple_1, 10, 10, 20, 20, 64, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S798__backbone_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S798__backbone_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S798__backbone_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S798__backbone_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S798__backbone_Concat_multiple_1", &gen_ctrl_S798__backbone_Concat_multiple_1, 2, 400, 2, KOP_CONCAT, 64,64);
    
    CNN_GenControl_T gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _backbone_C3_p4_conv1_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S800__backbone_C3_p4_conv1_conv_Conv_multiple_1", &gen_ctrl_S800__backbone_C3_p4_conv1_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1", &gen_ctrl_S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1, 2, 400, 2, KOP_SPLIT, 32,32);
    
    
    // generator for expr_38
    s803_multiple_1_kernel_gen("S803_expr_38_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_39");
    // generator for _backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_40");
    // generator for _backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_41");
    // generator for _backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_42");
    // generator for _backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_43");
    // generator for _backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_44");
    // generator for _backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_45");
    // generator for _backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_46");
    // generator for _backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_47");
    // generator for _backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    
    // generator for expr_48
    s816_multiple_1_kernel_gen("S816_expr_48_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S817__backbone_C3_p4_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S817__backbone_C3_p4_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S817__backbone_C3_p4_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S817__backbone_C3_p4_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S817__backbone_C3_p4_Concat_multiple_1", &gen_ctrl_S817__backbone_C3_p4_Concat_multiple_1, 2, 400, 2, KOP_CONCAT, 32,32);
    
    CNN_GenControl_T gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_49");
    // generator for _backbone_C3_p4_conv3_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1", &gen_ctrl_S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_50");
    // generator for _backbone_reduce_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S821__backbone_upsample_1_Resize_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S821__backbone_upsample_1_Resize_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S821__backbone_upsample_1_Resize_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S821__backbone_upsample_1_Resize_multiple_1, "HWC", AT_OPT_VAL(1));
    // generator for _backbone_upsample_1_Resize
    GenerateResizeMultiChannel_fp16("S821__backbone_upsample_1_Resize_multiple_1", &gen_ctrl_S821__backbone_upsample_1_Resize_multiple_1, 20, 20, 40, 40, 32, SIGNED_INOUT, KOP_NEAREST_NEIGHBOR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S822__backbone_Concat_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S822__backbone_Concat_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S822__backbone_Concat_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S822__backbone_Concat_1_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S822__backbone_Concat_1_multiple_1", &gen_ctrl_S822__backbone_Concat_1_multiple_1, 2, 1600, 2, KOP_CONCAT, 32,32);
    
    CNN_GenControl_T gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_51");
    // generator for _backbone_C3_p3_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 32, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1", &gen_ctrl_S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1, 2, 1600, 2, KOP_SPLIT, 16,16);
    
    CNN_GenControl_T gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_52");
    // generator for _backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_53");
    // generator for _backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_54");
    // generator for _backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_55");
    // generator for _backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_56");
    // generator for _backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_57");
    // generator for _backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_58");
    // generator for _backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_59");
    // generator for _backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_60");
    // generator for _backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 16, 16, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S834__backbone_C3_p3_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S834__backbone_C3_p3_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S834__backbone_C3_p3_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S834__backbone_C3_p3_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S834__backbone_C3_p3_Concat_multiple_1", &gen_ctrl_S834__backbone_C3_p3_Concat_multiple_1, 2, 1600, 2, KOP_CONCAT, 16,16);
    
    CNN_GenControl_T gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_62");
    // generator for _backbone_C3_p3_conv3_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1", &gen_ctrl_S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_72");
    // generator for _backbone_bu_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_73");
    // generator for _backbone_bu_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_63");
    // generator for _head_stems_0_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S838__head_stems_0_conv_Conv_fusion_multiple_1", &gen_ctrl_S838__head_stems_0_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_64");
    // generator for _head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_65");
    // generator for _head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_66");
    // generator for _head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1", &gen_ctrl_S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_67
    s843_multiple_1_kernel_gen("S843_expr_67_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_preds_0_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S844__head_cls_preds_0_Conv_fusion_multiple_1", &gen_ctrl_S844__head_cls_preds_0_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 80, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_68");
    // generator for _head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_69");
    // generator for _head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_70");
    // generator for _head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_71");
    // generator for _head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_reg_preds_0_Conv
    CNN_ConvolutionPoolAct_fp16("S849__head_reg_preds_0_Conv_multiple_1", &gen_ctrl_S849__head_reg_preds_0_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 4, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_obj_preds_0_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S850__head_obj_preds_0_Conv_fusion_multiple_1", &gen_ctrl_S850__head_obj_preds_0_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 1, 40, 40,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S851__head_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S851__head_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S851__head_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S851__head_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S851__head_Concat_multiple_1", &gen_ctrl_S851__head_Concat_multiple_1, 2, 1600, 3, KOP_CONCAT, 4,1,80);
    
    CNN_GenControl_T gen_ctrl_S853__backbone_Concat_2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S853__backbone_Concat_2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S853__backbone_Concat_2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S853__backbone_Concat_2_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S853__backbone_Concat_2_multiple_1", &gen_ctrl_S853__backbone_Concat_2_multiple_1, 2, 400, 2, KOP_CONCAT, 32,32);
    
    CNN_GenControl_T gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_74");
    // generator for _backbone_C3_n3_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1", &gen_ctrl_S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1, 2, 400, 2, KOP_SPLIT, 32,32);
    
    CNN_GenControl_T gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_75");
    // generator for _backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_76");
    // generator for _backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_77");
    // generator for _backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_78");
    // generator for _backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_79");
    // generator for _backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_80");
    // generator for _backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_81");
    // generator for _backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_82");
    // generator for _backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_83");
    // generator for _backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 32, 32, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S865__backbone_C3_n3_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S865__backbone_C3_n3_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S865__backbone_C3_n3_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S865__backbone_C3_n3_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S865__backbone_C3_n3_Concat_multiple_1", &gen_ctrl_S865__backbone_C3_n3_Concat_multiple_1, 2, 400, 2, KOP_CONCAT, 32,32);
    
    CNN_GenControl_T gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_85");
    // generator for _backbone_C3_n3_conv3_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1", &gen_ctrl_S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_95");
    // generator for _backbone_bu_conv1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_96");
    // generator for _backbone_bu_conv1_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_86");
    // generator for _head_stems_1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S869__head_stems_1_conv_Conv_fusion_multiple_1", &gen_ctrl_S869__head_stems_1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_87");
    // generator for _head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_88");
    // generator for _head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_89");
    // generator for _head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1", &gen_ctrl_S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_90
    s874_multiple_1_kernel_gen("S874_expr_90_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_preds_1_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S875__head_cls_preds_1_Conv_fusion_multiple_1", &gen_ctrl_S875__head_cls_preds_1_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 80, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_91");
    // generator for _head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_92");
    // generator for _head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_93");
    // generator for _head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_94");
    // generator for _head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_reg_preds_1_Conv
    CNN_ConvolutionPoolAct_fp16("S880__head_reg_preds_1_Conv_multiple_1", &gen_ctrl_S880__head_reg_preds_1_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 4, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_obj_preds_1_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S881__head_obj_preds_1_Conv_fusion_multiple_1", &gen_ctrl_S881__head_obj_preds_1_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 1, 20, 20,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S882__head_Concat_1_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S882__head_Concat_1_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S882__head_Concat_1_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S882__head_Concat_1_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S882__head_Concat_1_multiple_1", &gen_ctrl_S882__head_Concat_1_multiple_1, 2, 400, 3, KOP_CONCAT, 4,1,80);
    
    CNN_GenControl_T gen_ctrl_S884__backbone_Concat_3_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S884__backbone_Concat_3_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S884__backbone_Concat_3_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S884__backbone_Concat_3_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S884__backbone_Concat_3_multiple_1", &gen_ctrl_S884__backbone_Concat_3_multiple_1, 2, 100, 2, KOP_CONCAT, 64,64);
    
    CNN_GenControl_T gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_107");
    // generator for _backbone_C3_n4_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1", &gen_ctrl_S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1, 2, 100, 2, KOP_SPLIT, 64,64);
    
    CNN_GenControl_T gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_98");
    // generator for _backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_99");
    // generator for _backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_100");
    // generator for _backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_101");
    // generator for _backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_102");
    // generator for _backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_103");
    // generator for _backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_104");
    // generator for _backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1", &gen_ctrl_S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_105");
    // generator for _backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_106");
    // generator for _backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S896__backbone_C3_n4_Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S896__backbone_C3_n4_Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S896__backbone_C3_n4_Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S896__backbone_C3_n4_Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S896__backbone_C3_n4_Concat_multiple_1", &gen_ctrl_S896__backbone_C3_n4_Concat_multiple_1, 2, 100, 2, KOP_CONCAT, 64,64);
    
    CNN_GenControl_T gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_108");
    // generator for _backbone_C3_n4_conv3_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1", &gen_ctrl_S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 128, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_109");
    // generator for _head_stems_2_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S898__head_stems_2_conv_Conv_fusion_multiple_1", &gen_ctrl_S898__head_stems_2_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 128, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_110");
    // generator for _head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_111");
    // generator for _head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_112");
    // generator for _head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv
    CNN_ConvolutionPoolAct_fp16("S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1", &gen_ctrl_S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    
    // generator for expr_113
    s903_multiple_1_kernel_gen("S903_expr_113_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_cls_preds_2_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S904__head_cls_preds_2_Conv_fusion_multiple_1", &gen_ctrl_S904__head_cls_preds_2_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 80, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_114");
    // generator for _head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_115");
    // generator for _head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_116");
    // generator for _head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1, "CUSTOM_ACTIVATION_NAME", "expr_117");
    // generator for _head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1", &gen_ctrl_S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 64, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_CUSTOM);
    
    CNN_GenControl_T gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_reg_preds_2_Conv
    CNN_ConvolutionPoolAct_fp16("S909__head_reg_preds_2_Conv_multiple_1", &gen_ctrl_S909__head_reg_preds_2_Conv_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 4, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1, "HWC", AT_OPT_VAL(1));
    CNN_SetGenCtrl(&gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for _head_obj_preds_2_Conv_fusion
    CNN_ConvolutionPoolAct_fp16("S910__head_obj_preds_2_Conv_fusion_multiple_1", &gen_ctrl_S910__head_obj_preds_2_Conv_fusion_multiple_1,
                                /* InFeat, OutFeat, InW, InH */
                                 64, 1, 10, 10,
                                /* Op1, Fcx, Fcy, Dcx, Dcy, Scx, Scy, Pad */
                                 KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                                /* Op2, Fpx, Fpy, Dpx, Dpy, Spx, Spy, Pad */
                                 KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                                 KOP_SIGMOID);
    
    CNN_GenControl_T gen_ctrl_S911__head_Concat_2_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S911__head_Concat_2_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S911__head_Concat_2_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S911__head_Concat_2_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S911__head_Concat_2_multiple_1", &gen_ctrl_S911__head_Concat_2_multiple_1, 2, 100, 3, KOP_CONCAT, 4,1,80);
    
    CNN_GenControl_T gen_ctrl_S914__Slice_split_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S914__Slice_split_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S914__Slice_split_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S914__Slice_split_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_SplitLastAxis_Generator2("S914__Slice_split_multiple_1", &gen_ctrl_S914__Slice_split_multiple_1, 2, 2100, 3, KOP_SPLIT, 4,1,80);
    
    
    // generator for expr_118
    s915_multiple_1_kernel_gen("S915_expr_118_multiple_1");
    
    CNN_GenControl_T gen_ctrl_S916__Concat_multiple_1;
    CNN_InitGenCtrl(&gen_ctrl_S916__Concat_multiple_1);
    CNN_SetGenCtrl(&gen_ctrl_S916__Concat_multiple_1, "ARG_DTYPE", AT_OPT_VAL(ARG_IEEE16));
    CNN_SetGenCtrl(&gen_ctrl_S916__Concat_multiple_1, "FLOAT_DUMP", AT_OPT_VAL(1));
    
    CNN_ConcatLastAxis_Generator2("S916__Concat_multiple_1", &gen_ctrl_S916__Concat_multiple_1, 2, 2100, 2, KOP_CONCAT, 4,80);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("onnx_graphCNN",
        /* Arguments either passed or globals */
            CArgs(416,
                TCArgInfo("f16 * __restrict__", "Input_1", ARG_SCOPE_ARG_ALLOC, ARG_DIR_IN, AT_MEM_L3_DEFAULTRAM, AT_MEM_L3_DEFAULTRAM, 0),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stem_in_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stem_in_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1586", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1586.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stem_res0_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stem_res0_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1589", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1589.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stem_res0_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stem_res0_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1592", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1592.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1595", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1595.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1598", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1598.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1601", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1601.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1604", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1604.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1607", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1607.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1610", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1610.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1613", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1613.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1616", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1616.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1619", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1619.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1622", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1622.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1625", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1625.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1628", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1628.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_0_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_0_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_0_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_0_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1631", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1631.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_1_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_1_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1634", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1634.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_1_local_module_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_1_local_module_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_1_local_module_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_1_local_module_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1637", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1637.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_1_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_1_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1640", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1640.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_2_local_module_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_2_local_module_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_2_blocks_2_local_module_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_2_blocks_2_local_module_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1643", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1643.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_0_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_0_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_0_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_0_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1646", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1646.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_1_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_1_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1649", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1649.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_1_local_module_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_1_local_module_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_1_local_module_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_1_local_module_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1652", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1652.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_1_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_1_biases.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1655", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1655.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_2_local_module_main_inverted_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_2_local_module_main_inverted_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_backbone_backbone_stages_3_blocks_2_local_module_main_depth_conv_conv_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_backbone_backbone_stages_3_blocks_2_local_module_main_depth_conv_conv_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1658", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1658.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_lateral_conv0_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_lateral_conv0_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1661", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1661.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1664", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1664.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_0_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_0_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1670", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1670.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_0_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_0_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1673", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1673.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_0_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_0_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1676", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1676.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_1_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_1_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1679", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1679.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_1_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_1_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1682", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1682.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_1_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_1_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1685", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1685.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_2_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_2_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1688", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1688.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_2_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_2_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1691", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1691.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_m_m_2_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_m_m_2_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1694", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1694.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p4_conv3_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p4_conv3_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1697", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1697.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_reduce_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_reduce_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1700", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1700.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1703", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1703.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_0_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_0_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1709", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1709.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_0_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_0_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1712", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1712.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_0_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_0_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1715", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1715.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_1_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_1_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1718", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1718.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_1_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_1_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1721", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1721.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_1_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_1_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1724", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1724.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_2_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_2_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1727", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1727.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_2_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_2_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1730", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1730.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_m_m_2_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_m_m_2_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1733", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1733.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_p3_conv3_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_p3_conv3_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1736", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1736.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_bu_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_bu_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1739", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1739.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_bu_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_bu_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1742", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1742.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1745", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1745.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_0_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_0_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1751", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1751.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_0_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_0_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1754", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1754.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_0_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_0_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1757", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1757.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_1_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_1_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1760", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1760.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_1_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_1_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1763", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1763.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_1_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_1_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1766", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1766.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_2_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_2_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1769", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1769.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_2_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_2_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1772", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1772.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_m_m_2_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_m_m_2_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1775", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1775.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n3_conv3_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n3_conv3_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1778", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1778.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_bu_conv1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_bu_conv1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1781", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1781.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_bu_conv1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_bu_conv1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1784", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1784.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1787", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1787.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_0_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_0_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1793", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1793.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_0_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_0_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1796", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1796.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_0_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_0_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1799", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1799.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_1_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_1_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1802", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1802.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_1_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_1_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1805", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1805.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_1_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_1_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1808", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1808.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_2_conv1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_2_conv1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1811", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1811.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_2_conv2_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_2_conv2_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1814", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1814.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_m_m_2_conv2_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_m_m_2_conv2_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1817", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1817.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_c3_n4_conv3_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_c3_n4_conv3_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1820", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1820.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_stems_0_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_stems_0_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1823", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1823.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_0_cls_convs_0_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_0_cls_convs_0_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1826", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1826.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_0_cls_convs_0_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_0_cls_convs_0_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1829", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1829.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_0_cls_convs_0_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_0_cls_convs_0_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1832", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1832.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_0_cls_convs_0_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_0_cls_convs_0_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1835", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1835.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_preds_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_preds_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_cls_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_0_reg_convs_0_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_0_reg_convs_0_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1838", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1838.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_0_reg_convs_0_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_0_reg_convs_0_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1841", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1841.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_0_reg_convs_0_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_0_reg_convs_0_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1844", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1844.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_0_reg_convs_0_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_0_reg_convs_0_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1847", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1847.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_preds_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_preds_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_reg_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_obj_preds_0_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_obj_preds_0_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_obj_preds_0_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_0_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_stems_1_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_stems_1_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1850", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1850.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_1_cls_convs_1_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_1_cls_convs_1_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1853", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1853.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_1_cls_convs_1_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_1_cls_convs_1_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1856", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1856.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_1_cls_convs_1_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_1_cls_convs_1_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1859", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1859.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_1_cls_convs_1_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_1_cls_convs_1_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1862", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1862.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_preds_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_preds_1_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_cls_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_1_reg_convs_1_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_1_reg_convs_1_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1865", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1865.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_1_reg_convs_1_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_1_reg_convs_1_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1868", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1868.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_1_reg_convs_1_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_1_reg_convs_1_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1871", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1871.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_1_reg_convs_1_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_1_reg_convs_1_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1874", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1874.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_preds_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_preds_1_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_reg_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_obj_preds_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_obj_preds_1_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_obj_preds_1_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_1_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_stems_2_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_stems_2_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1877", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1877.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_2_cls_convs_2_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_2_cls_convs_2_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1880", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1880.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_2_cls_convs_2_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_2_cls_convs_2_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1883", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1883.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_2_cls_convs_2_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_2_cls_convs_2_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1886", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1886.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_convs_2_cls_convs_2_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_convs_2_cls_convs_2_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1889", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1889.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_cls_preds_2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_cls_preds_2_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_cls_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_cls_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_2_reg_convs_2_0_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_2_reg_convs_2_0_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1892", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1892.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_2_reg_convs_2_0_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_2_reg_convs_2_0_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1895", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1895.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_2_reg_convs_2_1_dconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_2_reg_convs_2_1_dconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1898", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1898.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_convs_2_reg_convs_2_1_pconv_conv_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_convs_2_reg_convs_2_1_pconv_conv_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_onnx__conv_1901", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_onnx__conv_1901.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_reg_preds_2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_reg_preds_2_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_reg_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_reg_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_head_obj_preds_2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_head_obj_preds_2_conv_weights.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Constant_head_obj_preds_2_bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("Constant_head_obj_preds_2_bias.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_DEFAULTFLASH, AT_MEM_UNDEF, ConstInfo("_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23.tensor", 1, 1, 16, 0)),
                TCArgInfo("f16 * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L3_DEFAULTRAM, AT_MEM_L3_DEFAULTRAM, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(378,
            TCArgInfo("f16 * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S35_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S38_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S41_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S44_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S47_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S48_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S51_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S57_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S61_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S66_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_5", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_10", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_8", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_11", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_4", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_9", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_6", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_7", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S67_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S300_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S305_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S310_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S315_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S320_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S325_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S330_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S335_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S340_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S345_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S350_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S355_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S358_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S360_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S361_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S361_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S361_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S362_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S364_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S366_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S367_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S369_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S370_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S371_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S372_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S374_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S375_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S377_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S378_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S380_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S382_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S383_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S384_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S386_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S388_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S391_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_9", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_11", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_10", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_4", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_7", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_6", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_5", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_8", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S392_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S397_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S402_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S407_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S412_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S417_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S422_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S427_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S432_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S437_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S442_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S447_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S452_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S455_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S457_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S458_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S458_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S458_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S459_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S460_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S461_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S462_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S463_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S464_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S465_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S466_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S468_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S469_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S471_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S472_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S474_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S476_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S477_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S478_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S480_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S482_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S483_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S484_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S487_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S490_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_16", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_6", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_17", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_22", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_13", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_23", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_9", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_15", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_8", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_4", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_18", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_21", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_11", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_10", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_19", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_14", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_7", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_12", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_20", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S491_Output_5", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S496_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S501_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S506_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S511_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S516_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S521_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S526_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S531_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S536_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S541_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S546_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S551_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S556_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S561_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S566_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S571_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S576_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S581_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S586_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S591_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S596_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S601_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S606_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S611_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S614_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S616_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S617_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S617_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S617_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S618_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S619_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S620_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S621_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S622_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S623_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S624_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S625_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S627_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S628_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S630_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S631_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S633_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S635_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S636_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S637_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S639_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S641_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S644_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_22", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_20", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_23", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_18", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_21", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_16", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_19", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_14", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_17", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_12", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_15", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_10", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_13", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_8", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_6", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_11", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_9", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_4", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_7", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_5", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S645_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S650_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S655_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S660_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S665_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S670_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S675_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S680_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S685_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S690_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S695_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S700_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S705_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S710_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S715_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S720_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S725_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S730_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S735_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S740_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S745_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S750_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S755_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S760_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S765_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S768_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S770_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S771_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S771_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S771_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S772_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S773_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S774_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S775_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S776_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S777_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S778_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S779_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S781_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S782_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S784_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S785_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S787_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S789_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S790_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S791_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S793_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S795_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S796_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S798_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S800_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S801_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S801_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S803_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S805_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S806_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S807_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S808_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S809_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S810_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S811_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S812_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S813_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S816_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S817_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S819_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S820_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S821_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S822_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S823_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S824_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S824_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S825_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S826_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S827_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S828_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S829_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S830_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S831_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S832_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S833_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S834_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S835_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S836_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S837_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S838_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S839_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S840_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S841_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S842_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S843_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S844_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S845_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S846_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S847_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S848_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S849_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S850_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S853_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S854_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S855_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S855_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S856_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S857_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S858_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S859_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S860_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S861_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S862_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S863_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S864_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S865_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S866_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S867_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S868_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S869_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S870_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S871_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S872_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S873_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S874_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S875_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S876_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S877_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S878_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S879_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S880_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S881_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S884_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S885_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S886_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S886_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S887_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S888_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S889_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S890_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S891_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S892_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S893_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S894_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S895_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S896_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S897_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S898_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S899_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S900_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S901_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S902_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S903_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S904_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S905_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S906_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S907_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S908_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S909_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S910_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S913_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S914_Output_0", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S914_Output_2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S914_Output_1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("f16 * __restrict__", "S915_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    // Stacked tensors for concats and splits
    AddStackedTensors(
        "S358_Output",
        13, "S62_Output", "S301_Output", "S306_Output", "S311_Output", "S316_Output", "S321_Output", "S326_Output", "S331_Output", "S336_Output", "S341_Output", "S346_Output", "S351_Output", "S356_Output"
    );
    AddStackedTensors(
        "S455_Output",
        13, "S393_Output", "S398_Output", "S403_Output", "S408_Output", "S413_Output", "S418_Output", "S423_Output", "S428_Output", "S433_Output", "S438_Output", "S443_Output", "S448_Output", "S453_Output"
    );
    AddStackedTensors(
        "S614_Output",
        25, "S492_Output", "S497_Output", "S502_Output", "S507_Output", "S512_Output", "S517_Output", "S522_Output", "S527_Output", "S532_Output", "S537_Output", "S542_Output", "S547_Output", "S552_Output", "S557_Output", "S562_Output", "S567_Output", "S572_Output", "S577_Output", "S582_Output", "S587_Output", "S592_Output", "S597_Output", "S602_Output", "S607_Output", "S612_Output"
    );
    AddStackedTensors(
        "S768_Output",
        25, "S646_Output", "S651_Output", "S656_Output", "S661_Output", "S666_Output", "S671_Output", "S676_Output", "S681_Output", "S686_Output", "S691_Output", "S696_Output", "S701_Output", "S706_Output", "S711_Output", "S716_Output", "S721_Output", "S726_Output", "S731_Output", "S736_Output", "S741_Output", "S746_Output", "S751_Output", "S756_Output", "S761_Output", "S766_Output"
    );
    AddStackedTensors(
        "S913_Output",
        3, "S851_Output", "S882_Output", "S911_Output"
    );
    AddStackedTensors(
        "S372_Output",
        2, "S373_Output_0", "S373_Output_1"
    );
    AddStackedTensors(
        "S466_Output",
        2, "S467_Output_0", "S467_Output_1"
    );
    AddStackedTensors(
        "S625_Output",
        2, "S626_Output_0", "S626_Output_1"
    );
    AddStackedTensors(
        "S779_Output",
        2, "S780_Output_0", "S780_Output_1"
    );

    // Node S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1
    AddNode("S3__backbone_backbone_stem_in_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stem_in_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1586", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0)
        )
    );
    // Node S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S6__backbone_backbone_stem_res0_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stem_res0_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1589", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0)
        )
    );
    // Node S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1
    AddNode("S9__backbone_backbone_stem_res0_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stem_res0_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1592", 0),
            GNodeArg(GNA_OUT, "S9_Output", 0)
        )
    );
    // Node expr_1 in_qs [f16,f16] out_qs [f16]
    AddNode("S10_expr_1_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S9_Output", 0),
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0)
        )
    );
    // Node S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S13__backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_0_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1595", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0)
        )
    );
    // Node S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S16__backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_0_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1598", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0)
        )
    );
    // Node S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1
    AddNode("S19__backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S16_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_0_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1601", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0)
        )
    );
    // Node S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S22__backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_1_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1604", 0),
            GNodeArg(GNA_OUT, "S22_Output", 0)
        )
    );
    // Node S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S25__backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S22_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_1_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1607", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0)
        )
    );
    // Node S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1
    AddNode("S28__backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_0_blocks_blocks_1_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1610", 0),
            GNodeArg(GNA_OUT, "S28_Output", 0)
        )
    );
    // Node expr_7 in_qs [f16,f16] out_qs [f16]
    AddNode("S29_expr_7_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_OUT, "S29_Output", 0)
        )
    );
    // Node S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S32__backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S29_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_0_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1613", 0),
            GNodeArg(GNA_OUT, "S32_Output", 0)
        )
    );
    // Node S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S35__backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S32_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_0_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1616", 0),
            GNodeArg(GNA_OUT, "S35_Output", 0)
        )
    );
    // Node S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1
    AddNode("S38__backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_0_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1619", 0),
            GNodeArg(GNA_OUT, "S38_Output", 0)
        )
    );
    // Node S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S41__backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_1_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1622", 0),
            GNodeArg(GNA_OUT, "S41_Output", 0)
        )
    );
    // Node S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S44__backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S41_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_1_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1625", 0),
            GNodeArg(GNA_OUT, "S44_Output", 0)
        )
    );
    // Node S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1
    AddNode("S47__backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S44_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_1_blocks_blocks_1_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1628", 0),
            GNodeArg(GNA_OUT, "S47_Output", 0)
        )
    );
    // Node expr_12 in_qs [f16,f16] out_qs [f16]
    AddNode("S48_expr_12_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S47_Output", 0),
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_OUT, "S48_Output", 0)
        )
    );
    // Node S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S51__backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S48_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_0_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_0_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S51_Output", 0)
        )
    );
    // Node S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S54__backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S51_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_0_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_0_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S54_Output", 0)
        )
    );
    // Node S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1
    AddNode("S57__backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_0_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1631", 0),
            GNodeArg(GNA_OUT, "S57_Output", 0)
        )
    );
    // Node S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1
    AddNode("S61__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_qkv_conv_conv_biases", 0),
            GNodeArg(GNA_OUT, "S61_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans inq f16 outq f16
    AddNode("S62__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_OUT, "S62_Output", 0)
        )
    );
    // Node S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1
    AddNode("S66__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S61_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases", 0),
            GNodeArg(GNA_OUT, "S66_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc inq f16 outq f16
    AddNode("S67__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1",
        Bindings(13,
            GNodeArg(GNA_IN, "S66_Output", 0),
            GNodeArg(GNA_OUT, "S67_Output_0", 0),
            GNodeArg(GNA_OUT, "S67_Output_1", 0),
            GNodeArg(GNA_OUT, "S67_Output_2", 0),
            GNodeArg(GNA_OUT, "S67_Output_3", 0),
            GNodeArg(GNA_OUT, "S67_Output_4", 0),
            GNodeArg(GNA_OUT, "S67_Output_5", 0),
            GNodeArg(GNA_OUT, "S67_Output_6", 0),
            GNodeArg(GNA_OUT, "S67_Output_7", 0),
            GNodeArg(GNA_OUT, "S67_Output_8", 0),
            GNodeArg(GNA_OUT, "S67_Output_9", 0),
            GNodeArg(GNA_OUT, "S67_Output_10", 0),
            GNodeArg(GNA_OUT, "S67_Output_11", 0)
        )
    );
    // Node S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1
    AddNode("S300__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", 0),
            GNodeArg(GNA_OUT, "S300_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 inq f16 outq f16
    AddNode("S301__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S300_Output", 0),
            GNodeArg(GNA_OUT, "S301_Output", 0)
        )
    );
    // Node S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1
    AddNode("S305__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", 0),
            GNodeArg(GNA_OUT, "S305_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 inq f16 outq f16
    AddNode("S306__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S305_Output", 0),
            GNodeArg(GNA_OUT, "S306_Output", 0)
        )
    );
    // Node S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1
    AddNode("S310__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", 0),
            GNodeArg(GNA_OUT, "S310_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 inq f16 outq f16
    AddNode("S311__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S310_Output", 0),
            GNodeArg(GNA_OUT, "S311_Output", 0)
        )
    );
    // Node S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1
    AddNode("S315__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", 0),
            GNodeArg(GNA_OUT, "S315_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 inq f16 outq f16
    AddNode("S316__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S315_Output", 0),
            GNodeArg(GNA_OUT, "S316_Output", 0)
        )
    );
    // Node S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1
    AddNode("S320__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", 0),
            GNodeArg(GNA_OUT, "S320_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 inq f16 outq f16
    AddNode("S321__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S320_Output", 0),
            GNodeArg(GNA_OUT, "S321_Output", 0)
        )
    );
    // Node S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1
    AddNode("S325__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", 0),
            GNodeArg(GNA_OUT, "S325_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 inq f16 outq f16
    AddNode("S326__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S325_Output", 0),
            GNodeArg(GNA_OUT, "S326_Output", 0)
        )
    );
    // Node S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1
    AddNode("S330__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", 0),
            GNodeArg(GNA_OUT, "S330_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 inq f16 outq f16
    AddNode("S331__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S330_Output", 0),
            GNodeArg(GNA_OUT, "S331_Output", 0)
        )
    );
    // Node S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1
    AddNode("S335__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", 0),
            GNodeArg(GNA_OUT, "S335_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 inq f16 outq f16
    AddNode("S336__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S335_Output", 0),
            GNodeArg(GNA_OUT, "S336_Output", 0)
        )
    );
    // Node S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1
    AddNode("S340__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", 0),
            GNodeArg(GNA_OUT, "S340_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 inq f16 outq f16
    AddNode("S341__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S340_Output", 0),
            GNodeArg(GNA_OUT, "S341_Output", 0)
        )
    );
    // Node S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1
    AddNode("S345__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", 0),
            GNodeArg(GNA_OUT, "S345_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 inq f16 outq f16
    AddNode("S346__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S345_Output", 0),
            GNodeArg(GNA_OUT, "S346_Output", 0)
        )
    );
    // Node S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1
    AddNode("S350__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", 0),
            GNodeArg(GNA_OUT, "S350_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 inq f16 outq f16
    AddNode("S351__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S350_Output", 0),
            GNodeArg(GNA_OUT, "S351_Output", 0)
        )
    );
    // Node S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1
    AddNode("S355__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S67_Output_11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", 0),
            GNodeArg(GNA_OUT, "S355_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 inq f16 outq f16
    AddNode("S356__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S355_Output", 0),
            GNodeArg(GNA_OUT, "S356_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin inq f16 outq f16
    AddNode("S360__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S358_Output", 0),
            GNodeArg(GNA_OUT, "S360_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split inq f16 outq f16
    AddNode("S361__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_split_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S360_Output", 0),
            GNodeArg(GNA_OUT, "S361_Output_0", 0),
            GNodeArg(GNA_OUT, "S361_Output_1", 0),
            GNodeArg(GNA_OUT, "S361_Output_2", 0)
        )
    );
    // Node S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1
    AddNode("S362__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S361_Output_0", 0),
            GNodeArg(GNA_OUT, "S362_Output", 0)
        )
    );
    // Node S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1
    AddNode("S364__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S361_Output_1", 0),
            GNodeArg(GNA_OUT, "S364_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans inq f16 outq f16
    AddNode("S366__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S364_Output", 0),
            GNodeArg(GNA_OUT, "S366_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad inq f16 outq f16
    AddNode("S367__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Pad_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S361_Output_2", 0),
            GNodeArg(GNA_OUT, "S367_Output", 0)
        )
    );
    // Node S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S369__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S367_Output", 0),
            GNodeArg(GNA_IN, "S366_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_biases", 0),
            GNodeArg(GNA_OUT, "S369_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3 inq f16 outq f16
    AddNode("S370__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Transpose_3_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S369_Output", 0),
            GNodeArg(GNA_OUT, "S370_Output", 0)
        )
    );
    // Node S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S371__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_MatMul_1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S362_Output", 0),
            GNodeArg(GNA_IN, "S370_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_matmul_1_biases", 0),
            GNodeArg(GNA_OUT, "S371_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin inq f16 outq f16
    AddNode("S372__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S371_Output", 0),
            GNodeArg(GNA_OUT, "S372_Output", 0)
        )
    );
    // Node expr_15 in_qs [f16,f16] out_qs [f16]
    AddNode("S374_expr_15_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S373_Output_1", 0),
            GNodeArg(GNA_IN, "S373_Output_0", 0),
            GNodeArg(GNA_OUT, "S374_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans inq f16 outq f16
    AddNode("S375__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S374_Output", 0),
            GNodeArg(GNA_OUT, "S375_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0 inq f16 outq f16
    AddNode("S377__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S375_Output", 0),
            GNodeArg(GNA_OUT, "S377_Output", 0)
        )
    );
    // Node S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1
    AddNode("S378__backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S377_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_context_module_main_proj_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1634", 0),
            GNodeArg(GNA_OUT, "S378_Output", 0)
        )
    );
    // Node expr_16 in_qs [f16,f16] out_qs [f16]
    AddNode("S380_expr_16_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S378_Output", 0),
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_OUT, "S380_Output", 0)
        )
    );
    // Node S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S382__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S380_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_1_local_module_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S382_Output", 0)
        )
    );
    // Node S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S383__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S382_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_1_local_module_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S383_Output", 0)
        )
    );
    // Node S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1
    AddNode("S384__backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S383_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1637", 0),
            GNodeArg(GNA_OUT, "S384_Output", 0)
        )
    );
    // Node expr_17 in_qs [f16,f16] out_qs [f16]
    AddNode("S386_expr_17_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S384_Output", 0),
            GNodeArg(GNA_IN, "S380_Output", 0),
            GNodeArg(GNA_OUT, "S386_Output", 0)
        )
    );
    // Node S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1
    AddNode("S388__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S386_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_qkv_conv_conv_biases", 0),
            GNodeArg(GNA_OUT, "S388_Output", 0)
        )
    );
    // Node S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1
    AddNode("S391__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S388_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases", 0),
            GNodeArg(GNA_OUT, "S391_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc inq f16 outq f16
    AddNode("S392__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1",
        Bindings(13,
            GNodeArg(GNA_IN, "S391_Output", 0),
            GNodeArg(GNA_OUT, "S392_Output_0", 0),
            GNodeArg(GNA_OUT, "S392_Output_1", 0),
            GNodeArg(GNA_OUT, "S392_Output_2", 0),
            GNodeArg(GNA_OUT, "S392_Output_3", 0),
            GNodeArg(GNA_OUT, "S392_Output_4", 0),
            GNodeArg(GNA_OUT, "S392_Output_5", 0),
            GNodeArg(GNA_OUT, "S392_Output_6", 0),
            GNodeArg(GNA_OUT, "S392_Output_7", 0),
            GNodeArg(GNA_OUT, "S392_Output_8", 0),
            GNodeArg(GNA_OUT, "S392_Output_9", 0),
            GNodeArg(GNA_OUT, "S392_Output_10", 0),
            GNodeArg(GNA_OUT, "S392_Output_11", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans inq f16 outq f16
    AddNode("S393__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S388_Output", 0),
            GNodeArg(GNA_OUT, "S393_Output", 0)
        )
    );
    // Node S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1
    AddNode("S397__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", 0),
            GNodeArg(GNA_OUT, "S397_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 inq f16 outq f16
    AddNode("S398__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S397_Output", 0),
            GNodeArg(GNA_OUT, "S398_Output", 0)
        )
    );
    // Node S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1
    AddNode("S402__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", 0),
            GNodeArg(GNA_OUT, "S402_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 inq f16 outq f16
    AddNode("S403__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S402_Output", 0),
            GNodeArg(GNA_OUT, "S403_Output", 0)
        )
    );
    // Node S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1
    AddNode("S407__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", 0),
            GNodeArg(GNA_OUT, "S407_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 inq f16 outq f16
    AddNode("S408__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S407_Output", 0),
            GNodeArg(GNA_OUT, "S408_Output", 0)
        )
    );
    // Node S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1
    AddNode("S412__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", 0),
            GNodeArg(GNA_OUT, "S412_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 inq f16 outq f16
    AddNode("S413__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S412_Output", 0),
            GNodeArg(GNA_OUT, "S413_Output", 0)
        )
    );
    // Node S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1
    AddNode("S417__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", 0),
            GNodeArg(GNA_OUT, "S417_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 inq f16 outq f16
    AddNode("S418__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S417_Output", 0),
            GNodeArg(GNA_OUT, "S418_Output", 0)
        )
    );
    // Node S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1
    AddNode("S422__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", 0),
            GNodeArg(GNA_OUT, "S422_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 inq f16 outq f16
    AddNode("S423__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S422_Output", 0),
            GNodeArg(GNA_OUT, "S423_Output", 0)
        )
    );
    // Node S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1
    AddNode("S427__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", 0),
            GNodeArg(GNA_OUT, "S427_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 inq f16 outq f16
    AddNode("S428__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S427_Output", 0),
            GNodeArg(GNA_OUT, "S428_Output", 0)
        )
    );
    // Node S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1
    AddNode("S432__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", 0),
            GNodeArg(GNA_OUT, "S432_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 inq f16 outq f16
    AddNode("S433__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S432_Output", 0),
            GNodeArg(GNA_OUT, "S433_Output", 0)
        )
    );
    // Node S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1
    AddNode("S437__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", 0),
            GNodeArg(GNA_OUT, "S437_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 inq f16 outq f16
    AddNode("S438__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S437_Output", 0),
            GNodeArg(GNA_OUT, "S438_Output", 0)
        )
    );
    // Node S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1
    AddNode("S442__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", 0),
            GNodeArg(GNA_OUT, "S442_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 inq f16 outq f16
    AddNode("S443__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S442_Output", 0),
            GNodeArg(GNA_OUT, "S443_Output", 0)
        )
    );
    // Node S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1
    AddNode("S447__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", 0),
            GNodeArg(GNA_OUT, "S447_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 inq f16 outq f16
    AddNode("S448__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S447_Output", 0),
            GNodeArg(GNA_OUT, "S448_Output", 0)
        )
    );
    // Node S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1
    AddNode("S452__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S392_Output_11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", 0),
            GNodeArg(GNA_OUT, "S452_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 inq f16 outq f16
    AddNode("S453__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S452_Output", 0),
            GNodeArg(GNA_OUT, "S453_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin inq f16 outq f16
    AddNode("S457__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S455_Output", 0),
            GNodeArg(GNA_OUT, "S457_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split inq f16 outq f16
    AddNode("S458__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_split_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S457_Output", 0),
            GNodeArg(GNA_OUT, "S458_Output_0", 0),
            GNodeArg(GNA_OUT, "S458_Output_1", 0),
            GNodeArg(GNA_OUT, "S458_Output_2", 0)
        )
    );
    // Node S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1
    AddNode("S459__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S458_Output_0", 0),
            GNodeArg(GNA_OUT, "S459_Output", 0)
        )
    );
    // Node S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1
    AddNode("S460__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S458_Output_1", 0),
            GNodeArg(GNA_OUT, "S460_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans inq f16 outq f16
    AddNode("S461__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S460_Output", 0),
            GNodeArg(GNA_OUT, "S461_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad inq f16 outq f16
    AddNode("S462__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Pad_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S458_Output_2", 0),
            GNodeArg(GNA_OUT, "S462_Output", 0)
        )
    );
    // Node S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S463__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S462_Output", 0),
            GNodeArg(GNA_IN, "S461_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_biases", 0),
            GNodeArg(GNA_OUT, "S463_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3 inq f16 outq f16
    AddNode("S464__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Transpose_3_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S463_Output", 0),
            GNodeArg(GNA_OUT, "S464_Output", 0)
        )
    );
    // Node S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S465__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_MatMul_1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S459_Output", 0),
            GNodeArg(GNA_IN, "S464_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_matmul_1_biases", 0),
            GNodeArg(GNA_OUT, "S465_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin inq f16 outq f16
    AddNode("S466__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S465_Output", 0),
            GNodeArg(GNA_OUT, "S466_Output", 0)
        )
    );
    // Node expr_22 in_qs [f16,f16] out_qs [f16]
    AddNode("S468_expr_22_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S467_Output_1", 0),
            GNodeArg(GNA_IN, "S467_Output_0", 0),
            GNodeArg(GNA_OUT, "S468_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans inq f16 outq f16
    AddNode("S469__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S468_Output", 0),
            GNodeArg(GNA_OUT, "S469_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0 inq f16 outq f16
    AddNode("S471__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S469_Output", 0),
            GNodeArg(GNA_OUT, "S471_Output", 0)
        )
    );
    // Node S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1
    AddNode("S472__backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S471_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_context_module_main_proj_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1640", 0),
            GNodeArg(GNA_OUT, "S472_Output", 0)
        )
    );
    // Node expr_18 in_qs [f16,f16] out_qs [f16]
    AddNode("S474_expr_18_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S472_Output", 0),
            GNodeArg(GNA_IN, "S386_Output", 0),
            GNodeArg(GNA_OUT, "S474_Output", 0)
        )
    );
    // Node S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S476__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S474_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_2_local_module_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S476_Output", 0)
        )
    );
    // Node S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S477__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S476_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_2_blocks_2_local_module_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S477_Output", 0)
        )
    );
    // Node S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1
    AddNode("S478__backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S477_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_2_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1643", 0),
            GNodeArg(GNA_OUT, "S478_Output", 0)
        )
    );
    // Node expr_19 in_qs [f16,f16] out_qs [f16]
    AddNode("S480_expr_19_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S478_Output", 0),
            GNodeArg(GNA_IN, "S474_Output", 0),
            GNodeArg(GNA_OUT, "S480_Output", 0)
        )
    );
    // Node S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S482__backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S480_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_0_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_0_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S482_Output", 0)
        )
    );
    // Node S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S483__backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S482_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_0_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_0_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S483_Output", 0)
        )
    );
    // Node S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1
    AddNode("S484__backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S483_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_0_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1646", 0),
            GNodeArg(GNA_OUT, "S484_Output", 0)
        )
    );
    // Node S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1
    AddNode("S487__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S484_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_qkv_conv_conv_biases", 0),
            GNodeArg(GNA_OUT, "S487_Output", 0)
        )
    );
    // Node S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1
    AddNode("S490__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S487_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_0_conv_biases", 0),
            GNodeArg(GNA_OUT, "S490_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc inq f16 outq f16
    AddNode("S491__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1",
        Bindings(25,
            GNodeArg(GNA_IN, "S490_Output", 0),
            GNodeArg(GNA_OUT, "S491_Output_0", 0),
            GNodeArg(GNA_OUT, "S491_Output_1", 0),
            GNodeArg(GNA_OUT, "S491_Output_2", 0),
            GNodeArg(GNA_OUT, "S491_Output_3", 0),
            GNodeArg(GNA_OUT, "S491_Output_4", 0),
            GNodeArg(GNA_OUT, "S491_Output_5", 0),
            GNodeArg(GNA_OUT, "S491_Output_6", 0),
            GNodeArg(GNA_OUT, "S491_Output_7", 0),
            GNodeArg(GNA_OUT, "S491_Output_8", 0),
            GNodeArg(GNA_OUT, "S491_Output_9", 0),
            GNodeArg(GNA_OUT, "S491_Output_10", 0),
            GNodeArg(GNA_OUT, "S491_Output_11", 0),
            GNodeArg(GNA_OUT, "S491_Output_12", 0),
            GNodeArg(GNA_OUT, "S491_Output_13", 0),
            GNodeArg(GNA_OUT, "S491_Output_14", 0),
            GNodeArg(GNA_OUT, "S491_Output_15", 0),
            GNodeArg(GNA_OUT, "S491_Output_16", 0),
            GNodeArg(GNA_OUT, "S491_Output_17", 0),
            GNodeArg(GNA_OUT, "S491_Output_18", 0),
            GNodeArg(GNA_OUT, "S491_Output_19", 0),
            GNodeArg(GNA_OUT, "S491_Output_20", 0),
            GNodeArg(GNA_OUT, "S491_Output_21", 0),
            GNodeArg(GNA_OUT, "S491_Output_22", 0),
            GNodeArg(GNA_OUT, "S491_Output_23", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans inq f16 outq f16
    AddNode("S492__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Concat_flat0_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S487_Output", 0),
            GNodeArg(GNA_OUT, "S492_Output", 0)
        )
    );
    // Node S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1
    AddNode("S496__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", 0),
            GNodeArg(GNA_OUT, "S496_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 inq f16 outq f16
    AddNode("S497__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S496_Output", 0),
            GNodeArg(GNA_OUT, "S497_Output", 0)
        )
    );
    // Node S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1
    AddNode("S501__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", 0),
            GNodeArg(GNA_OUT, "S501_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 inq f16 outq f16
    AddNode("S502__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S501_Output", 0),
            GNodeArg(GNA_OUT, "S502_Output", 0)
        )
    );
    // Node S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1
    AddNode("S506__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", 0),
            GNodeArg(GNA_OUT, "S506_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 inq f16 outq f16
    AddNode("S507__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S506_Output", 0),
            GNodeArg(GNA_OUT, "S507_Output", 0)
        )
    );
    // Node S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1
    AddNode("S511__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", 0),
            GNodeArg(GNA_OUT, "S511_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 inq f16 outq f16
    AddNode("S512__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S511_Output", 0),
            GNodeArg(GNA_OUT, "S512_Output", 0)
        )
    );
    // Node S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1
    AddNode("S516__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", 0),
            GNodeArg(GNA_OUT, "S516_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 inq f16 outq f16
    AddNode("S517__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S516_Output", 0),
            GNodeArg(GNA_OUT, "S517_Output", 0)
        )
    );
    // Node S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1
    AddNode("S521__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", 0),
            GNodeArg(GNA_OUT, "S521_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 inq f16 outq f16
    AddNode("S522__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S521_Output", 0),
            GNodeArg(GNA_OUT, "S522_Output", 0)
        )
    );
    // Node S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1
    AddNode("S526__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", 0),
            GNodeArg(GNA_OUT, "S526_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 inq f16 outq f16
    AddNode("S527__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S526_Output", 0),
            GNodeArg(GNA_OUT, "S527_Output", 0)
        )
    );
    // Node S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1
    AddNode("S531__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", 0),
            GNodeArg(GNA_OUT, "S531_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 inq f16 outq f16
    AddNode("S532__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S531_Output", 0),
            GNodeArg(GNA_OUT, "S532_Output", 0)
        )
    );
    // Node S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1
    AddNode("S536__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", 0),
            GNodeArg(GNA_OUT, "S536_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 inq f16 outq f16
    AddNode("S537__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S536_Output", 0),
            GNodeArg(GNA_OUT, "S537_Output", 0)
        )
    );
    // Node S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1
    AddNode("S541__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", 0),
            GNodeArg(GNA_OUT, "S541_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 inq f16 outq f16
    AddNode("S542__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S541_Output", 0),
            GNodeArg(GNA_OUT, "S542_Output", 0)
        )
    );
    // Node S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1
    AddNode("S546__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", 0),
            GNodeArg(GNA_OUT, "S546_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 inq f16 outq f16
    AddNode("S547__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S546_Output", 0),
            GNodeArg(GNA_OUT, "S547_Output", 0)
        )
    );
    // Node S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1
    AddNode("S551__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", 0),
            GNodeArg(GNA_OUT, "S551_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 inq f16 outq f16
    AddNode("S552__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S551_Output", 0),
            GNodeArg(GNA_OUT, "S552_Output", 0)
        )
    );
    // Node S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1
    AddNode("S556__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_12", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12", 0),
            GNodeArg(GNA_OUT, "S556_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0 inq f16 outq f16
    AddNode("S557__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S556_Output", 0),
            GNodeArg(GNA_OUT, "S557_Output", 0)
        )
    );
    // Node S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1
    AddNode("S561__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_13", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13", 0),
            GNodeArg(GNA_OUT, "S561_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0 inq f16 outq f16
    AddNode("S562__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S561_Output", 0),
            GNodeArg(GNA_OUT, "S562_Output", 0)
        )
    );
    // Node S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1
    AddNode("S566__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_14", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14", 0),
            GNodeArg(GNA_OUT, "S566_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0 inq f16 outq f16
    AddNode("S567__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S566_Output", 0),
            GNodeArg(GNA_OUT, "S567_Output", 0)
        )
    );
    // Node S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1
    AddNode("S571__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_15", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15", 0),
            GNodeArg(GNA_OUT, "S571_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0 inq f16 outq f16
    AddNode("S572__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S571_Output", 0),
            GNodeArg(GNA_OUT, "S572_Output", 0)
        )
    );
    // Node S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1
    AddNode("S576__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_16", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16", 0),
            GNodeArg(GNA_OUT, "S576_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0 inq f16 outq f16
    AddNode("S577__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S576_Output", 0),
            GNodeArg(GNA_OUT, "S577_Output", 0)
        )
    );
    // Node S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1
    AddNode("S581__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_17", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17", 0),
            GNodeArg(GNA_OUT, "S581_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0 inq f16 outq f16
    AddNode("S582__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S581_Output", 0),
            GNodeArg(GNA_OUT, "S582_Output", 0)
        )
    );
    // Node S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1
    AddNode("S586__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_18", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18", 0),
            GNodeArg(GNA_OUT, "S586_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0 inq f16 outq f16
    AddNode("S587__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S586_Output", 0),
            GNodeArg(GNA_OUT, "S587_Output", 0)
        )
    );
    // Node S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1
    AddNode("S591__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_19", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19", 0),
            GNodeArg(GNA_OUT, "S591_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0 inq f16 outq f16
    AddNode("S592__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S591_Output", 0),
            GNodeArg(GNA_OUT, "S592_Output", 0)
        )
    );
    // Node S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1
    AddNode("S596__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_20", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20", 0),
            GNodeArg(GNA_OUT, "S596_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0 inq f16 outq f16
    AddNode("S597__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S596_Output", 0),
            GNodeArg(GNA_OUT, "S597_Output", 0)
        )
    );
    // Node S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1
    AddNode("S601__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_21", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21", 0),
            GNodeArg(GNA_OUT, "S601_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0 inq f16 outq f16
    AddNode("S602__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S601_Output", 0),
            GNodeArg(GNA_OUT, "S602_Output", 0)
        )
    );
    // Node S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1
    AddNode("S606__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_22", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22", 0),
            GNodeArg(GNA_OUT, "S606_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0 inq f16 outq f16
    AddNode("S607__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S606_Output", 0),
            GNodeArg(GNA_OUT, "S607_Output", 0)
        )
    );
    // Node S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1
    AddNode("S611__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S491_Output_23", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23", 0),
            GNodeArg(GNA_OUT, "S611_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0 inq f16 outq f16
    AddNode("S612__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S611_Output", 0),
            GNodeArg(GNA_OUT, "S612_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin inq f16 outq f16
    AddNode("S616__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S614_Output", 0),
            GNodeArg(GNA_OUT, "S616_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split inq f16 outq f16
    AddNode("S617__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_split_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S616_Output", 0),
            GNodeArg(GNA_OUT, "S617_Output_0", 0),
            GNodeArg(GNA_OUT, "S617_Output_1", 0),
            GNodeArg(GNA_OUT, "S617_Output_2", 0)
        )
    );
    // Node S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1
    AddNode("S618__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S617_Output_0", 0),
            GNodeArg(GNA_OUT, "S618_Output", 0)
        )
    );
    // Node S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1
    AddNode("S619__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_kernel_func_1_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S617_Output_1", 0),
            GNodeArg(GNA_OUT, "S619_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans inq f16 outq f16
    AddNode("S620__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S619_Output", 0),
            GNodeArg(GNA_OUT, "S620_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad inq f16 outq f16
    AddNode("S621__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Pad_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S617_Output_2", 0),
            GNodeArg(GNA_OUT, "S621_Output", 0)
        )
    );
    // Node S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S622__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S621_Output", 0),
            GNodeArg(GNA_IN, "S620_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_biases", 0),
            GNodeArg(GNA_OUT, "S622_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3 inq f16 outq f16
    AddNode("S623__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Transpose_3_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S622_Output", 0),
            GNodeArg(GNA_OUT, "S623_Output", 0)
        )
    );
    // Node S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S624__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_MatMul_1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S618_Output", 0),
            GNodeArg(GNA_IN, "S623_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_matmul_1_biases", 0),
            GNodeArg(GNA_OUT, "S624_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin inq f16 outq f16
    AddNode("S625__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Slice_4_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S624_Output", 0),
            GNodeArg(GNA_OUT, "S625_Output", 0)
        )
    );
    // Node expr_27 in_qs [f16,f16] out_qs [f16]
    AddNode("S627_expr_27_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S626_Output_1", 0),
            GNodeArg(GNA_IN, "S626_Output_0", 0),
            GNodeArg(GNA_OUT, "S627_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans inq f16 outq f16
    AddNode("S628__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_Reshape_3_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S627_Output", 0),
            GNodeArg(GNA_OUT, "S628_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0 inq f16 outq f16
    AddNode("S630__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_trans_in0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S628_Output", 0),
            GNodeArg(GNA_OUT, "S630_Output", 0)
        )
    );
    // Node S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1
    AddNode("S631__backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S630_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_context_module_main_proj_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1649", 0),
            GNodeArg(GNA_OUT, "S631_Output", 0)
        )
    );
    // Node expr_28 in_qs [f16,f16] out_qs [f16]
    AddNode("S633_expr_28_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S631_Output", 0),
            GNodeArg(GNA_IN, "S484_Output", 0),
            GNodeArg(GNA_OUT, "S633_Output", 0)
        )
    );
    // Node S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S635__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S633_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_1_local_module_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S635_Output", 0)
        )
    );
    // Node S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S636__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S635_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_1_local_module_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S636_Output", 0)
        )
    );
    // Node S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1
    AddNode("S637__backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S636_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_1_local_module_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1652", 0),
            GNodeArg(GNA_OUT, "S637_Output", 0)
        )
    );
    // Node expr_29 in_qs [f16,f16] out_qs [f16]
    AddNode("S639_expr_29_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S637_Output", 0),
            GNodeArg(GNA_IN, "S633_Output", 0),
            GNodeArg(GNA_OUT, "S639_Output", 0)
        )
    );
    // Node S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1
    AddNode("S641__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S639_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_qkv_conv_conv_biases", 0),
            GNodeArg(GNA_OUT, "S641_Output", 0)
        )
    );
    // Node S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1
    AddNode("S644__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S641_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_weights", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_0_conv_biases", 0),
            GNodeArg(GNA_OUT, "S644_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc inq f16 outq f16
    AddNode("S645__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_split_inc_multiple_1",
        Bindings(25,
            GNodeArg(GNA_IN, "S644_Output", 0),
            GNodeArg(GNA_OUT, "S645_Output_0", 0),
            GNodeArg(GNA_OUT, "S645_Output_1", 0),
            GNodeArg(GNA_OUT, "S645_Output_2", 0),
            GNodeArg(GNA_OUT, "S645_Output_3", 0),
            GNodeArg(GNA_OUT, "S645_Output_4", 0),
            GNodeArg(GNA_OUT, "S645_Output_5", 0),
            GNodeArg(GNA_OUT, "S645_Output_6", 0),
            GNodeArg(GNA_OUT, "S645_Output_7", 0),
            GNodeArg(GNA_OUT, "S645_Output_8", 0),
            GNodeArg(GNA_OUT, "S645_Output_9", 0),
            GNodeArg(GNA_OUT, "S645_Output_10", 0),
            GNodeArg(GNA_OUT, "S645_Output_11", 0),
            GNodeArg(GNA_OUT, "S645_Output_12", 0),
            GNodeArg(GNA_OUT, "S645_Output_13", 0),
            GNodeArg(GNA_OUT, "S645_Output_14", 0),
            GNodeArg(GNA_OUT, "S645_Output_15", 0),
            GNodeArg(GNA_OUT, "S645_Output_16", 0),
            GNodeArg(GNA_OUT, "S645_Output_17", 0),
            GNodeArg(GNA_OUT, "S645_Output_18", 0),
            GNodeArg(GNA_OUT, "S645_Output_19", 0),
            GNodeArg(GNA_OUT, "S645_Output_20", 0),
            GNodeArg(GNA_OUT, "S645_Output_21", 0),
            GNodeArg(GNA_OUT, "S645_Output_22", 0),
            GNodeArg(GNA_OUT, "S645_Output_23", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans inq f16 outq f16
    AddNode("S646__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Concat_flat0_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S641_Output", 0),
            GNodeArg(GNA_OUT, "S646_Output", 0)
        )
    );
    // Node S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1
    AddNode("S650__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp0", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp0", 0),
            GNodeArg(GNA_OUT, "S650_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0 inq f16 outq f16
    AddNode("S651__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp0_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S650_Output", 0),
            GNodeArg(GNA_OUT, "S651_Output", 0)
        )
    );
    // Node S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1
    AddNode("S655__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp1", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp1", 0),
            GNodeArg(GNA_OUT, "S655_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0 inq f16 outq f16
    AddNode("S656__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp1_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S655_Output", 0),
            GNodeArg(GNA_OUT, "S656_Output", 0)
        )
    );
    // Node S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1
    AddNode("S660__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp2", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp2", 0),
            GNodeArg(GNA_OUT, "S660_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0 inq f16 outq f16
    AddNode("S661__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp2_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S660_Output", 0),
            GNodeArg(GNA_OUT, "S661_Output", 0)
        )
    );
    // Node S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1
    AddNode("S665__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp3", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp3", 0),
            GNodeArg(GNA_OUT, "S665_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0 inq f16 outq f16
    AddNode("S666__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp3_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S665_Output", 0),
            GNodeArg(GNA_OUT, "S666_Output", 0)
        )
    );
    // Node S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1
    AddNode("S670__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp4", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp4", 0),
            GNodeArg(GNA_OUT, "S670_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0 inq f16 outq f16
    AddNode("S671__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp4_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S670_Output", 0),
            GNodeArg(GNA_OUT, "S671_Output", 0)
        )
    );
    // Node S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1
    AddNode("S675__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp5", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp5", 0),
            GNodeArg(GNA_OUT, "S675_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0 inq f16 outq f16
    AddNode("S676__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp5_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S675_Output", 0),
            GNodeArg(GNA_OUT, "S676_Output", 0)
        )
    );
    // Node S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1
    AddNode("S680__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp6", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp6", 0),
            GNodeArg(GNA_OUT, "S680_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0 inq f16 outq f16
    AddNode("S681__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp6_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S680_Output", 0),
            GNodeArg(GNA_OUT, "S681_Output", 0)
        )
    );
    // Node S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1
    AddNode("S685__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp7", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp7", 0),
            GNodeArg(GNA_OUT, "S685_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0 inq f16 outq f16
    AddNode("S686__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp7_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S685_Output", 0),
            GNodeArg(GNA_OUT, "S686_Output", 0)
        )
    );
    // Node S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1
    AddNode("S690__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp8", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp8", 0),
            GNodeArg(GNA_OUT, "S690_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0 inq f16 outq f16
    AddNode("S691__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp8_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S690_Output", 0),
            GNodeArg(GNA_OUT, "S691_Output", 0)
        )
    );
    // Node S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1
    AddNode("S695__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp9", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp9", 0),
            GNodeArg(GNA_OUT, "S695_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0 inq f16 outq f16
    AddNode("S696__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp9_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S695_Output", 0),
            GNodeArg(GNA_OUT, "S696_Output", 0)
        )
    );
    // Node S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1
    AddNode("S700__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp10", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp10", 0),
            GNodeArg(GNA_OUT, "S700_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0 inq f16 outq f16
    AddNode("S701__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp10_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S700_Output", 0),
            GNodeArg(GNA_OUT, "S701_Output", 0)
        )
    );
    // Node S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1
    AddNode("S705__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp11", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp11", 0),
            GNodeArg(GNA_OUT, "S705_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0 inq f16 outq f16
    AddNode("S706__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp11_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S705_Output", 0),
            GNodeArg(GNA_OUT, "S706_Output", 0)
        )
    );
    // Node S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1
    AddNode("S710__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_12", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp12", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp12", 0),
            GNodeArg(GNA_OUT, "S710_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0 inq f16 outq f16
    AddNode("S711__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp12_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S710_Output", 0),
            GNodeArg(GNA_OUT, "S711_Output", 0)
        )
    );
    // Node S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1
    AddNode("S715__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_13", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp13", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp13", 0),
            GNodeArg(GNA_OUT, "S715_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0 inq f16 outq f16
    AddNode("S716__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp13_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S715_Output", 0),
            GNodeArg(GNA_OUT, "S716_Output", 0)
        )
    );
    // Node S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1
    AddNode("S720__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_14", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp14", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp14", 0),
            GNodeArg(GNA_OUT, "S720_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0 inq f16 outq f16
    AddNode("S721__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp14_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S720_Output", 0),
            GNodeArg(GNA_OUT, "S721_Output", 0)
        )
    );
    // Node S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1
    AddNode("S725__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_15", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp15", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp15", 0),
            GNodeArg(GNA_OUT, "S725_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0 inq f16 outq f16
    AddNode("S726__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp15_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S725_Output", 0),
            GNodeArg(GNA_OUT, "S726_Output", 0)
        )
    );
    // Node S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1
    AddNode("S730__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_16", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp16", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp16", 0),
            GNodeArg(GNA_OUT, "S730_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0 inq f16 outq f16
    AddNode("S731__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp16_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S730_Output", 0),
            GNodeArg(GNA_OUT, "S731_Output", 0)
        )
    );
    // Node S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1
    AddNode("S735__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_17", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp17", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp17", 0),
            GNodeArg(GNA_OUT, "S735_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0 inq f16 outq f16
    AddNode("S736__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp17_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S735_Output", 0),
            GNodeArg(GNA_OUT, "S736_Output", 0)
        )
    );
    // Node S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1
    AddNode("S740__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_18", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp18", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp18", 0),
            GNodeArg(GNA_OUT, "S740_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0 inq f16 outq f16
    AddNode("S741__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp18_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S740_Output", 0),
            GNodeArg(GNA_OUT, "S741_Output", 0)
        )
    );
    // Node S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1
    AddNode("S745__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_19", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp19", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp19", 0),
            GNodeArg(GNA_OUT, "S745_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0 inq f16 outq f16
    AddNode("S746__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp19_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S745_Output", 0),
            GNodeArg(GNA_OUT, "S746_Output", 0)
        )
    );
    // Node S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1
    AddNode("S750__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_20", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp20", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp20", 0),
            GNodeArg(GNA_OUT, "S750_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0 inq f16 outq f16
    AddNode("S751__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp20_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S750_Output", 0),
            GNodeArg(GNA_OUT, "S751_Output", 0)
        )
    );
    // Node S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1
    AddNode("S755__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_21", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp21", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp21", 0),
            GNodeArg(GNA_OUT, "S755_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0 inq f16 outq f16
    AddNode("S756__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp21_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S755_Output", 0),
            GNodeArg(GNA_OUT, "S756_Output", 0)
        )
    );
    // Node S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1
    AddNode("S760__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_22", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp22", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp22", 0),
            GNodeArg(GNA_OUT, "S760_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0 inq f16 outq f16
    AddNode("S761__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp22_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S760_Output", 0),
            GNodeArg(GNA_OUT, "S761_Output", 0)
        )
    );
    // Node S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1
    AddNode("S765__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S645_Output_23", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_weights_grp23", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_conv_biases_grp23", 0),
            GNodeArg(GNA_OUT, "S765_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0 inq f16 outq f16
    AddNode("S766__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_aggreg_0_aggreg_0_1_Conv_grp23_trans_out0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S765_Output", 0),
            GNodeArg(GNA_OUT, "S766_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin inq f16 outq f16
    AddNode("S770__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S768_Output", 0),
            GNodeArg(GNA_OUT, "S770_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split inq f16 outq f16
    AddNode("S771__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_split_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S770_Output", 0),
            GNodeArg(GNA_OUT, "S771_Output_0", 0),
            GNodeArg(GNA_OUT, "S771_Output_1", 0),
            GNodeArg(GNA_OUT, "S771_Output_2", 0)
        )
    );
    // Node S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1
    AddNode("S772__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S771_Output_0", 0),
            GNodeArg(GNA_OUT, "S772_Output", 0)
        )
    );
    // Node S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1
    AddNode("S773__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_kernel_func_1_Relu_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S771_Output_1", 0),
            GNodeArg(GNA_OUT, "S773_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans inq f16 outq f16
    AddNode("S774__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S773_Output", 0),
            GNodeArg(GNA_OUT, "S774_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad inq f16 outq f16
    AddNode("S775__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Pad_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S771_Output_2", 0),
            GNodeArg(GNA_OUT, "S775_Output", 0)
        )
    );
    // Node S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S776__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S775_Output", 0),
            GNodeArg(GNA_IN, "S774_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_biases", 0),
            GNodeArg(GNA_OUT, "S776_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3 inq f16 outq f16
    AddNode("S777__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Transpose_3_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S776_Output", 0),
            GNodeArg(GNA_OUT, "S777_Output", 0)
        )
    );
    // Node S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1 inq1 f16 inq2 f16 outq f16 biasesq f16
    AddNode("S778__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_MatMul_1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S772_Output", 0),
            GNodeArg(GNA_IN, "S777_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_matmul_1_biases", 0),
            GNodeArg(GNA_OUT, "S778_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin inq f16 outq f16
    AddNode("S779__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Slice_4_tin_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S778_Output", 0),
            GNodeArg(GNA_OUT, "S779_Output", 0)
        )
    );
    // Node expr_34 in_qs [f16,f16] out_qs [f16]
    AddNode("S781_expr_34_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S780_Output_1", 0),
            GNodeArg(GNA_IN, "S780_Output_0", 0),
            GNodeArg(GNA_OUT, "S781_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans inq f16 outq f16
    AddNode("S782__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_Reshape_3_trans_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S781_Output", 0),
            GNodeArg(GNA_OUT, "S782_Output", 0)
        )
    );
    // Node _backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0 inq f16 outq f16
    AddNode("S784__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_trans_in0_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S782_Output", 0),
            GNodeArg(GNA_OUT, "S784_Output", 0)
        )
    );
    // Node S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1
    AddNode("S785__backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S784_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_context_module_main_proj_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1655", 0),
            GNodeArg(GNA_OUT, "S785_Output", 0)
        )
    );
    // Node expr_30 in_qs [f16,f16] out_qs [f16]
    AddNode("S787_expr_30_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S785_Output", 0),
            GNodeArg(GNA_IN, "S639_Output", 0),
            GNodeArg(GNA_OUT, "S787_Output", 0)
        )
    );
    // Node S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1
    AddNode("S789__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S787_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_inverted_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_2_local_module_main_inverted_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S789_Output", 0)
        )
    );
    // Node S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1
    AddNode("S790__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S789_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_depth_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_backbone_backbone_stages_3_blocks_2_local_module_main_depth_conv_conv_bias", 0),
            GNodeArg(GNA_OUT, "S790_Output", 0)
        )
    );
    // Node S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1
    AddNode("S791__backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S790_Output", 0),
            GNodeArg(GNA_IN, "_backbone_backbone_stages_3_blocks_blocks_2_local_module_main_point_conv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1658", 0),
            GNodeArg(GNA_OUT, "S791_Output", 0)
        )
    );
    // Node expr_31 in_qs [f16,f16] out_qs [f16]
    AddNode("S793_expr_31_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S791_Output", 0),
            GNodeArg(GNA_IN, "S787_Output", 0),
            GNodeArg(GNA_OUT, "S793_Output", 0)
        )
    );
    // Node S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1
    AddNode("S795__backbone_lateral_conv0_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S793_Output", 0),
            GNodeArg(GNA_IN, "_backbone_lateral_conv0_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1661", 0),
            GNodeArg(GNA_OUT, "S795_Output", 0)
        )
    );
    // Node _backbone_upsample_Resize inq f16 outq f16
    AddNode("S796__backbone_upsample_Resize_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S795_Output", 0),
            GNodeArg(GNA_OUT, "S796_Output", 0)
        )
    );
    // Node _backbone_Concat inq f16 outq f16
    AddNode("S798__backbone_Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S796_Output", 0),
            GNodeArg(GNA_IN, "S480_Output", 0),
            GNodeArg(GNA_OUT, "S798_Output", 0)
        )
    );
    // Node S800__backbone_C3_p4_conv1_conv_Conv_multiple_1
    AddNode("S800__backbone_C3_p4_conv1_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S798_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1664", 0),
            GNodeArg(GNA_OUT, "S800_Output", 0)
        )
    );
    // Node _backbone_C3_p4_conv1_conv_Conv_split inq f16 outq f16
    AddNode("S801__backbone_C3_p4_conv1_conv_Conv_split_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S800_Output", 0),
            GNodeArg(GNA_OUT, "S801_Output_0", 0),
            GNodeArg(GNA_OUT, "S801_Output_1", 0)
        )
    );
    // Node expr_38 in_qs [f16] out_qs [f16]
    AddNode("S803_expr_38_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S801_Output_0", 0),
            GNodeArg(GNA_OUT, "S803_Output", 0)
        )
    );
    // Node S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1
    AddNode("S805__backbone_C3_p4_m_m_0_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S803_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_0_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1670", 0),
            GNodeArg(GNA_OUT, "S805_Output", 0)
        )
    );
    // Node S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S806__backbone_C3_p4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S805_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_0_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1673", 0),
            GNodeArg(GNA_OUT, "S806_Output", 0)
        )
    );
    // Node S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S807__backbone_C3_p4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S806_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_0_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1676", 0),
            GNodeArg(GNA_OUT, "S807_Output", 0)
        )
    );
    // Node S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1
    AddNode("S808__backbone_C3_p4_m_m_1_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S807_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_1_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1679", 0),
            GNodeArg(GNA_OUT, "S808_Output", 0)
        )
    );
    // Node S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S809__backbone_C3_p4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S808_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_1_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1682", 0),
            GNodeArg(GNA_OUT, "S809_Output", 0)
        )
    );
    // Node S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S810__backbone_C3_p4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S809_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_1_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1685", 0),
            GNodeArg(GNA_OUT, "S810_Output", 0)
        )
    );
    // Node S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1
    AddNode("S811__backbone_C3_p4_m_m_2_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S810_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_2_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1688", 0),
            GNodeArg(GNA_OUT, "S811_Output", 0)
        )
    );
    // Node S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S812__backbone_C3_p4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S811_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_2_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1691", 0),
            GNodeArg(GNA_OUT, "S812_Output", 0)
        )
    );
    // Node S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S813__backbone_C3_p4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S812_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_m_m_2_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1694", 0),
            GNodeArg(GNA_OUT, "S813_Output", 0)
        )
    );
    // Node expr_48 in_qs [f16] out_qs [f16]
    AddNode("S816_expr_48_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S801_Output_1", 0),
            GNodeArg(GNA_OUT, "S816_Output", 0)
        )
    );
    // Node _backbone_C3_p4_Concat inq f16 outq f16
    AddNode("S817__backbone_C3_p4_Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S813_Output", 0),
            GNodeArg(GNA_IN, "S816_Output", 0),
            GNodeArg(GNA_OUT, "S817_Output", 0)
        )
    );
    // Node S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1
    AddNode("S819__backbone_C3_p4_conv3_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S817_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p4_conv3_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1697", 0),
            GNodeArg(GNA_OUT, "S819_Output", 0)
        )
    );
    // Node S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1
    AddNode("S820__backbone_reduce_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S819_Output", 0),
            GNodeArg(GNA_IN, "_backbone_reduce_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1700", 0),
            GNodeArg(GNA_OUT, "S820_Output", 0)
        )
    );
    // Node _backbone_upsample_1_Resize inq f16 outq f16
    AddNode("S821__backbone_upsample_1_Resize_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S820_Output", 0),
            GNodeArg(GNA_OUT, "S821_Output", 0)
        )
    );
    // Node _backbone_Concat_1 inq f16 outq f16
    AddNode("S822__backbone_Concat_1_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S821_Output", 0),
            GNodeArg(GNA_IN, "S48_Output", 0),
            GNodeArg(GNA_OUT, "S822_Output", 0)
        )
    );
    // Node S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1
    AddNode("S823__backbone_C3_p3_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S822_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1703", 0),
            GNodeArg(GNA_OUT, "S823_Output", 0)
        )
    );
    // Node _backbone_C3_p3_conv1_conv_Conv_split inq f16 outq f16
    AddNode("S824__backbone_C3_p3_conv1_conv_Conv_split_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S823_Output", 0),
            GNodeArg(GNA_OUT, "S824_Output_0", 0),
            GNodeArg(GNA_OUT, "S824_Output_1", 0)
        )
    );
    // Node S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1
    AddNode("S825__backbone_C3_p3_m_m_0_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S824_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_0_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1709", 0),
            GNodeArg(GNA_OUT, "S825_Output", 0)
        )
    );
    // Node S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S826__backbone_C3_p3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S825_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_0_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1712", 0),
            GNodeArg(GNA_OUT, "S826_Output", 0)
        )
    );
    // Node S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S827__backbone_C3_p3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S826_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_0_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1715", 0),
            GNodeArg(GNA_OUT, "S827_Output", 0)
        )
    );
    // Node S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1
    AddNode("S828__backbone_C3_p3_m_m_1_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S827_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_1_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1718", 0),
            GNodeArg(GNA_OUT, "S828_Output", 0)
        )
    );
    // Node S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S829__backbone_C3_p3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S828_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_1_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1721", 0),
            GNodeArg(GNA_OUT, "S829_Output", 0)
        )
    );
    // Node S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S830__backbone_C3_p3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S829_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_1_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1724", 0),
            GNodeArg(GNA_OUT, "S830_Output", 0)
        )
    );
    // Node S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1
    AddNode("S831__backbone_C3_p3_m_m_2_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S830_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_2_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1727", 0),
            GNodeArg(GNA_OUT, "S831_Output", 0)
        )
    );
    // Node S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S832__backbone_C3_p3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S831_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_2_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1730", 0),
            GNodeArg(GNA_OUT, "S832_Output", 0)
        )
    );
    // Node S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S833__backbone_C3_p3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S832_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_m_m_2_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1733", 0),
            GNodeArg(GNA_OUT, "S833_Output", 0)
        )
    );
    // Node _backbone_C3_p3_Concat inq f16 outq f16
    AddNode("S834__backbone_C3_p3_Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S833_Output", 0),
            GNodeArg(GNA_IN, "S824_Output_1", 0),
            GNodeArg(GNA_OUT, "S834_Output", 0)
        )
    );
    // Node S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1
    AddNode("S835__backbone_C3_p3_conv3_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S834_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_p3_conv3_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1736", 0),
            GNodeArg(GNA_OUT, "S835_Output", 0)
        )
    );
    // Node S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S836__backbone_bu_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S835_Output", 0),
            GNodeArg(GNA_IN, "_backbone_bu_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1739", 0),
            GNodeArg(GNA_OUT, "S836_Output", 0)
        )
    );
    // Node S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S837__backbone_bu_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S836_Output", 0),
            GNodeArg(GNA_IN, "_backbone_bu_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1742", 0),
            GNodeArg(GNA_OUT, "S837_Output", 0)
        )
    );
    // Node S838__head_stems_0_conv_Conv_fusion_multiple_1
    AddNode("S838__head_stems_0_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S835_Output", 0),
            GNodeArg(GNA_IN, "_head_stems_0_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1823", 0),
            GNodeArg(GNA_OUT, "S838_Output", 0)
        )
    );
    // Node S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S839__head_cls_convs_0_cls_convs_0_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S838_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_0_cls_convs_0_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1826", 0),
            GNodeArg(GNA_OUT, "S839_Output", 0)
        )
    );
    // Node S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S840__head_cls_convs_0_cls_convs_0_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S839_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_0_cls_convs_0_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1829", 0),
            GNodeArg(GNA_OUT, "S840_Output", 0)
        )
    );
    // Node S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S841__head_cls_convs_0_cls_convs_0_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S840_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_0_cls_convs_0_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1832", 0),
            GNodeArg(GNA_OUT, "S841_Output", 0)
        )
    );
    // Node S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1
    AddNode("S842__head_cls_convs_0_cls_convs_0_1_pconv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S841_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_0_cls_convs_0_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1835", 0),
            GNodeArg(GNA_OUT, "S842_Output", 0)
        )
    );
    // Node expr_67 in_qs [f16] out_qs [f16]
    AddNode("S843_expr_67_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S842_Output", 0),
            GNodeArg(GNA_OUT, "S843_Output", 0)
        )
    );
    // Node S844__head_cls_preds_0_Conv_fusion_multiple_1
    AddNode("S844__head_cls_preds_0_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S843_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_preds_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S844_Output", 0)
        )
    );
    // Node S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S845__head_reg_convs_0_reg_convs_0_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S838_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_0_reg_convs_0_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1838", 0),
            GNodeArg(GNA_OUT, "S845_Output", 0)
        )
    );
    // Node S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S846__head_reg_convs_0_reg_convs_0_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S845_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_0_reg_convs_0_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1841", 0),
            GNodeArg(GNA_OUT, "S846_Output", 0)
        )
    );
    // Node S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S847__head_reg_convs_0_reg_convs_0_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S846_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_0_reg_convs_0_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1844", 0),
            GNodeArg(GNA_OUT, "S847_Output", 0)
        )
    );
    // Node S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1
    AddNode("S848__head_reg_convs_0_reg_convs_0_1_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S847_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_0_reg_convs_0_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1847", 0),
            GNodeArg(GNA_OUT, "S848_Output", 0)
        )
    );
    // Node S849__head_reg_preds_0_Conv_multiple_1
    AddNode("S849__head_reg_preds_0_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S848_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_preds_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S849_Output", 0)
        )
    );
    // Node S850__head_obj_preds_0_Conv_fusion_multiple_1
    AddNode("S850__head_obj_preds_0_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S848_Output", 0),
            GNodeArg(GNA_IN, "_head_obj_preds_0_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_0_bias", 0),
            GNodeArg(GNA_OUT, "S850_Output", 0)
        )
    );
    // Node _head_Concat inq f16 outq f16
    AddNode("S851__head_Concat_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S849_Output", 0),
            GNodeArg(GNA_IN, "S850_Output", 0),
            GNodeArg(GNA_IN, "S844_Output", 0),
            GNodeArg(GNA_OUT, "S851_Output", 0)
        )
    );
    // Node _backbone_Concat_2 inq f16 outq f16
    AddNode("S853__backbone_Concat_2_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S837_Output", 0),
            GNodeArg(GNA_IN, "S820_Output", 0),
            GNodeArg(GNA_OUT, "S853_Output", 0)
        )
    );
    // Node S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1
    AddNode("S854__backbone_C3_n3_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S853_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1745", 0),
            GNodeArg(GNA_OUT, "S854_Output", 0)
        )
    );
    // Node _backbone_C3_n3_conv1_conv_Conv_split inq f16 outq f16
    AddNode("S855__backbone_C3_n3_conv1_conv_Conv_split_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S854_Output", 0),
            GNodeArg(GNA_OUT, "S855_Output_0", 0),
            GNodeArg(GNA_OUT, "S855_Output_1", 0)
        )
    );
    // Node S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1
    AddNode("S856__backbone_C3_n3_m_m_0_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S855_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_0_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1751", 0),
            GNodeArg(GNA_OUT, "S856_Output", 0)
        )
    );
    // Node S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S857__backbone_C3_n3_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S856_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_0_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1754", 0),
            GNodeArg(GNA_OUT, "S857_Output", 0)
        )
    );
    // Node S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S858__backbone_C3_n3_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S857_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_0_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1757", 0),
            GNodeArg(GNA_OUT, "S858_Output", 0)
        )
    );
    // Node S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1
    AddNode("S859__backbone_C3_n3_m_m_1_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S858_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_1_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1760", 0),
            GNodeArg(GNA_OUT, "S859_Output", 0)
        )
    );
    // Node S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S860__backbone_C3_n3_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S859_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_1_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1763", 0),
            GNodeArg(GNA_OUT, "S860_Output", 0)
        )
    );
    // Node S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S861__backbone_C3_n3_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S860_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_1_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1766", 0),
            GNodeArg(GNA_OUT, "S861_Output", 0)
        )
    );
    // Node S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1
    AddNode("S862__backbone_C3_n3_m_m_2_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S861_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_2_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1769", 0),
            GNodeArg(GNA_OUT, "S862_Output", 0)
        )
    );
    // Node S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S863__backbone_C3_n3_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S862_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_2_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1772", 0),
            GNodeArg(GNA_OUT, "S863_Output", 0)
        )
    );
    // Node S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S864__backbone_C3_n3_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S863_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_m_m_2_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1775", 0),
            GNodeArg(GNA_OUT, "S864_Output", 0)
        )
    );
    // Node _backbone_C3_n3_Concat inq f16 outq f16
    AddNode("S865__backbone_C3_n3_Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S864_Output", 0),
            GNodeArg(GNA_IN, "S855_Output_1", 0),
            GNodeArg(GNA_OUT, "S865_Output", 0)
        )
    );
    // Node S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1
    AddNode("S866__backbone_C3_n3_conv3_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S865_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n3_conv3_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1778", 0),
            GNodeArg(GNA_OUT, "S866_Output", 0)
        )
    );
    // Node S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S867__backbone_bu_conv1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S866_Output", 0),
            GNodeArg(GNA_IN, "_backbone_bu_conv1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1781", 0),
            GNodeArg(GNA_OUT, "S867_Output", 0)
        )
    );
    // Node S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1
    AddNode("S868__backbone_bu_conv1_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S867_Output", 0),
            GNodeArg(GNA_IN, "_backbone_bu_conv1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1784", 0),
            GNodeArg(GNA_OUT, "S868_Output", 0)
        )
    );
    // Node S869__head_stems_1_conv_Conv_fusion_multiple_1
    AddNode("S869__head_stems_1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S866_Output", 0),
            GNodeArg(GNA_IN, "_head_stems_1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1850", 0),
            GNodeArg(GNA_OUT, "S869_Output", 0)
        )
    );
    // Node S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S870__head_cls_convs_1_cls_convs_1_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S869_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_1_cls_convs_1_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1853", 0),
            GNodeArg(GNA_OUT, "S870_Output", 0)
        )
    );
    // Node S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S871__head_cls_convs_1_cls_convs_1_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S870_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_1_cls_convs_1_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1856", 0),
            GNodeArg(GNA_OUT, "S871_Output", 0)
        )
    );
    // Node S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S872__head_cls_convs_1_cls_convs_1_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S871_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_1_cls_convs_1_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1859", 0),
            GNodeArg(GNA_OUT, "S872_Output", 0)
        )
    );
    // Node S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1
    AddNode("S873__head_cls_convs_1_cls_convs_1_1_pconv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S872_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_1_cls_convs_1_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1862", 0),
            GNodeArg(GNA_OUT, "S873_Output", 0)
        )
    );
    // Node expr_90 in_qs [f16] out_qs [f16]
    AddNode("S874_expr_90_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S873_Output", 0),
            GNodeArg(GNA_OUT, "S874_Output", 0)
        )
    );
    // Node S875__head_cls_preds_1_Conv_fusion_multiple_1
    AddNode("S875__head_cls_preds_1_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S874_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_preds_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S875_Output", 0)
        )
    );
    // Node S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S876__head_reg_convs_1_reg_convs_1_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S869_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_1_reg_convs_1_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1865", 0),
            GNodeArg(GNA_OUT, "S876_Output", 0)
        )
    );
    // Node S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S877__head_reg_convs_1_reg_convs_1_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S876_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_1_reg_convs_1_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1868", 0),
            GNodeArg(GNA_OUT, "S877_Output", 0)
        )
    );
    // Node S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S878__head_reg_convs_1_reg_convs_1_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S877_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_1_reg_convs_1_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1871", 0),
            GNodeArg(GNA_OUT, "S878_Output", 0)
        )
    );
    // Node S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1
    AddNode("S879__head_reg_convs_1_reg_convs_1_1_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S878_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_1_reg_convs_1_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1874", 0),
            GNodeArg(GNA_OUT, "S879_Output", 0)
        )
    );
    // Node S880__head_reg_preds_1_Conv_multiple_1
    AddNode("S880__head_reg_preds_1_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S879_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_preds_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S880_Output", 0)
        )
    );
    // Node S881__head_obj_preds_1_Conv_fusion_multiple_1
    AddNode("S881__head_obj_preds_1_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S879_Output", 0),
            GNodeArg(GNA_IN, "_head_obj_preds_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_1_bias", 0),
            GNodeArg(GNA_OUT, "S881_Output", 0)
        )
    );
    // Node _head_Concat_1 inq f16 outq f16
    AddNode("S882__head_Concat_1_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S880_Output", 0),
            GNodeArg(GNA_IN, "S881_Output", 0),
            GNodeArg(GNA_IN, "S875_Output", 0),
            GNodeArg(GNA_OUT, "S882_Output", 0)
        )
    );
    // Node _backbone_Concat_3 inq f16 outq f16
    AddNode("S884__backbone_Concat_3_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S868_Output", 0),
            GNodeArg(GNA_IN, "S795_Output", 0),
            GNodeArg(GNA_OUT, "S884_Output", 0)
        )
    );
    // Node S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1
    AddNode("S885__backbone_C3_n4_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S884_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1787", 0),
            GNodeArg(GNA_OUT, "S885_Output", 0)
        )
    );
    // Node _backbone_C3_n4_conv1_conv_Conv_split inq f16 outq f16
    AddNode("S886__backbone_C3_n4_conv1_conv_Conv_split_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S885_Output", 0),
            GNodeArg(GNA_OUT, "S886_Output_0", 0),
            GNodeArg(GNA_OUT, "S886_Output_1", 0)
        )
    );
    // Node S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1
    AddNode("S887__backbone_C3_n4_m_m_0_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S886_Output_0", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_0_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1793", 0),
            GNodeArg(GNA_OUT, "S887_Output", 0)
        )
    );
    // Node S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S888__backbone_C3_n4_m_m_0_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S887_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_0_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1796", 0),
            GNodeArg(GNA_OUT, "S888_Output", 0)
        )
    );
    // Node S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S889__backbone_C3_n4_m_m_0_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S888_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_0_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1799", 0),
            GNodeArg(GNA_OUT, "S889_Output", 0)
        )
    );
    // Node S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1
    AddNode("S890__backbone_C3_n4_m_m_1_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S889_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_1_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1802", 0),
            GNodeArg(GNA_OUT, "S890_Output", 0)
        )
    );
    // Node S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S891__backbone_C3_n4_m_m_1_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S890_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_1_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1805", 0),
            GNodeArg(GNA_OUT, "S891_Output", 0)
        )
    );
    // Node S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S892__backbone_C3_n4_m_m_1_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S891_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_1_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1808", 0),
            GNodeArg(GNA_OUT, "S892_Output", 0)
        )
    );
    // Node S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1
    AddNode("S893__backbone_C3_n4_m_m_2_conv1_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S892_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_2_conv1_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1811", 0),
            GNodeArg(GNA_OUT, "S893_Output", 0)
        )
    );
    // Node S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1
    AddNode("S894__backbone_C3_n4_m_m_2_conv2_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S893_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_2_conv2_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1814", 0),
            GNodeArg(GNA_OUT, "S894_Output", 0)
        )
    );
    // Node S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1
    AddNode("S895__backbone_C3_n4_m_m_2_conv2_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S894_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_m_m_2_conv2_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1817", 0),
            GNodeArg(GNA_OUT, "S895_Output", 0)
        )
    );
    // Node _backbone_C3_n4_Concat inq f16 outq f16
    AddNode("S896__backbone_C3_n4_Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S895_Output", 0),
            GNodeArg(GNA_IN, "S886_Output_1", 0),
            GNodeArg(GNA_OUT, "S896_Output", 0)
        )
    );
    // Node S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1
    AddNode("S897__backbone_C3_n4_conv3_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S896_Output", 0),
            GNodeArg(GNA_IN, "_backbone_c3_n4_conv3_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1820", 0),
            GNodeArg(GNA_OUT, "S897_Output", 0)
        )
    );
    // Node S898__head_stems_2_conv_Conv_fusion_multiple_1
    AddNode("S898__head_stems_2_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S897_Output", 0),
            GNodeArg(GNA_IN, "_head_stems_2_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1877", 0),
            GNodeArg(GNA_OUT, "S898_Output", 0)
        )
    );
    // Node S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S899__head_cls_convs_2_cls_convs_2_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S898_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_2_cls_convs_2_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1880", 0),
            GNodeArg(GNA_OUT, "S899_Output", 0)
        )
    );
    // Node S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S900__head_cls_convs_2_cls_convs_2_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S899_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_2_cls_convs_2_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1883", 0),
            GNodeArg(GNA_OUT, "S900_Output", 0)
        )
    );
    // Node S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S901__head_cls_convs_2_cls_convs_2_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S900_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_2_cls_convs_2_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1886", 0),
            GNodeArg(GNA_OUT, "S901_Output", 0)
        )
    );
    // Node S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1
    AddNode("S902__head_cls_convs_2_cls_convs_2_1_pconv_conv_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S901_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_convs_2_cls_convs_2_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1889", 0),
            GNodeArg(GNA_OUT, "S902_Output", 0)
        )
    );
    // Node expr_113 in_qs [f16] out_qs [f16]
    AddNode("S903_expr_113_multiple_1",
        Bindings(2,
            GNodeArg(GNA_IN, "S902_Output", 0),
            GNodeArg(GNA_OUT, "S903_Output", 0)
        )
    );
    // Node S904__head_cls_preds_2_Conv_fusion_multiple_1
    AddNode("S904__head_cls_preds_2_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S903_Output", 0),
            GNodeArg(GNA_IN, "_head_cls_preds_2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_cls_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S904_Output", 0)
        )
    );
    // Node S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1
    AddNode("S905__head_reg_convs_2_reg_convs_2_0_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S898_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_2_reg_convs_2_0_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1892", 0),
            GNodeArg(GNA_OUT, "S905_Output", 0)
        )
    );
    // Node S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1
    AddNode("S906__head_reg_convs_2_reg_convs_2_0_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S905_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_2_reg_convs_2_0_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1895", 0),
            GNodeArg(GNA_OUT, "S906_Output", 0)
        )
    );
    // Node S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1
    AddNode("S907__head_reg_convs_2_reg_convs_2_1_dconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S906_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_2_reg_convs_2_1_dconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1898", 0),
            GNodeArg(GNA_OUT, "S907_Output", 0)
        )
    );
    // Node S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1
    AddNode("S908__head_reg_convs_2_reg_convs_2_1_pconv_conv_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S907_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_convs_2_reg_convs_2_1_pconv_conv_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_1901", 0),
            GNodeArg(GNA_OUT, "S908_Output", 0)
        )
    );
    // Node S909__head_reg_preds_2_Conv_multiple_1
    AddNode("S909__head_reg_preds_2_Conv_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S908_Output", 0),
            GNodeArg(GNA_IN, "_head_reg_preds_2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_reg_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S909_Output", 0)
        )
    );
    // Node S910__head_obj_preds_2_Conv_fusion_multiple_1
    AddNode("S910__head_obj_preds_2_Conv_fusion_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S908_Output", 0),
            GNodeArg(GNA_IN, "_head_obj_preds_2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_head_obj_preds_2_bias", 0),
            GNodeArg(GNA_OUT, "S910_Output", 0)
        )
    );
    // Node _head_Concat_2 inq f16 outq f16
    AddNode("S911__head_Concat_2_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S909_Output", 0),
            GNodeArg(GNA_IN, "S910_Output", 0),
            GNodeArg(GNA_IN, "S904_Output", 0),
            GNodeArg(GNA_OUT, "S911_Output", 0)
        )
    );
    // Node _Slice_split inq f16 outq f16
    AddNode("S914__Slice_split_multiple_1",
        Bindings(4,
            GNodeArg(GNA_IN, "S913_Output", 0),
            GNodeArg(GNA_OUT, "S914_Output_0", 0),
            GNodeArg(GNA_OUT, "S914_Output_1", 0),
            GNodeArg(GNA_OUT, "S914_Output_2", 0)
        )
    );
    // Node expr_118 in_qs [f16,f16] out_qs [f16]
    AddNode("S915_expr_118_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S914_Output_1", 0),
            GNodeArg(GNA_IN, "S914_Output_2", 0),
            GNodeArg(GNA_OUT, "S915_Output", 0)
        )
    );
    // Node _Concat inq f16 outq f16
    AddNode("S916__Concat_multiple_1",
        Bindings(3,
            GNodeArg(GNA_IN, "S914_Output_0", 0),
            GNodeArg(GNA_IN, "S915_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    onnx_graphModel(128000, 692000, 32000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
