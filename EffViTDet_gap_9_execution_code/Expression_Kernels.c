#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "Expression_Kernels.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

#ifndef AT_NORM
#define AT_NORM(x, n)   gap_roundnorm_reg((x), (n))
#endif
#define ATLShift(x, n)  ((x) << (n))


#ifndef B_CLR
#define B_CLR(x, bits) ((x) & (~((1 << (bits)) - 1)))
#endif

static inline void __attribute__((always_inline)) Copy(char *__restrict__ To, char *__restrict__ From, unsigned int Size)
{
	int *pFrom = (int *)(From), *pTo = (int *)(To);
	for (int i = 0; i < Size / 8; i++)
	{
		int V0 = pFrom[2 * i], V1 = pFrom[2 * i + 1];
		pTo[2 * i] = V0;
		pTo[2 * i + 1] = V1;
	}
	if (Size & 0x4)
		*((int *)(To + B_CLR(Size, 3))) = *((int *)(From + B_CLR(Size, 3)));
	if (Size & 0x2)
		*((short int *)(To + B_CLR(Size, 2))) = *((short int *)(From + B_CLR(Size, 2)));
	if (Size & 0x1)
		*((signed char *)(To + Size - 1)) = *((signed char *)(From + Size - 1));
}

static inline void __attribute__((always_inline)) ParCopy(char *__restrict__ To, char *__restrict__ From, unsigned int Size, unsigned int CoreId)
{
	unsigned int Chunk = ChunkSize(Size), First = Min(Chunk * CoreId, Size), Last = Min(First + Chunk, Size);
	unsigned int Iter = Last - First;
	int *pFrom = (int *)(From + First), *pTo = (int *)(To + First);
	for (int i = 0; i < Iter / 8; i++)
	{
		int V0 = pFrom[2 * i], V1 = pFrom[2 * i + 1];
		pTo[2 * i] = V0;
		pTo[2 * i + 1] = V1;
	}
	if (Iter & 0x4)
		*((int *)(To + First + B_CLR(Iter, 3))) = *((int *)(From + First + B_CLR(Iter, 3)));
	if (Iter & 0x2)
		*((short int *)(To + First + B_CLR(Iter, 2))) = *((short int *)(From + First + B_CLR(Iter, 2)));
	if (Iter & 0x1)
		*((signed char *)(To + First + Iter - 1)) = *((signed char *)(From + First + Iter - 1));
}

void CNN_Split_Width_In12(SplitWidthIn12Arg_T *Arg)
{
    char *__restrict__ In = (char *__restrict__)Arg->In;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ Out1 = (char *__restrict__)Arg->Out1;
    int W1 = (int)Arg->W1 * DataSize;
    int S1 = (int)Arg->S1 * DataSize;
    char *__restrict__ Out2 = (char *__restrict__)Arg->Out2;
    int W2 = (int)Arg->W2 * DataSize;
    int S2 = (int)Arg->S2 * DataSize;
    char *__restrict__ Out3 = (char *__restrict__)Arg->Out3;
    int W3 = (int)Arg->W3 * DataSize;
    int S3 = (int)Arg->S3 * DataSize;
    char *__restrict__ Out4 = (char *__restrict__)Arg->Out4;
    int W4 = (int)Arg->W4 * DataSize;
    int S4 = (int)Arg->S4 * DataSize;
    char *__restrict__ Out5 = (char *__restrict__)Arg->Out5;
    int W5 = (int)Arg->W5 * DataSize;
    int S5 = (int)Arg->S5 * DataSize;
    char *__restrict__ Out6 = (char *__restrict__)Arg->Out6;
    int W6 = (int)Arg->W6 * DataSize;
    int S6 = (int)Arg->S6 * DataSize;
    char *__restrict__ Out7 = (char *__restrict__)Arg->Out7;
    int W7 = (int)Arg->W7 * DataSize;
    int S7 = (int)Arg->S7 * DataSize;
    char *__restrict__ Out8 = (char *__restrict__)Arg->Out8;
    int W8 = (int)Arg->W8 * DataSize;
    int S8 = (int)Arg->S8 * DataSize;
    char *__restrict__ Out9 = (char *__restrict__)Arg->Out9;
    int W9 = (int)Arg->W9 * DataSize;
    int S9 = (int)Arg->S9 * DataSize;
    char *__restrict__ Out10 = (char *__restrict__)Arg->Out10;
    int W10 = (int)Arg->W10 * DataSize;
    int S10 = (int)Arg->S10 * DataSize;
    char *__restrict__ Out11 = (char *__restrict__)Arg->Out11;
    int W11 = (int)Arg->W11 * DataSize;
    int S11 = (int)Arg->S11 * DataSize;
    char *__restrict__ Out12 = (char *__restrict__)Arg->Out12;
    int W12 = (int)Arg->W12 * DataSize;
    int S12 = (int)Arg->S12 * DataSize;
    int Wi = (int)Arg->InWidth * DataSize;
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(H), First = Min(Chunk * CoreId, H), Last = Min(First + Chunk, H);
    for (int h = First; h < Last; h++)
    {
        Copy(Out1 + h * W1, In + h * Wi + S1, W1);
        Copy(Out2 + h * W2, In + h * Wi + S2, W2);
        Copy(Out3 + h * W3, In + h * Wi + S3, W3);
        Copy(Out4 + h * W4, In + h * Wi + S4, W4);
        Copy(Out5 + h * W5, In + h * Wi + S5, W5);
        Copy(Out6 + h * W6, In + h * Wi + S6, W6);
        Copy(Out7 + h * W7, In + h * Wi + S7, W7);
        Copy(Out8 + h * W8, In + h * Wi + S8, W8);
        Copy(Out9 + h * W9, In + h * Wi + S9, W9);
        Copy(Out10 + h * W10, In + h * Wi + S10, W10);
        Copy(Out11 + h * W11, In + h * Wi + S11, W11);
        Copy(Out12 + h * W12, In + h * Wi + S12, W12);
    }
    gap_waitbarrier(0);
}

void CNN_ParSplit_Width_In12(SplitWidthIn12Arg_T *Arg)
{
    char *__restrict__ In = (char *__restrict__)Arg->In;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ Out1 = (char *__restrict__)Arg->Out1;
    int W1 = (int)Arg->W1 * DataSize;
    int S1 = (int)Arg->S1 * DataSize;
    char *__restrict__ Out2 = (char *__restrict__)Arg->Out2;
    int W2 = (int)Arg->W2 * DataSize;
    int S2 = (int)Arg->S2 * DataSize;
    char *__restrict__ Out3 = (char *__restrict__)Arg->Out3;
    int W3 = (int)Arg->W3 * DataSize;
    int S3 = (int)Arg->S3 * DataSize;
    char *__restrict__ Out4 = (char *__restrict__)Arg->Out4;
    int W4 = (int)Arg->W4 * DataSize;
    int S4 = (int)Arg->S4 * DataSize;
    char *__restrict__ Out5 = (char *__restrict__)Arg->Out5;
    int W5 = (int)Arg->W5 * DataSize;
    int S5 = (int)Arg->S5 * DataSize;
    char *__restrict__ Out6 = (char *__restrict__)Arg->Out6;
    int W6 = (int)Arg->W6 * DataSize;
    int S6 = (int)Arg->S6 * DataSize;
    char *__restrict__ Out7 = (char *__restrict__)Arg->Out7;
    int W7 = (int)Arg->W7 * DataSize;
    int S7 = (int)Arg->S7 * DataSize;
    char *__restrict__ Out8 = (char *__restrict__)Arg->Out8;
    int W8 = (int)Arg->W8 * DataSize;
    int S8 = (int)Arg->S8 * DataSize;
    char *__restrict__ Out9 = (char *__restrict__)Arg->Out9;
    int W9 = (int)Arg->W9 * DataSize;
    int S9 = (int)Arg->S9 * DataSize;
    char *__restrict__ Out10 = (char *__restrict__)Arg->Out10;
    int W10 = (int)Arg->W10 * DataSize;
    int S10 = (int)Arg->S10 * DataSize;
    char *__restrict__ Out11 = (char *__restrict__)Arg->Out11;
    int W11 = (int)Arg->W11 * DataSize;
    int S11 = (int)Arg->S11 * DataSize;
    char *__restrict__ Out12 = (char *__restrict__)Arg->Out12;
    int W12 = (int)Arg->W12 * DataSize;
    int S12 = (int)Arg->S12 * DataSize;
    int Wi = (int)Arg->InWidth * DataSize;
    unsigned int CoreId = gap_coreid();
    for (int h = 0; h < H; h++)
    {
        ParCopy(Out1 + h * W1, In + h * Wi + S1, W1, CoreId);
        ParCopy(Out2 + h * W2, In + h * Wi + S2, W2, CoreId);
        ParCopy(Out3 + h * W3, In + h * Wi + S3, W3, CoreId);
        ParCopy(Out4 + h * W4, In + h * Wi + S4, W4, CoreId);
        ParCopy(Out5 + h * W5, In + h * Wi + S5, W5, CoreId);
        ParCopy(Out6 + h * W6, In + h * Wi + S6, W6, CoreId);
        ParCopy(Out7 + h * W7, In + h * Wi + S7, W7, CoreId);
        ParCopy(Out8 + h * W8, In + h * Wi + S8, W8, CoreId);
        ParCopy(Out9 + h * W9, In + h * Wi + S9, W9, CoreId);
        ParCopy(Out10 + h * W10, In + h * Wi + S10, W10, CoreId);
        ParCopy(Out11 + h * W11, In + h * Wi + S11, W11, CoreId);
        ParCopy(Out12 + h * W12, In + h * Wi + S12, W12, CoreId);
    }
    gap_waitbarrier(0);
}

void CNN_Concat_Width_In13(ConcatWidthIn13Arg_T *Arg)
{
    char *__restrict__ Out = (char *__restrict__)Arg->Out;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ In1 = (char *__restrict__)Arg->In1;
    int W1 = (int)Arg->W1 * DataSize;
    char *__restrict__ In2 = (char *__restrict__)Arg->In2;
    int W2 = (int)Arg->W2 * DataSize;
    char *__restrict__ In3 = (char *__restrict__)Arg->In3;
    int W3 = (int)Arg->W3 * DataSize;
    char *__restrict__ In4 = (char *__restrict__)Arg->In4;
    int W4 = (int)Arg->W4 * DataSize;
    char *__restrict__ In5 = (char *__restrict__)Arg->In5;
    int W5 = (int)Arg->W5 * DataSize;
    char *__restrict__ In6 = (char *__restrict__)Arg->In6;
    int W6 = (int)Arg->W6 * DataSize;
    char *__restrict__ In7 = (char *__restrict__)Arg->In7;
    int W7 = (int)Arg->W7 * DataSize;
    char *__restrict__ In8 = (char *__restrict__)Arg->In8;
    int W8 = (int)Arg->W8 * DataSize;
    char *__restrict__ In9 = (char *__restrict__)Arg->In9;
    int W9 = (int)Arg->W9 * DataSize;
    char *__restrict__ In10 = (char *__restrict__)Arg->In10;
    int W10 = (int)Arg->W10 * DataSize;
    char *__restrict__ In11 = (char *__restrict__)Arg->In11;
    int W11 = (int)Arg->W11 * DataSize;
    char *__restrict__ In12 = (char *__restrict__)Arg->In12;
    int W12 = (int)Arg->W12 * DataSize;
    char *__restrict__ In13 = (char *__restrict__)Arg->In13;
    int W13 = (int)Arg->W13 * DataSize;
    int Wo = W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13;
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(H), First = Min(Chunk * CoreId, H), Last = Min(First + Chunk, H);
    for (int h = First; h < Last; h++)
    {
        Copy(Out + h * Wo, In1 + h * W1, W1);
        Copy(Out + h * Wo + W1, In2 + h * W2, W2);
        Copy(Out + h * Wo + W1 + W2, In3 + h * W3, W3);
        Copy(Out + h * Wo + W1 + W2 + W3, In4 + h * W4, W4);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4, In5 + h * W5, W5);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5, In6 + h * W6, W6);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6, In7 + h * W7, W7);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7, In8 + h * W8, W8);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8, In9 + h * W9, W9);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9, In10 + h * W10, W10);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10, In11 + h * W11, W11);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11, In12 + h * W12, W12);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12, In13 + h * W13, W13);
    }
    gap_waitbarrier(0);
}

void CNN_Split_Width_In24(SplitWidthIn24Arg_T *Arg)
{
    char *__restrict__ In = (char *__restrict__)Arg->In;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ Out1 = (char *__restrict__)Arg->Out1;
    int W1 = (int)Arg->W1 * DataSize;
    int S1 = (int)Arg->S1 * DataSize;
    char *__restrict__ Out2 = (char *__restrict__)Arg->Out2;
    int W2 = (int)Arg->W2 * DataSize;
    int S2 = (int)Arg->S2 * DataSize;
    char *__restrict__ Out3 = (char *__restrict__)Arg->Out3;
    int W3 = (int)Arg->W3 * DataSize;
    int S3 = (int)Arg->S3 * DataSize;
    char *__restrict__ Out4 = (char *__restrict__)Arg->Out4;
    int W4 = (int)Arg->W4 * DataSize;
    int S4 = (int)Arg->S4 * DataSize;
    char *__restrict__ Out5 = (char *__restrict__)Arg->Out5;
    int W5 = (int)Arg->W5 * DataSize;
    int S5 = (int)Arg->S5 * DataSize;
    char *__restrict__ Out6 = (char *__restrict__)Arg->Out6;
    int W6 = (int)Arg->W6 * DataSize;
    int S6 = (int)Arg->S6 * DataSize;
    char *__restrict__ Out7 = (char *__restrict__)Arg->Out7;
    int W7 = (int)Arg->W7 * DataSize;
    int S7 = (int)Arg->S7 * DataSize;
    char *__restrict__ Out8 = (char *__restrict__)Arg->Out8;
    int W8 = (int)Arg->W8 * DataSize;
    int S8 = (int)Arg->S8 * DataSize;
    char *__restrict__ Out9 = (char *__restrict__)Arg->Out9;
    int W9 = (int)Arg->W9 * DataSize;
    int S9 = (int)Arg->S9 * DataSize;
    char *__restrict__ Out10 = (char *__restrict__)Arg->Out10;
    int W10 = (int)Arg->W10 * DataSize;
    int S10 = (int)Arg->S10 * DataSize;
    char *__restrict__ Out11 = (char *__restrict__)Arg->Out11;
    int W11 = (int)Arg->W11 * DataSize;
    int S11 = (int)Arg->S11 * DataSize;
    char *__restrict__ Out12 = (char *__restrict__)Arg->Out12;
    int W12 = (int)Arg->W12 * DataSize;
    int S12 = (int)Arg->S12 * DataSize;
    char *__restrict__ Out13 = (char *__restrict__)Arg->Out13;
    int W13 = (int)Arg->W13 * DataSize;
    int S13 = (int)Arg->S13 * DataSize;
    char *__restrict__ Out14 = (char *__restrict__)Arg->Out14;
    int W14 = (int)Arg->W14 * DataSize;
    int S14 = (int)Arg->S14 * DataSize;
    char *__restrict__ Out15 = (char *__restrict__)Arg->Out15;
    int W15 = (int)Arg->W15 * DataSize;
    int S15 = (int)Arg->S15 * DataSize;
    char *__restrict__ Out16 = (char *__restrict__)Arg->Out16;
    int W16 = (int)Arg->W16 * DataSize;
    int S16 = (int)Arg->S16 * DataSize;
    char *__restrict__ Out17 = (char *__restrict__)Arg->Out17;
    int W17 = (int)Arg->W17 * DataSize;
    int S17 = (int)Arg->S17 * DataSize;
    char *__restrict__ Out18 = (char *__restrict__)Arg->Out18;
    int W18 = (int)Arg->W18 * DataSize;
    int S18 = (int)Arg->S18 * DataSize;
    char *__restrict__ Out19 = (char *__restrict__)Arg->Out19;
    int W19 = (int)Arg->W19 * DataSize;
    int S19 = (int)Arg->S19 * DataSize;
    char *__restrict__ Out20 = (char *__restrict__)Arg->Out20;
    int W20 = (int)Arg->W20 * DataSize;
    int S20 = (int)Arg->S20 * DataSize;
    char *__restrict__ Out21 = (char *__restrict__)Arg->Out21;
    int W21 = (int)Arg->W21 * DataSize;
    int S21 = (int)Arg->S21 * DataSize;
    char *__restrict__ Out22 = (char *__restrict__)Arg->Out22;
    int W22 = (int)Arg->W22 * DataSize;
    int S22 = (int)Arg->S22 * DataSize;
    char *__restrict__ Out23 = (char *__restrict__)Arg->Out23;
    int W23 = (int)Arg->W23 * DataSize;
    int S23 = (int)Arg->S23 * DataSize;
    char *__restrict__ Out24 = (char *__restrict__)Arg->Out24;
    int W24 = (int)Arg->W24 * DataSize;
    int S24 = (int)Arg->S24 * DataSize;
    int Wi = (int)Arg->InWidth * DataSize;
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(H), First = Min(Chunk * CoreId, H), Last = Min(First + Chunk, H);
    for (int h = First; h < Last; h++)
    {
        Copy(Out1 + h * W1, In + h * Wi + S1, W1);
        Copy(Out2 + h * W2, In + h * Wi + S2, W2);
        Copy(Out3 + h * W3, In + h * Wi + S3, W3);
        Copy(Out4 + h * W4, In + h * Wi + S4, W4);
        Copy(Out5 + h * W5, In + h * Wi + S5, W5);
        Copy(Out6 + h * W6, In + h * Wi + S6, W6);
        Copy(Out7 + h * W7, In + h * Wi + S7, W7);
        Copy(Out8 + h * W8, In + h * Wi + S8, W8);
        Copy(Out9 + h * W9, In + h * Wi + S9, W9);
        Copy(Out10 + h * W10, In + h * Wi + S10, W10);
        Copy(Out11 + h * W11, In + h * Wi + S11, W11);
        Copy(Out12 + h * W12, In + h * Wi + S12, W12);
        Copy(Out13 + h * W13, In + h * Wi + S13, W13);
        Copy(Out14 + h * W14, In + h * Wi + S14, W14);
        Copy(Out15 + h * W15, In + h * Wi + S15, W15);
        Copy(Out16 + h * W16, In + h * Wi + S16, W16);
        Copy(Out17 + h * W17, In + h * Wi + S17, W17);
        Copy(Out18 + h * W18, In + h * Wi + S18, W18);
        Copy(Out19 + h * W19, In + h * Wi + S19, W19);
        Copy(Out20 + h * W20, In + h * Wi + S20, W20);
        Copy(Out21 + h * W21, In + h * Wi + S21, W21);
        Copy(Out22 + h * W22, In + h * Wi + S22, W22);
        Copy(Out23 + h * W23, In + h * Wi + S23, W23);
        Copy(Out24 + h * W24, In + h * Wi + S24, W24);
    }
    gap_waitbarrier(0);
}

void CNN_ParSplit_Width_In24(SplitWidthIn24Arg_T *Arg)
{
    char *__restrict__ In = (char *__restrict__)Arg->In;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ Out1 = (char *__restrict__)Arg->Out1;
    int W1 = (int)Arg->W1 * DataSize;
    int S1 = (int)Arg->S1 * DataSize;
    char *__restrict__ Out2 = (char *__restrict__)Arg->Out2;
    int W2 = (int)Arg->W2 * DataSize;
    int S2 = (int)Arg->S2 * DataSize;
    char *__restrict__ Out3 = (char *__restrict__)Arg->Out3;
    int W3 = (int)Arg->W3 * DataSize;
    int S3 = (int)Arg->S3 * DataSize;
    char *__restrict__ Out4 = (char *__restrict__)Arg->Out4;
    int W4 = (int)Arg->W4 * DataSize;
    int S4 = (int)Arg->S4 * DataSize;
    char *__restrict__ Out5 = (char *__restrict__)Arg->Out5;
    int W5 = (int)Arg->W5 * DataSize;
    int S5 = (int)Arg->S5 * DataSize;
    char *__restrict__ Out6 = (char *__restrict__)Arg->Out6;
    int W6 = (int)Arg->W6 * DataSize;
    int S6 = (int)Arg->S6 * DataSize;
    char *__restrict__ Out7 = (char *__restrict__)Arg->Out7;
    int W7 = (int)Arg->W7 * DataSize;
    int S7 = (int)Arg->S7 * DataSize;
    char *__restrict__ Out8 = (char *__restrict__)Arg->Out8;
    int W8 = (int)Arg->W8 * DataSize;
    int S8 = (int)Arg->S8 * DataSize;
    char *__restrict__ Out9 = (char *__restrict__)Arg->Out9;
    int W9 = (int)Arg->W9 * DataSize;
    int S9 = (int)Arg->S9 * DataSize;
    char *__restrict__ Out10 = (char *__restrict__)Arg->Out10;
    int W10 = (int)Arg->W10 * DataSize;
    int S10 = (int)Arg->S10 * DataSize;
    char *__restrict__ Out11 = (char *__restrict__)Arg->Out11;
    int W11 = (int)Arg->W11 * DataSize;
    int S11 = (int)Arg->S11 * DataSize;
    char *__restrict__ Out12 = (char *__restrict__)Arg->Out12;
    int W12 = (int)Arg->W12 * DataSize;
    int S12 = (int)Arg->S12 * DataSize;
    char *__restrict__ Out13 = (char *__restrict__)Arg->Out13;
    int W13 = (int)Arg->W13 * DataSize;
    int S13 = (int)Arg->S13 * DataSize;
    char *__restrict__ Out14 = (char *__restrict__)Arg->Out14;
    int W14 = (int)Arg->W14 * DataSize;
    int S14 = (int)Arg->S14 * DataSize;
    char *__restrict__ Out15 = (char *__restrict__)Arg->Out15;
    int W15 = (int)Arg->W15 * DataSize;
    int S15 = (int)Arg->S15 * DataSize;
    char *__restrict__ Out16 = (char *__restrict__)Arg->Out16;
    int W16 = (int)Arg->W16 * DataSize;
    int S16 = (int)Arg->S16 * DataSize;
    char *__restrict__ Out17 = (char *__restrict__)Arg->Out17;
    int W17 = (int)Arg->W17 * DataSize;
    int S17 = (int)Arg->S17 * DataSize;
    char *__restrict__ Out18 = (char *__restrict__)Arg->Out18;
    int W18 = (int)Arg->W18 * DataSize;
    int S18 = (int)Arg->S18 * DataSize;
    char *__restrict__ Out19 = (char *__restrict__)Arg->Out19;
    int W19 = (int)Arg->W19 * DataSize;
    int S19 = (int)Arg->S19 * DataSize;
    char *__restrict__ Out20 = (char *__restrict__)Arg->Out20;
    int W20 = (int)Arg->W20 * DataSize;
    int S20 = (int)Arg->S20 * DataSize;
    char *__restrict__ Out21 = (char *__restrict__)Arg->Out21;
    int W21 = (int)Arg->W21 * DataSize;
    int S21 = (int)Arg->S21 * DataSize;
    char *__restrict__ Out22 = (char *__restrict__)Arg->Out22;
    int W22 = (int)Arg->W22 * DataSize;
    int S22 = (int)Arg->S22 * DataSize;
    char *__restrict__ Out23 = (char *__restrict__)Arg->Out23;
    int W23 = (int)Arg->W23 * DataSize;
    int S23 = (int)Arg->S23 * DataSize;
    char *__restrict__ Out24 = (char *__restrict__)Arg->Out24;
    int W24 = (int)Arg->W24 * DataSize;
    int S24 = (int)Arg->S24 * DataSize;
    int Wi = (int)Arg->InWidth * DataSize;
    unsigned int CoreId = gap_coreid();
    for (int h = 0; h < H; h++)
    {
        ParCopy(Out1 + h * W1, In + h * Wi + S1, W1, CoreId);
        ParCopy(Out2 + h * W2, In + h * Wi + S2, W2, CoreId);
        ParCopy(Out3 + h * W3, In + h * Wi + S3, W3, CoreId);
        ParCopy(Out4 + h * W4, In + h * Wi + S4, W4, CoreId);
        ParCopy(Out5 + h * W5, In + h * Wi + S5, W5, CoreId);
        ParCopy(Out6 + h * W6, In + h * Wi + S6, W6, CoreId);
        ParCopy(Out7 + h * W7, In + h * Wi + S7, W7, CoreId);
        ParCopy(Out8 + h * W8, In + h * Wi + S8, W8, CoreId);
        ParCopy(Out9 + h * W9, In + h * Wi + S9, W9, CoreId);
        ParCopy(Out10 + h * W10, In + h * Wi + S10, W10, CoreId);
        ParCopy(Out11 + h * W11, In + h * Wi + S11, W11, CoreId);
        ParCopy(Out12 + h * W12, In + h * Wi + S12, W12, CoreId);
        ParCopy(Out13 + h * W13, In + h * Wi + S13, W13, CoreId);
        ParCopy(Out14 + h * W14, In + h * Wi + S14, W14, CoreId);
        ParCopy(Out15 + h * W15, In + h * Wi + S15, W15, CoreId);
        ParCopy(Out16 + h * W16, In + h * Wi + S16, W16, CoreId);
        ParCopy(Out17 + h * W17, In + h * Wi + S17, W17, CoreId);
        ParCopy(Out18 + h * W18, In + h * Wi + S18, W18, CoreId);
        ParCopy(Out19 + h * W19, In + h * Wi + S19, W19, CoreId);
        ParCopy(Out20 + h * W20, In + h * Wi + S20, W20, CoreId);
        ParCopy(Out21 + h * W21, In + h * Wi + S21, W21, CoreId);
        ParCopy(Out22 + h * W22, In + h * Wi + S22, W22, CoreId);
        ParCopy(Out23 + h * W23, In + h * Wi + S23, W23, CoreId);
        ParCopy(Out24 + h * W24, In + h * Wi + S24, W24, CoreId);
    }
    gap_waitbarrier(0);
}

void CNN_Concat_Width_In25(ConcatWidthIn25Arg_T *Arg)
{
    char *__restrict__ Out = (char *__restrict__)Arg->Out;
    int DataSize = (int)Arg->DataSize;
    int H = (int)Arg->H;
    char *__restrict__ In1 = (char *__restrict__)Arg->In1;
    int W1 = (int)Arg->W1 * DataSize;
    char *__restrict__ In2 = (char *__restrict__)Arg->In2;
    int W2 = (int)Arg->W2 * DataSize;
    char *__restrict__ In3 = (char *__restrict__)Arg->In3;
    int W3 = (int)Arg->W3 * DataSize;
    char *__restrict__ In4 = (char *__restrict__)Arg->In4;
    int W4 = (int)Arg->W4 * DataSize;
    char *__restrict__ In5 = (char *__restrict__)Arg->In5;
    int W5 = (int)Arg->W5 * DataSize;
    char *__restrict__ In6 = (char *__restrict__)Arg->In6;
    int W6 = (int)Arg->W6 * DataSize;
    char *__restrict__ In7 = (char *__restrict__)Arg->In7;
    int W7 = (int)Arg->W7 * DataSize;
    char *__restrict__ In8 = (char *__restrict__)Arg->In8;
    int W8 = (int)Arg->W8 * DataSize;
    char *__restrict__ In9 = (char *__restrict__)Arg->In9;
    int W9 = (int)Arg->W9 * DataSize;
    char *__restrict__ In10 = (char *__restrict__)Arg->In10;
    int W10 = (int)Arg->W10 * DataSize;
    char *__restrict__ In11 = (char *__restrict__)Arg->In11;
    int W11 = (int)Arg->W11 * DataSize;
    char *__restrict__ In12 = (char *__restrict__)Arg->In12;
    int W12 = (int)Arg->W12 * DataSize;
    char *__restrict__ In13 = (char *__restrict__)Arg->In13;
    int W13 = (int)Arg->W13 * DataSize;
    char *__restrict__ In14 = (char *__restrict__)Arg->In14;
    int W14 = (int)Arg->W14 * DataSize;
    char *__restrict__ In15 = (char *__restrict__)Arg->In15;
    int W15 = (int)Arg->W15 * DataSize;
    char *__restrict__ In16 = (char *__restrict__)Arg->In16;
    int W16 = (int)Arg->W16 * DataSize;
    char *__restrict__ In17 = (char *__restrict__)Arg->In17;
    int W17 = (int)Arg->W17 * DataSize;
    char *__restrict__ In18 = (char *__restrict__)Arg->In18;
    int W18 = (int)Arg->W18 * DataSize;
    char *__restrict__ In19 = (char *__restrict__)Arg->In19;
    int W19 = (int)Arg->W19 * DataSize;
    char *__restrict__ In20 = (char *__restrict__)Arg->In20;
    int W20 = (int)Arg->W20 * DataSize;
    char *__restrict__ In21 = (char *__restrict__)Arg->In21;
    int W21 = (int)Arg->W21 * DataSize;
    char *__restrict__ In22 = (char *__restrict__)Arg->In22;
    int W22 = (int)Arg->W22 * DataSize;
    char *__restrict__ In23 = (char *__restrict__)Arg->In23;
    int W23 = (int)Arg->W23 * DataSize;
    char *__restrict__ In24 = (char *__restrict__)Arg->In24;
    int W24 = (int)Arg->W24 * DataSize;
    char *__restrict__ In25 = (char *__restrict__)Arg->In25;
    int W25 = (int)Arg->W25 * DataSize;
    int Wo = W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20 + W21 + W22 + W23 + W24 + W25;
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(H), First = Min(Chunk * CoreId, H), Last = Min(First + Chunk, H);
    for (int h = First; h < Last; h++)
    {
        Copy(Out + h * Wo, In1 + h * W1, W1);
        Copy(Out + h * Wo + W1, In2 + h * W2, W2);
        Copy(Out + h * Wo + W1 + W2, In3 + h * W3, W3);
        Copy(Out + h * Wo + W1 + W2 + W3, In4 + h * W4, W4);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4, In5 + h * W5, W5);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5, In6 + h * W6, W6);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6, In7 + h * W7, W7);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7, In8 + h * W8, W8);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8, In9 + h * W9, W9);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9, In10 + h * W10, W10);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10, In11 + h * W11, W11);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11, In12 + h * W12, W12);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12, In13 + h * W13, W13);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13, In14 + h * W14, W14);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14, In15 + h * W15, W15);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15, In16 + h * W16, W16);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16, In17 + h * W17, W17);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17, In18 + h * W18, W18);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18, In19 + h * W19, W19);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19, In20 + h * W20, W20);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20, In21 + h * W21, W21);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20 + W21, In22 + h * W22, W22);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20 + W21 + W22, In23 + h * W23, W23);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20 + W21 + W22 + W23, In24 + h * W24, W24);
        Copy(Out + h * Wo + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10 + W11 + W12 + W13 + W14 + W15 + W16 + W17 + W18 + W19 + W20 + W21 + W22 + W23 + W24, In25 + h * W25, W25);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s14_multiple_1_kernel(s14_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (1, 160, 160, 8) f16
    f16 *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (1, 160, 160, 8) f16
    f16 *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (1, 160, 160, 8) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 160, 160, 8) var shapes:
    // expr_1_out_0: (1, 160, 160, 8) expr_1_in_0: (1, 160, 160, 8)
    // expr_1_in_1: (1, 160, 160, 8)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_1_in_0: f16 expr_1_in_1: f16
        // expr_1_out_0 = Add(expr_1_in_0, expr_1_in_1)
        expr_1_out_0[i0] = (expr_1_in_0[i0]+expr_1_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s37_multiple_1_kernel(s37_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_7_in_0 = Args->expr_7_in_0; // (1, 80, 80, 16) f16
    f16 *__restrict__  expr_7_in_1 = Args->expr_7_in_1; // (1, 80, 80, 16) f16
    f16 *__restrict__  expr_7_out_0 = Args->expr_7_out_0; // (1, 80, 80, 16) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 80, 80, 16) var shapes:
    // expr_7_out_0: (1, 80, 80, 16) expr_7_in_0: (1, 80, 80, 16) expr_7_in_1:
    // (1, 80, 80, 16)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_7_in_0: f16 expr_7_in_1: f16
        // expr_7_out_0 = Add(expr_7_in_0, expr_7_in_1)
        expr_7_out_0[i0] = (expr_7_in_0[i0]+expr_7_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s60_multiple_1_kernel(s60_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_12_in_0 = Args->expr_12_in_0; // (40, 40, 32, 1) f16
    f16 *__restrict__  expr_12_in_1 = Args->expr_12_in_1; // (40, 40, 32, 1) f16
    f16 *__restrict__  expr_12_out_0 = Args->expr_12_out_0; // (40, 40, 32, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (40, 40, 32, 1) var shapes:
    // expr_12_out_0: (40, 40, 32, 1) expr_12_in_0: (40, 40, 32, 1)
    // expr_12_in_1: (40, 40, 32, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_12_in_0: f16 expr_12_in_1: f16
        // expr_12_out_0 = Add(expr_12_in_0, expr_12_in_1)
        expr_12_out_0[i0] = (expr_12_in_0[i0]+expr_12_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s387_multiple_1_kernel(s387_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    f16 *__restrict__  expr_15_in_0 = Args->expr_15_in_0; // (1, 1, 8, 400)  f16
    f16 *__restrict__  expr_15_in_1 = Args->expr_15_in_1; // (16, 1, 8, 400) f16
    f16 *__restrict__  expr_15_out_0 = Args->expr_15_out_0; // (16, 1, 8, 400) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 1, 8, 400) var shapes:
    // expr_15_out_0: (16, 1, 8, 400) expr_15_in_1: (16, 1, 8, 400)
    // expr_15_in_0: (1, 1, 8, 400)
    // Iteration reduced to spaces ((0,), (2, 3))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (2, 3))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_15_in_1: f16 expr_15_in_0: f16
            // expr_15_out_0 = Div(expr_15_in_1, Add(expr_15_in_0, [1.e-05]))
            expr_15_out_0[(i0*I1)+i1] = (expr_15_in_1[(i0*I1)+i1]/(expr_15_in_0[i1]+(f16)(1.000000e-05f)));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s393_multiple_1_kernel(s393_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_16_in_0 = Args->expr_16_in_0; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_16_in_1 = Args->expr_16_in_1; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_16_out_0 = Args->expr_16_out_0; // (20, 20, 64, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 64, 1) var shapes:
    // expr_16_out_0: (20, 20, 64, 1) expr_16_in_0: (20, 20, 64, 1)
    // expr_16_in_1: (20, 20, 64, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_16_in_0: f16 expr_16_in_1: f16
        // expr_16_out_0 = Add(expr_16_in_0, expr_16_in_1)
        expr_16_out_0[i0] = (expr_16_in_0[i0]+expr_16_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s399_multiple_1_kernel(s399_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_17_in_0 = Args->expr_17_in_0; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_17_in_1 = Args->expr_17_in_1; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_17_out_0 = Args->expr_17_out_0; // (20, 20, 64, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 64, 1) var shapes:
    // expr_17_out_0: (20, 20, 64, 1) expr_17_in_0: (20, 20, 64, 1)
    // expr_17_in_1: (20, 20, 64, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_17_in_0: f16 expr_17_in_1: f16
        // expr_17_out_0 = Add(expr_17_in_0, expr_17_in_1)
        expr_17_out_0[i0] = (expr_17_in_0[i0]+expr_17_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s487_multiple_1_kernel(s487_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_18_in_0 = Args->expr_18_in_0; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_18_in_1 = Args->expr_18_in_1; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_18_out_0 = Args->expr_18_out_0; // (20, 20, 64, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 64, 1) var shapes:
    // expr_18_out_0: (20, 20, 64, 1) expr_18_in_0: (20, 20, 64, 1)
    // expr_18_in_1: (20, 20, 64, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_18_in_0: f16 expr_18_in_1: f16
        // expr_18_out_0 = Add(expr_18_in_0, expr_18_in_1)
        expr_18_out_0[i0] = (expr_18_in_0[i0]+expr_18_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s493_multiple_1_kernel(s493_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_19_in_0 = Args->expr_19_in_0; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_19_in_1 = Args->expr_19_in_1; // (20, 20, 64, 1) f16
    f16 *__restrict__  expr_19_out_0 = Args->expr_19_out_0; // (20, 20, 64, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 64, 1) var shapes:
    // expr_19_out_0: (20, 20, 64, 1) expr_19_in_0: (20, 20, 64, 1)
    // expr_19_in_1: (20, 20, 64, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_19_in_0: f16 expr_19_in_1: f16
        // expr_19_out_0 = Add(expr_19_in_0, expr_19_in_1)
        expr_19_out_0[i0] = (expr_19_in_0[i0]+expr_19_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s481_multiple_1_kernel(s481_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    f16 *__restrict__  expr_22_in_0 = Args->expr_22_in_0; // (1, 1, 8, 400)  f16
    f16 *__restrict__  expr_22_in_1 = Args->expr_22_in_1; // (16, 1, 8, 400) f16
    f16 *__restrict__  expr_22_out_0 = Args->expr_22_out_0; // (16, 1, 8, 400) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 1, 8, 400) var shapes:
    // expr_22_out_0: (16, 1, 8, 400) expr_22_in_1: (16, 1, 8, 400)
    // expr_22_in_0: (1, 1, 8, 400)
    // Iteration reduced to spaces ((0,), (2, 3))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (2, 3))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_22_in_1: f16 expr_22_in_0: f16
            // expr_22_out_0 = Div(expr_22_in_1, Add(expr_22_in_0, [1.e-05]))
            expr_22_out_0[(i0*I1)+i1] = (expr_22_in_1[(i0*I1)+i1]/(expr_22_in_0[i1]+(f16)(1.000000e-05f)));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s640_multiple_1_kernel(s640_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    f16 *__restrict__  expr_27_in_0 = Args->expr_27_in_0; // (1, 1, 16, 100)  f16
    f16 *__restrict__  expr_27_in_1 = Args->expr_27_in_1; // (16, 1, 16, 100) f16
    f16 *__restrict__  expr_27_out_0 = Args->expr_27_out_0; // (16, 1, 16, 100) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 1, 16, 100) var shapes:
    // expr_27_out_0: (16, 1, 16, 100) expr_27_in_1: (16, 1, 16, 100)
    // expr_27_in_0: (1, 1, 16, 100)
    // Iteration reduced to spaces ((0,), (2, 3))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (2, 3))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_27_in_1: f16 expr_27_in_0: f16
            // expr_27_out_0 = Div(expr_27_in_1, Add(expr_27_in_0, [1.e-05]))
            expr_27_out_0[(i0*I1)+i1] = (expr_27_in_1[(i0*I1)+i1]/(expr_27_in_0[i1]+(f16)(1.000000e-05f)));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s646_multiple_1_kernel(s646_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_28_in_0 = Args->expr_28_in_0; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_28_in_1 = Args->expr_28_in_1; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_28_out_0 = Args->expr_28_out_0; // (1, 10, 10, 128) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 10, 10, 128) var shapes:
    // expr_28_out_0: (1, 10, 10, 128) expr_28_in_0: (1, 10, 10, 128)
    // expr_28_in_1: (1, 10, 10, 128)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_28_in_0: f16 expr_28_in_1: f16
        // expr_28_out_0 = Add(expr_28_in_0, expr_28_in_1)
        expr_28_out_0[i0] = (expr_28_in_0[i0]+expr_28_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s652_multiple_1_kernel(s652_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_29_in_0 = Args->expr_29_in_0; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_29_in_1 = Args->expr_29_in_1; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_29_out_0 = Args->expr_29_out_0; // (1, 10, 10, 128) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 10, 10, 128) var shapes:
    // expr_29_out_0: (1, 10, 10, 128) expr_29_in_0: (1, 10, 10, 128)
    // expr_29_in_1: (1, 10, 10, 128)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_29_in_0: f16 expr_29_in_1: f16
        // expr_29_out_0 = Add(expr_29_in_0, expr_29_in_1)
        expr_29_out_0[i0] = (expr_29_in_0[i0]+expr_29_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s800_multiple_1_kernel(s800_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_30_in_0 = Args->expr_30_in_0; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_30_in_1 = Args->expr_30_in_1; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_30_out_0 = Args->expr_30_out_0; // (1, 10, 10, 128) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 10, 10, 128) var shapes:
    // expr_30_out_0: (1, 10, 10, 128) expr_30_in_0: (1, 10, 10, 128)
    // expr_30_in_1: (1, 10, 10, 128)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_30_in_0: f16 expr_30_in_1: f16
        // expr_30_out_0 = Add(expr_30_in_0, expr_30_in_1)
        expr_30_out_0[i0] = (expr_30_in_0[i0]+expr_30_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s806_multiple_1_kernel(s806_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_31_in_0 = Args->expr_31_in_0; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_31_in_1 = Args->expr_31_in_1; // (1, 10, 10, 128) f16
    f16 *__restrict__  expr_31_out_0 = Args->expr_31_out_0; // (1, 10, 10, 128) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 10, 10, 128) var shapes:
    // expr_31_out_0: (1, 10, 10, 128) expr_31_in_0: (1, 10, 10, 128)
    // expr_31_in_1: (1, 10, 10, 128)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_31_in_0: f16 expr_31_in_1: f16
        // expr_31_out_0 = Add(expr_31_in_0, expr_31_in_1)
        expr_31_out_0[i0] = (expr_31_in_0[i0]+expr_31_in_1[i0]);
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s794_multiple_1_kernel(s794_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    f16 *__restrict__  expr_34_in_0 = Args->expr_34_in_0; // (1, 1, 16, 100)  f16
    f16 *__restrict__  expr_34_in_1 = Args->expr_34_in_1; // (16, 1, 16, 100) f16
    f16 *__restrict__  expr_34_out_0 = Args->expr_34_out_0; // (16, 1, 16, 100) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 1, 16, 100) var shapes:
    // expr_34_out_0: (16, 1, 16, 100) expr_34_in_1: (16, 1, 16, 100)
    // expr_34_in_0: (1, 1, 16, 100)
    // Iteration reduced to spaces ((0,), (2, 3))
    // Fixed spaces ()
    // Parameteric spaces ((0,), (2, 3))
    // Paralelized space (0,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_34_in_1: f16 expr_34_in_0: f16
            // expr_34_out_0 = Div(expr_34_in_1, Add(expr_34_in_0, [1.e-05]))
            expr_34_out_0[(i0*I1)+i1] = (expr_34_in_1[(i0*I1)+i1]/(expr_34_in_0[i1]+(f16)(1.000000e-05f)));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s818_multiple_1_kernel(s818_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_38_in_0 = Args->expr_38_in_0; // (1, 20, 20, 32) f16
    f16 *__restrict__  expr_38_out_0 = Args->expr_38_out_0; // (1, 20, 20, 32) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 20, 20, 32) var shapes:
    // expr_38_out_0: (1, 20, 20, 32) expr_38_in_0: (1, 20, 20, 32)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_38_in_0: f16
        // expr_38_out_0 = Mul(expr_38_in_0, FastFloatSigmoid(expr_38_in_0))
        expr_38_out_0[i0] = (expr_38_in_0[i0]*fastsigmoid_f16(expr_38_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s831_multiple_1_kernel(s831_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_48_in_0 = Args->expr_48_in_0; // (20, 20, 32, 1) f16
    f16 *__restrict__  expr_48_out_0 = Args->expr_48_out_0; // (20, 20, 32, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 32, 1) var shapes:
    // expr_48_out_0: (20, 20, 32, 1) expr_48_in_0: (20, 20, 32, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_48_in_0: f16
        // expr_48_out_0 = Mul(expr_48_in_0, FastFloatSigmoid(expr_48_in_0))
        expr_48_out_0[i0] = (expr_48_in_0[i0]*fastsigmoid_f16(expr_48_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s845_multiple_1_kernel(s845_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_51_in_0 = Args->expr_51_in_0; // (1, 40, 40, 16) f16
    f16 *__restrict__  expr_51_out_0 = Args->expr_51_out_0; // (1, 40, 40, 16) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 40, 40, 16) var shapes:
    // expr_51_out_0: (1, 40, 40, 16) expr_51_in_0: (1, 40, 40, 16)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_51_in_0: f16
        // expr_51_out_0 = Mul(expr_51_in_0, FastFloatSigmoid(expr_51_in_0))
        expr_51_out_0[i0] = (expr_51_in_0[i0]*fastsigmoid_f16(expr_51_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s858_multiple_1_kernel(s858_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_61_in_0 = Args->expr_61_in_0; // (40, 40, 16, 1) f16
    f16 *__restrict__  expr_61_out_0 = Args->expr_61_out_0; // (40, 40, 16, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (40, 40, 16, 1) var shapes:
    // expr_61_out_0: (40, 40, 16, 1) expr_61_in_0: (40, 40, 16, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_61_in_0: f16
        // expr_61_out_0 = Mul(expr_61_in_0, FastFloatSigmoid(expr_61_in_0))
        expr_61_out_0[i0] = (expr_61_in_0[i0]*fastsigmoid_f16(expr_61_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s887_multiple_1_kernel(s887_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_74_in_0 = Args->expr_74_in_0; // (1, 20, 20, 32) f16
    f16 *__restrict__  expr_74_out_0 = Args->expr_74_out_0; // (1, 20, 20, 32) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 20, 20, 32) var shapes:
    // expr_74_out_0: (1, 20, 20, 32) expr_74_in_0: (1, 20, 20, 32)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_74_in_0: f16
        // expr_74_out_0 = Mul(expr_74_in_0, FastFloatSigmoid(expr_74_in_0))
        expr_74_out_0[i0] = (expr_74_in_0[i0]*fastsigmoid_f16(expr_74_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s900_multiple_1_kernel(s900_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_84_in_0 = Args->expr_84_in_0; // (20, 20, 32, 1) f16
    f16 *__restrict__  expr_84_out_0 = Args->expr_84_out_0; // (20, 20, 32, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (20, 20, 32, 1) var shapes:
    // expr_84_out_0: (20, 20, 32, 1) expr_84_in_0: (20, 20, 32, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_84_in_0: f16
        // expr_84_out_0 = Mul(expr_84_in_0, FastFloatSigmoid(expr_84_in_0))
        expr_84_out_0[i0] = (expr_84_in_0[i0]*fastsigmoid_f16(expr_84_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s929_multiple_1_kernel(s929_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_97_in_0 = Args->expr_97_in_0; // (1, 10, 10, 64) f16
    f16 *__restrict__  expr_97_out_0 = Args->expr_97_out_0; // (1, 10, 10, 64) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 10, 10, 64) var shapes:
    // expr_97_out_0: (1, 10, 10, 64) expr_97_in_0: (1, 10, 10, 64)
    // Iteration reduced to spaces ((1, 2, 3),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2, 3),)
    // Paralelized space (1, 2, 3)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_97_in_0: f16
        // expr_97_out_0 = Mul(expr_97_in_0, FastFloatSigmoid(expr_97_in_0))
        expr_97_out_0[i0] = (expr_97_in_0[i0]*fastsigmoid_f16(expr_97_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s942_multiple_1_kernel(s942_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    f16 *__restrict__  expr_107_in_0 = Args->expr_107_in_0; // (10, 10, 64, 1) f16
    f16 *__restrict__  expr_107_out_0 = Args->expr_107_out_0; // (10, 10, 64, 1) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (10, 10, 64, 1) var shapes:
    // expr_107_out_0: (10, 10, 64, 1) expr_107_in_0: (10, 10, 64, 1)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_107_in_0: f16
        // expr_107_out_0 = Mul(expr_107_in_0, FastFloatSigmoid(expr_107_in_0))
        expr_107_out_0[i0] = (expr_107_in_0[i0]*fastsigmoid_f16(expr_107_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s965_multiple_1_kernel(s965_multiple_1_kernel_args_t * __restrict__ Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    f16 *__restrict__  expr_118_in_0 = Args->expr_118_in_0; // (1, 2100, 1)  f16
    f16 *__restrict__  expr_118_in_1 = Args->expr_118_in_1; // (1, 2100, 80) f16
    f16 *__restrict__  expr_118_out_0 = Args->expr_118_out_0; // (1, 2100, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 2100, 80) var shapes:
    // expr_118_out_0: (1, 2100, 80) expr_118_in_0: (1, 2100, 1) expr_118_in_1:
    // (1, 2100, 80)
    // Iteration reduced to spaces ((1,), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((1,), (2,))
    // Paralelized space (1,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        for (int i1=0; i1<I1; i1++) {
            // inputs expr_118_in_0: f16 expr_118_in_1: f16
            // expr_118_out_0 = Mul(expr_118_in_0, expr_118_in_1)
            expr_118_out_0[(i0*I1)+i1] = (expr_118_in_0[i0]*expr_118_in_1[(i0*I1)+i1]);
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_0multiple_1(expr_0multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_0_in_0 = Args->expr_0_in_0;
    f16 *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (8, 160, 160) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 160, 160) var shapes:
    // expr_0_out_0: (8, 160, 160) expr_0_in_0: (8, 160, 160)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_0_in_0: f16
        // expr_0_out_0 = Mul(expr_0_in_0, Mul(Min([6.], Max([0.], Add(expr_0_in_0, [3.]))), [1]/[6]))
        expr_0_out_0[i0] = (expr_0_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_0_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2multiple_1(expr_2multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    f16 *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (8, 160, 160) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 160, 160) var shapes:
    // expr_2_out_0: (8, 160, 160) expr_2_in_0: (8, 160, 160)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: f16
        // expr_2_out_0 = Mul(expr_2_in_0, Mul(Min([6.], Max([0.], Add(expr_2_in_0, [3.]))), [1]/[6]))
        expr_2_out_0[i0] = (expr_2_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_2_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_3multiple_1(expr_3multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_3_in_0 = Args->expr_3_in_0;
    f16 *__restrict__  expr_3_out_0 = Args->expr_3_out_0; // (32, 160, 160) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 160, 160) var shapes:
    // expr_3_out_0: (32, 160, 160) expr_3_in_0: (32, 160, 160)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_3_in_0: f16
        // expr_3_out_0 = Mul(expr_3_in_0, Mul(Min([6.], Max([0.], Add(expr_3_in_0, [3.]))), [1]/[6]))
        expr_3_out_0[i0] = (expr_3_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_3_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_4multiple_1(expr_4multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_4_in_0 = Args->expr_4_in_0;
    f16 *__restrict__  expr_4_out_0 = Args->expr_4_out_0; // (32, 80, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 80, 80) var shapes:
    // expr_4_out_0: (32, 80, 80) expr_4_in_0: (32, 80, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_4_in_0: f16
        // expr_4_out_0 = Mul(expr_4_in_0, Mul(Min([6.], Max([0.], Add(expr_4_in_0, [3.]))), [1]/[6]))
        expr_4_out_0[i0] = (expr_4_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_4_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_5multiple_1(expr_5multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_5_in_0 = Args->expr_5_in_0;
    f16 *__restrict__  expr_5_out_0 = Args->expr_5_out_0; // (64, 80, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 80, 80) var shapes:
    // expr_5_out_0: (64, 80, 80) expr_5_in_0: (64, 80, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_5_in_0: f16
        // expr_5_out_0 = Mul(expr_5_in_0, Mul(Min([6.], Max([0.], Add(expr_5_in_0, [3.]))), [1]/[6]))
        expr_5_out_0[i0] = (expr_5_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_5_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_6multiple_1(expr_6multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_6_in_0 = Args->expr_6_in_0;
    f16 *__restrict__  expr_6_out_0 = Args->expr_6_out_0; // (64, 80, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 80, 80) var shapes:
    // expr_6_out_0: (64, 80, 80) expr_6_in_0: (64, 80, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_6_in_0: f16
        // expr_6_out_0 = Mul(expr_6_in_0, Mul(Min([6.], Max([0.], Add(expr_6_in_0, [3.]))), [1]/[6]))
        expr_6_out_0[i0] = (expr_6_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_6_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_8multiple_1(expr_8multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_8_in_0 = Args->expr_8_in_0;
    f16 *__restrict__  expr_8_out_0 = Args->expr_8_out_0; // (64, 80, 80) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 80, 80) var shapes:
    // expr_8_out_0: (64, 80, 80) expr_8_in_0: (64, 80, 80)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_8_in_0: f16
        // expr_8_out_0 = Mul(expr_8_in_0, Mul(Min([6.], Max([0.], Add(expr_8_in_0, [3.]))), [1]/[6]))
        expr_8_out_0[i0] = (expr_8_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_8_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_9multiple_1(expr_9multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_9_in_0 = Args->expr_9_in_0;
    f16 *__restrict__  expr_9_out_0 = Args->expr_9_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_9_out_0: (64, 40, 40) expr_9_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_9_in_0: f16
        // expr_9_out_0 = Mul(expr_9_in_0, Mul(Min([6.], Max([0.], Add(expr_9_in_0, [3.]))), [1]/[6]))
        expr_9_out_0[i0] = (expr_9_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_9_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_10multiple_1(expr_10multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_10_in_0 = Args->expr_10_in_0;
    f16 *__restrict__  expr_10_out_0 = Args->expr_10_out_0; // (128, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 40, 40) var shapes:
    // expr_10_out_0: (128, 40, 40) expr_10_in_0: (128, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_10_in_0: f16
        // expr_10_out_0 = Mul(expr_10_in_0, Mul(Min([6.], Max([0.], Add(expr_10_in_0, [3.]))), [1]/[6]))
        expr_10_out_0[i0] = (expr_10_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_10_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_11multiple_1(expr_11multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_11_in_0 = Args->expr_11_in_0;
    f16 *__restrict__  expr_11_out_0 = Args->expr_11_out_0; // (128, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 40, 40) var shapes:
    // expr_11_out_0: (128, 40, 40) expr_11_in_0: (128, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_11_in_0: f16
        // expr_11_out_0 = Mul(expr_11_in_0, Mul(Min([6.], Max([0.], Add(expr_11_in_0, [3.]))), [1]/[6]))
        expr_11_out_0[i0] = (expr_11_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_11_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_13multiple_1(expr_13multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_13_in_0 = Args->expr_13_in_0;
    f16 *__restrict__  expr_13_out_0 = Args->expr_13_out_0; // (128, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 40, 40) var shapes:
    // expr_13_out_0: (128, 40, 40) expr_13_in_0: (128, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_13_in_0: f16
        // expr_13_out_0 = Mul(expr_13_in_0, Mul(Min([6.], Max([0.], Add(expr_13_in_0, [3.]))), [1]/[6]))
        expr_13_out_0[i0] = (expr_13_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_13_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_14multiple_1(expr_14multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_14_in_0 = Args->expr_14_in_0;
    f16 *__restrict__  expr_14_out_0 = Args->expr_14_out_0; // (128, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 20, 20) var shapes:
    // expr_14_out_0: (128, 20, 20) expr_14_in_0: (128, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_14_in_0: f16
        // expr_14_out_0 = Mul(expr_14_in_0, Mul(Min([6.], Max([0.], Add(expr_14_in_0, [3.]))), [1]/[6]))
        expr_14_out_0[i0] = (expr_14_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_14_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_20multiple_1(expr_20multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_20_in_0 = Args->expr_20_in_0;
    f16 *__restrict__  expr_20_out_0 = Args->expr_20_out_0; // (256, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 20, 20) var shapes:
    // expr_20_out_0: (256, 20, 20) expr_20_in_0: (256, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_20_in_0: f16
        // expr_20_out_0 = Mul(expr_20_in_0, Mul(Min([6.], Max([0.], Add(expr_20_in_0, [3.]))), [1]/[6]))
        expr_20_out_0[i0] = (expr_20_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_20_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_21multiple_1(expr_21multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_21_in_0 = Args->expr_21_in_0;
    f16 *__restrict__  expr_21_out_0 = Args->expr_21_out_0; // (256, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 20, 20) var shapes:
    // expr_21_out_0: (256, 20, 20) expr_21_in_0: (256, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_21_in_0: f16
        // expr_21_out_0 = Mul(expr_21_in_0, Mul(Min([6.], Max([0.], Add(expr_21_in_0, [3.]))), [1]/[6]))
        expr_21_out_0[i0] = (expr_21_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_21_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_23multiple_1(expr_23multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_23_in_0 = Args->expr_23_in_0;
    f16 *__restrict__  expr_23_out_0 = Args->expr_23_out_0; // (256, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 20, 20) var shapes:
    // expr_23_out_0: (256, 20, 20) expr_23_in_0: (256, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_23_in_0: f16
        // expr_23_out_0 = Mul(expr_23_in_0, Mul(Min([6.], Max([0.], Add(expr_23_in_0, [3.]))), [1]/[6]))
        expr_23_out_0[i0] = (expr_23_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_23_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_24multiple_1(expr_24multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_24_in_0 = Args->expr_24_in_0;
    f16 *__restrict__  expr_24_out_0 = Args->expr_24_out_0; // (256, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 20, 20) var shapes:
    // expr_24_out_0: (256, 20, 20) expr_24_in_0: (256, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_24_in_0: f16
        // expr_24_out_0 = Mul(expr_24_in_0, Mul(Min([6.], Max([0.], Add(expr_24_in_0, [3.]))), [1]/[6]))
        expr_24_out_0[i0] = (expr_24_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_24_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_25multiple_1(expr_25multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_25_in_0 = Args->expr_25_in_0;
    f16 *__restrict__  expr_25_out_0 = Args->expr_25_out_0; // (256, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 20, 20) var shapes:
    // expr_25_out_0: (256, 20, 20) expr_25_in_0: (256, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_25_in_0: f16
        // expr_25_out_0 = Mul(expr_25_in_0, Mul(Min([6.], Max([0.], Add(expr_25_in_0, [3.]))), [1]/[6]))
        expr_25_out_0[i0] = (expr_25_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_25_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_26multiple_1(expr_26multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_26_in_0 = Args->expr_26_in_0;
    f16 *__restrict__  expr_26_out_0 = Args->expr_26_out_0; // (256, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (256, 10, 10) var shapes:
    // expr_26_out_0: (256, 10, 10) expr_26_in_0: (256, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_26_in_0: f16
        // expr_26_out_0 = Mul(expr_26_in_0, Mul(Min([6.], Max([0.], Add(expr_26_in_0, [3.]))), [1]/[6]))
        expr_26_out_0[i0] = (expr_26_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_26_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_32multiple_1(expr_32multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_32_in_0 = Args->expr_32_in_0;
    f16 *__restrict__  expr_32_out_0 = Args->expr_32_out_0; // (512, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (512, 10, 10) var shapes:
    // expr_32_out_0: (512, 10, 10) expr_32_in_0: (512, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_32_in_0: f16
        // expr_32_out_0 = Mul(expr_32_in_0, Mul(Min([6.], Max([0.], Add(expr_32_in_0, [3.]))), [1]/[6]))
        expr_32_out_0[i0] = (expr_32_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_32_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_33multiple_1(expr_33multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_33_in_0 = Args->expr_33_in_0;
    f16 *__restrict__  expr_33_out_0 = Args->expr_33_out_0; // (512, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (512, 10, 10) var shapes:
    // expr_33_out_0: (512, 10, 10) expr_33_in_0: (512, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_33_in_0: f16
        // expr_33_out_0 = Mul(expr_33_in_0, Mul(Min([6.], Max([0.], Add(expr_33_in_0, [3.]))), [1]/[6]))
        expr_33_out_0[i0] = (expr_33_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_33_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_35multiple_1(expr_35multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_35_in_0 = Args->expr_35_in_0;
    f16 *__restrict__  expr_35_out_0 = Args->expr_35_out_0; // (512, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (512, 10, 10) var shapes:
    // expr_35_out_0: (512, 10, 10) expr_35_in_0: (512, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_35_in_0: f16
        // expr_35_out_0 = Mul(expr_35_in_0, Mul(Min([6.], Max([0.], Add(expr_35_in_0, [3.]))), [1]/[6]))
        expr_35_out_0[i0] = (expr_35_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_35_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_36multiple_1(expr_36multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_36_in_0 = Args->expr_36_in_0;
    f16 *__restrict__  expr_36_out_0 = Args->expr_36_out_0; // (512, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (512, 10, 10) var shapes:
    // expr_36_out_0: (512, 10, 10) expr_36_in_0: (512, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_36_in_0: f16
        // expr_36_out_0 = Mul(expr_36_in_0, Mul(Min([6.], Max([0.], Add(expr_36_in_0, [3.]))), [1]/[6]))
        expr_36_out_0[i0] = (expr_36_in_0[i0]*(Minf32(((f16)(6.0f)),(Maxf32(((f16)(0.0f)),((expr_36_in_0[i0]+(f16)(3.0f))))))*1/6));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_37multiple_1(expr_37multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_37_in_0 = Args->expr_37_in_0;
    f16 *__restrict__  expr_37_out_0 = Args->expr_37_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_37_out_0: (64, 10, 10) expr_37_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_37_in_0: f16
        // expr_37_out_0 = Mul(expr_37_in_0, FastFloatSigmoid(expr_37_in_0))
        expr_37_out_0[i0] = (expr_37_in_0[i0]*fastsigmoid_f16(expr_37_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_39multiple_1(expr_39multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_39_in_0 = Args->expr_39_in_0;
    f16 *__restrict__  expr_39_out_0 = Args->expr_39_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_39_out_0: (32, 20, 20) expr_39_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_39_in_0: f16
        // expr_39_out_0 = Mul(expr_39_in_0, FastFloatSigmoid(expr_39_in_0))
        expr_39_out_0[i0] = (expr_39_in_0[i0]*fastsigmoid_f16(expr_39_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_40multiple_1(expr_40multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_40_in_0 = Args->expr_40_in_0;
    f16 *__restrict__  expr_40_out_0 = Args->expr_40_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_40_out_0: (32, 20, 20) expr_40_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_40_in_0: f16
        // expr_40_out_0 = Mul(expr_40_in_0, FastFloatSigmoid(expr_40_in_0))
        expr_40_out_0[i0] = (expr_40_in_0[i0]*fastsigmoid_f16(expr_40_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_41multiple_1(expr_41multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_41_in_0 = Args->expr_41_in_0;
    f16 *__restrict__  expr_41_out_0 = Args->expr_41_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_41_out_0: (32, 20, 20) expr_41_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_41_in_0: f16
        // expr_41_out_0 = Mul(expr_41_in_0, FastFloatSigmoid(expr_41_in_0))
        expr_41_out_0[i0] = (expr_41_in_0[i0]*fastsigmoid_f16(expr_41_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_42multiple_1(expr_42multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_42_in_0 = Args->expr_42_in_0;
    f16 *__restrict__  expr_42_out_0 = Args->expr_42_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_42_out_0: (32, 20, 20) expr_42_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_42_in_0: f16
        // expr_42_out_0 = Mul(expr_42_in_0, FastFloatSigmoid(expr_42_in_0))
        expr_42_out_0[i0] = (expr_42_in_0[i0]*fastsigmoid_f16(expr_42_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_43multiple_1(expr_43multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_43_in_0 = Args->expr_43_in_0;
    f16 *__restrict__  expr_43_out_0 = Args->expr_43_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_43_out_0: (32, 20, 20) expr_43_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_43_in_0: f16
        // expr_43_out_0 = Mul(expr_43_in_0, FastFloatSigmoid(expr_43_in_0))
        expr_43_out_0[i0] = (expr_43_in_0[i0]*fastsigmoid_f16(expr_43_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_44multiple_1(expr_44multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_44_in_0 = Args->expr_44_in_0;
    f16 *__restrict__  expr_44_out_0 = Args->expr_44_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_44_out_0: (32, 20, 20) expr_44_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_44_in_0: f16
        // expr_44_out_0 = Mul(expr_44_in_0, FastFloatSigmoid(expr_44_in_0))
        expr_44_out_0[i0] = (expr_44_in_0[i0]*fastsigmoid_f16(expr_44_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_45multiple_1(expr_45multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_45_in_0 = Args->expr_45_in_0;
    f16 *__restrict__  expr_45_out_0 = Args->expr_45_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_45_out_0: (32, 20, 20) expr_45_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_45_in_0: f16
        // expr_45_out_0 = Mul(expr_45_in_0, FastFloatSigmoid(expr_45_in_0))
        expr_45_out_0[i0] = (expr_45_in_0[i0]*fastsigmoid_f16(expr_45_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_46multiple_1(expr_46multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_46_in_0 = Args->expr_46_in_0;
    f16 *__restrict__  expr_46_out_0 = Args->expr_46_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_46_out_0: (32, 20, 20) expr_46_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_46_in_0: f16
        // expr_46_out_0 = Mul(expr_46_in_0, FastFloatSigmoid(expr_46_in_0))
        expr_46_out_0[i0] = (expr_46_in_0[i0]*fastsigmoid_f16(expr_46_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_47multiple_1(expr_47multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_47_in_0 = Args->expr_47_in_0;
    f16 *__restrict__  expr_47_out_0 = Args->expr_47_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_47_out_0: (32, 20, 20) expr_47_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_47_in_0: f16
        // expr_47_out_0 = Mul(expr_47_in_0, FastFloatSigmoid(expr_47_in_0))
        expr_47_out_0[i0] = (expr_47_in_0[i0]*fastsigmoid_f16(expr_47_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_49multiple_1(expr_49multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_49_in_0 = Args->expr_49_in_0;
    f16 *__restrict__  expr_49_out_0 = Args->expr_49_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_49_out_0: (64, 20, 20) expr_49_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_49_in_0: f16
        // expr_49_out_0 = Mul(expr_49_in_0, FastFloatSigmoid(expr_49_in_0))
        expr_49_out_0[i0] = (expr_49_in_0[i0]*fastsigmoid_f16(expr_49_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_50multiple_1(expr_50multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_50_in_0 = Args->expr_50_in_0;
    f16 *__restrict__  expr_50_out_0 = Args->expr_50_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_50_out_0: (32, 20, 20) expr_50_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_50_in_0: f16
        // expr_50_out_0 = Mul(expr_50_in_0, FastFloatSigmoid(expr_50_in_0))
        expr_50_out_0[i0] = (expr_50_in_0[i0]*fastsigmoid_f16(expr_50_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_52multiple_1(expr_52multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_52_in_0 = Args->expr_52_in_0;
    f16 *__restrict__  expr_52_out_0 = Args->expr_52_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_52_out_0: (16, 40, 40) expr_52_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_52_in_0: f16
        // expr_52_out_0 = Mul(expr_52_in_0, FastFloatSigmoid(expr_52_in_0))
        expr_52_out_0[i0] = (expr_52_in_0[i0]*fastsigmoid_f16(expr_52_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_53multiple_1(expr_53multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_53_in_0 = Args->expr_53_in_0;
    f16 *__restrict__  expr_53_out_0 = Args->expr_53_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_53_out_0: (16, 40, 40) expr_53_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_53_in_0: f16
        // expr_53_out_0 = Mul(expr_53_in_0, FastFloatSigmoid(expr_53_in_0))
        expr_53_out_0[i0] = (expr_53_in_0[i0]*fastsigmoid_f16(expr_53_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_54multiple_1(expr_54multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_54_in_0 = Args->expr_54_in_0;
    f16 *__restrict__  expr_54_out_0 = Args->expr_54_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_54_out_0: (16, 40, 40) expr_54_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_54_in_0: f16
        // expr_54_out_0 = Mul(expr_54_in_0, FastFloatSigmoid(expr_54_in_0))
        expr_54_out_0[i0] = (expr_54_in_0[i0]*fastsigmoid_f16(expr_54_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_55multiple_1(expr_55multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_55_in_0 = Args->expr_55_in_0;
    f16 *__restrict__  expr_55_out_0 = Args->expr_55_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_55_out_0: (16, 40, 40) expr_55_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_55_in_0: f16
        // expr_55_out_0 = Mul(expr_55_in_0, FastFloatSigmoid(expr_55_in_0))
        expr_55_out_0[i0] = (expr_55_in_0[i0]*fastsigmoid_f16(expr_55_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_56multiple_1(expr_56multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_56_in_0 = Args->expr_56_in_0;
    f16 *__restrict__  expr_56_out_0 = Args->expr_56_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_56_out_0: (16, 40, 40) expr_56_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_56_in_0: f16
        // expr_56_out_0 = Mul(expr_56_in_0, FastFloatSigmoid(expr_56_in_0))
        expr_56_out_0[i0] = (expr_56_in_0[i0]*fastsigmoid_f16(expr_56_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_57multiple_1(expr_57multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_57_in_0 = Args->expr_57_in_0;
    f16 *__restrict__  expr_57_out_0 = Args->expr_57_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_57_out_0: (16, 40, 40) expr_57_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_57_in_0: f16
        // expr_57_out_0 = Mul(expr_57_in_0, FastFloatSigmoid(expr_57_in_0))
        expr_57_out_0[i0] = (expr_57_in_0[i0]*fastsigmoid_f16(expr_57_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_58multiple_1(expr_58multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_58_in_0 = Args->expr_58_in_0;
    f16 *__restrict__  expr_58_out_0 = Args->expr_58_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_58_out_0: (16, 40, 40) expr_58_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_58_in_0: f16
        // expr_58_out_0 = Mul(expr_58_in_0, FastFloatSigmoid(expr_58_in_0))
        expr_58_out_0[i0] = (expr_58_in_0[i0]*fastsigmoid_f16(expr_58_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_59multiple_1(expr_59multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_59_in_0 = Args->expr_59_in_0;
    f16 *__restrict__  expr_59_out_0 = Args->expr_59_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_59_out_0: (16, 40, 40) expr_59_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_59_in_0: f16
        // expr_59_out_0 = Mul(expr_59_in_0, FastFloatSigmoid(expr_59_in_0))
        expr_59_out_0[i0] = (expr_59_in_0[i0]*fastsigmoid_f16(expr_59_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_60multiple_1(expr_60multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_60_in_0 = Args->expr_60_in_0;
    f16 *__restrict__  expr_60_out_0 = Args->expr_60_out_0; // (16, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 40, 40) var shapes:
    // expr_60_out_0: (16, 40, 40) expr_60_in_0: (16, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_60_in_0: f16
        // expr_60_out_0 = Mul(expr_60_in_0, FastFloatSigmoid(expr_60_in_0))
        expr_60_out_0[i0] = (expr_60_in_0[i0]*fastsigmoid_f16(expr_60_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_62multiple_1(expr_62multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_62_in_0 = Args->expr_62_in_0;
    f16 *__restrict__  expr_62_out_0 = Args->expr_62_out_0; // (32, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 40, 40) var shapes:
    // expr_62_out_0: (32, 40, 40) expr_62_in_0: (32, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_62_in_0: f16
        // expr_62_out_0 = Mul(expr_62_in_0, FastFloatSigmoid(expr_62_in_0))
        expr_62_out_0[i0] = (expr_62_in_0[i0]*fastsigmoid_f16(expr_62_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_63multiple_1(expr_63multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_63_in_0 = Args->expr_63_in_0;
    f16 *__restrict__  expr_63_out_0 = Args->expr_63_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_63_out_0: (32, 20, 20) expr_63_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_63_in_0: f16
        // expr_63_out_0 = Mul(expr_63_in_0, FastFloatSigmoid(expr_63_in_0))
        expr_63_out_0[i0] = (expr_63_in_0[i0]*fastsigmoid_f16(expr_63_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_64multiple_1(expr_64multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_64_in_0 = Args->expr_64_in_0;
    f16 *__restrict__  expr_64_out_0 = Args->expr_64_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_64_out_0: (32, 20, 20) expr_64_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_64_in_0: f16
        // expr_64_out_0 = Mul(expr_64_in_0, FastFloatSigmoid(expr_64_in_0))
        expr_64_out_0[i0] = (expr_64_in_0[i0]*fastsigmoid_f16(expr_64_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_75multiple_1(expr_75multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_75_in_0 = Args->expr_75_in_0;
    f16 *__restrict__  expr_75_out_0 = Args->expr_75_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_75_out_0: (32, 20, 20) expr_75_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_75_in_0: f16
        // expr_75_out_0 = Mul(expr_75_in_0, FastFloatSigmoid(expr_75_in_0))
        expr_75_out_0[i0] = (expr_75_in_0[i0]*fastsigmoid_f16(expr_75_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_76multiple_1(expr_76multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_76_in_0 = Args->expr_76_in_0;
    f16 *__restrict__  expr_76_out_0 = Args->expr_76_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_76_out_0: (32, 20, 20) expr_76_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_76_in_0: f16
        // expr_76_out_0 = Mul(expr_76_in_0, FastFloatSigmoid(expr_76_in_0))
        expr_76_out_0[i0] = (expr_76_in_0[i0]*fastsigmoid_f16(expr_76_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_77multiple_1(expr_77multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_77_in_0 = Args->expr_77_in_0;
    f16 *__restrict__  expr_77_out_0 = Args->expr_77_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_77_out_0: (32, 20, 20) expr_77_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_77_in_0: f16
        // expr_77_out_0 = Mul(expr_77_in_0, FastFloatSigmoid(expr_77_in_0))
        expr_77_out_0[i0] = (expr_77_in_0[i0]*fastsigmoid_f16(expr_77_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_78multiple_1(expr_78multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_78_in_0 = Args->expr_78_in_0;
    f16 *__restrict__  expr_78_out_0 = Args->expr_78_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_78_out_0: (32, 20, 20) expr_78_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_78_in_0: f16
        // expr_78_out_0 = Mul(expr_78_in_0, FastFloatSigmoid(expr_78_in_0))
        expr_78_out_0[i0] = (expr_78_in_0[i0]*fastsigmoid_f16(expr_78_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_79multiple_1(expr_79multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_79_in_0 = Args->expr_79_in_0;
    f16 *__restrict__  expr_79_out_0 = Args->expr_79_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_79_out_0: (32, 20, 20) expr_79_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_79_in_0: f16
        // expr_79_out_0 = Mul(expr_79_in_0, FastFloatSigmoid(expr_79_in_0))
        expr_79_out_0[i0] = (expr_79_in_0[i0]*fastsigmoid_f16(expr_79_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_80multiple_1(expr_80multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_80_in_0 = Args->expr_80_in_0;
    f16 *__restrict__  expr_80_out_0 = Args->expr_80_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_80_out_0: (32, 20, 20) expr_80_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_80_in_0: f16
        // expr_80_out_0 = Mul(expr_80_in_0, FastFloatSigmoid(expr_80_in_0))
        expr_80_out_0[i0] = (expr_80_in_0[i0]*fastsigmoid_f16(expr_80_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_81multiple_1(expr_81multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_81_in_0 = Args->expr_81_in_0;
    f16 *__restrict__  expr_81_out_0 = Args->expr_81_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_81_out_0: (32, 20, 20) expr_81_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_81_in_0: f16
        // expr_81_out_0 = Mul(expr_81_in_0, FastFloatSigmoid(expr_81_in_0))
        expr_81_out_0[i0] = (expr_81_in_0[i0]*fastsigmoid_f16(expr_81_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_82multiple_1(expr_82multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_82_in_0 = Args->expr_82_in_0;
    f16 *__restrict__  expr_82_out_0 = Args->expr_82_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_82_out_0: (32, 20, 20) expr_82_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_82_in_0: f16
        // expr_82_out_0 = Mul(expr_82_in_0, FastFloatSigmoid(expr_82_in_0))
        expr_82_out_0[i0] = (expr_82_in_0[i0]*fastsigmoid_f16(expr_82_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_83multiple_1(expr_83multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_83_in_0 = Args->expr_83_in_0;
    f16 *__restrict__  expr_83_out_0 = Args->expr_83_out_0; // (32, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 20, 20) var shapes:
    // expr_83_out_0: (32, 20, 20) expr_83_in_0: (32, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_83_in_0: f16
        // expr_83_out_0 = Mul(expr_83_in_0, FastFloatSigmoid(expr_83_in_0))
        expr_83_out_0[i0] = (expr_83_in_0[i0]*fastsigmoid_f16(expr_83_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_85multiple_1(expr_85multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_85_in_0 = Args->expr_85_in_0;
    f16 *__restrict__  expr_85_out_0 = Args->expr_85_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_85_out_0: (64, 20, 20) expr_85_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_85_in_0: f16
        // expr_85_out_0 = Mul(expr_85_in_0, FastFloatSigmoid(expr_85_in_0))
        expr_85_out_0[i0] = (expr_85_in_0[i0]*fastsigmoid_f16(expr_85_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_86multiple_1(expr_86multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_86_in_0 = Args->expr_86_in_0;
    f16 *__restrict__  expr_86_out_0 = Args->expr_86_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_86_out_0: (64, 10, 10) expr_86_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_86_in_0: f16
        // expr_86_out_0 = Mul(expr_86_in_0, FastFloatSigmoid(expr_86_in_0))
        expr_86_out_0[i0] = (expr_86_in_0[i0]*fastsigmoid_f16(expr_86_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_87multiple_1(expr_87multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_87_in_0 = Args->expr_87_in_0;
    f16 *__restrict__  expr_87_out_0 = Args->expr_87_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_87_out_0: (64, 10, 10) expr_87_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_87_in_0: f16
        // expr_87_out_0 = Mul(expr_87_in_0, FastFloatSigmoid(expr_87_in_0))
        expr_87_out_0[i0] = (expr_87_in_0[i0]*fastsigmoid_f16(expr_87_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_98multiple_1(expr_98multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_98_in_0 = Args->expr_98_in_0;
    f16 *__restrict__  expr_98_out_0 = Args->expr_98_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_98_out_0: (64, 10, 10) expr_98_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_98_in_0: f16
        // expr_98_out_0 = Mul(expr_98_in_0, FastFloatSigmoid(expr_98_in_0))
        expr_98_out_0[i0] = (expr_98_in_0[i0]*fastsigmoid_f16(expr_98_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_99multiple_1(expr_99multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_99_in_0 = Args->expr_99_in_0;
    f16 *__restrict__  expr_99_out_0 = Args->expr_99_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_99_out_0: (64, 10, 10) expr_99_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_99_in_0: f16
        // expr_99_out_0 = Mul(expr_99_in_0, FastFloatSigmoid(expr_99_in_0))
        expr_99_out_0[i0] = (expr_99_in_0[i0]*fastsigmoid_f16(expr_99_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_100multiple_1(expr_100multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_100_in_0 = Args->expr_100_in_0;
    f16 *__restrict__  expr_100_out_0 = Args->expr_100_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_100_out_0: (64, 10, 10) expr_100_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_100_in_0: f16
        // expr_100_out_0 = Mul(expr_100_in_0, FastFloatSigmoid(expr_100_in_0))
        expr_100_out_0[i0] = (expr_100_in_0[i0]*fastsigmoid_f16(expr_100_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_101multiple_1(expr_101multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_101_in_0 = Args->expr_101_in_0;
    f16 *__restrict__  expr_101_out_0 = Args->expr_101_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_101_out_0: (64, 10, 10) expr_101_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_101_in_0: f16
        // expr_101_out_0 = Mul(expr_101_in_0, FastFloatSigmoid(expr_101_in_0))
        expr_101_out_0[i0] = (expr_101_in_0[i0]*fastsigmoid_f16(expr_101_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_102multiple_1(expr_102multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_102_in_0 = Args->expr_102_in_0;
    f16 *__restrict__  expr_102_out_0 = Args->expr_102_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_102_out_0: (64, 10, 10) expr_102_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_102_in_0: f16
        // expr_102_out_0 = Mul(expr_102_in_0, FastFloatSigmoid(expr_102_in_0))
        expr_102_out_0[i0] = (expr_102_in_0[i0]*fastsigmoid_f16(expr_102_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_103multiple_1(expr_103multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_103_in_0 = Args->expr_103_in_0;
    f16 *__restrict__  expr_103_out_0 = Args->expr_103_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_103_out_0: (64, 10, 10) expr_103_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_103_in_0: f16
        // expr_103_out_0 = Mul(expr_103_in_0, FastFloatSigmoid(expr_103_in_0))
        expr_103_out_0[i0] = (expr_103_in_0[i0]*fastsigmoid_f16(expr_103_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_104multiple_1(expr_104multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_104_in_0 = Args->expr_104_in_0;
    f16 *__restrict__  expr_104_out_0 = Args->expr_104_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_104_out_0: (64, 10, 10) expr_104_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_104_in_0: f16
        // expr_104_out_0 = Mul(expr_104_in_0, FastFloatSigmoid(expr_104_in_0))
        expr_104_out_0[i0] = (expr_104_in_0[i0]*fastsigmoid_f16(expr_104_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_105multiple_1(expr_105multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_105_in_0 = Args->expr_105_in_0;
    f16 *__restrict__  expr_105_out_0 = Args->expr_105_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_105_out_0: (64, 10, 10) expr_105_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_105_in_0: f16
        // expr_105_out_0 = Mul(expr_105_in_0, FastFloatSigmoid(expr_105_in_0))
        expr_105_out_0[i0] = (expr_105_in_0[i0]*fastsigmoid_f16(expr_105_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_106multiple_1(expr_106multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_106_in_0 = Args->expr_106_in_0;
    f16 *__restrict__  expr_106_out_0 = Args->expr_106_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_106_out_0: (64, 10, 10) expr_106_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_106_in_0: f16
        // expr_106_out_0 = Mul(expr_106_in_0, FastFloatSigmoid(expr_106_in_0))
        expr_106_out_0[i0] = (expr_106_in_0[i0]*fastsigmoid_f16(expr_106_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_108multiple_1(expr_108multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_108_in_0 = Args->expr_108_in_0;
    f16 *__restrict__  expr_108_out_0 = Args->expr_108_out_0; // (128, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 10, 10) var shapes:
    // expr_108_out_0: (128, 10, 10) expr_108_in_0: (128, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_108_in_0: f16
        // expr_108_out_0 = Mul(expr_108_in_0, FastFloatSigmoid(expr_108_in_0))
        expr_108_out_0[i0] = (expr_108_in_0[i0]*fastsigmoid_f16(expr_108_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_65multiple_1(expr_65multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_65_in_0 = Args->expr_65_in_0;
    f16 *__restrict__  expr_65_out_0 = Args->expr_65_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_65_out_0: (64, 40, 40) expr_65_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_65_in_0: f16
        // expr_65_out_0 = Mul(expr_65_in_0, FastFloatSigmoid(expr_65_in_0))
        expr_65_out_0[i0] = (expr_65_in_0[i0]*fastsigmoid_f16(expr_65_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_66multiple_1(expr_66multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_66_in_0 = Args->expr_66_in_0;
    f16 *__restrict__  expr_66_out_0 = Args->expr_66_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_66_out_0: (64, 40, 40) expr_66_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_66_in_0: f16
        // expr_66_out_0 = Mul(expr_66_in_0, FastFloatSigmoid(expr_66_in_0))
        expr_66_out_0[i0] = (expr_66_in_0[i0]*fastsigmoid_f16(expr_66_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_67multiple_1(expr_67multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_67_in_0 = Args->expr_67_in_0;
    f16 *__restrict__  expr_67_out_0 = Args->expr_67_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_67_out_0: (64, 40, 40) expr_67_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_67_in_0: f16
        // expr_67_out_0 = Mul(expr_67_in_0, FastFloatSigmoid(expr_67_in_0))
        expr_67_out_0[i0] = (expr_67_in_0[i0]*fastsigmoid_f16(expr_67_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_68multiple_1(expr_68multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_68_in_0 = Args->expr_68_in_0;
    f16 *__restrict__  expr_68_out_0 = Args->expr_68_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_68_out_0: (64, 40, 40) expr_68_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_68_in_0: f16
        // expr_68_out_0 = Mul(expr_68_in_0, FastFloatSigmoid(expr_68_in_0))
        expr_68_out_0[i0] = (expr_68_in_0[i0]*fastsigmoid_f16(expr_68_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_69multiple_1(expr_69multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_69_in_0 = Args->expr_69_in_0;
    f16 *__restrict__  expr_69_out_0 = Args->expr_69_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_69_out_0: (64, 40, 40) expr_69_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_69_in_0: f16
        // expr_69_out_0 = Mul(expr_69_in_0, FastFloatSigmoid(expr_69_in_0))
        expr_69_out_0[i0] = (expr_69_in_0[i0]*fastsigmoid_f16(expr_69_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_70multiple_1(expr_70multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_70_in_0 = Args->expr_70_in_0;
    f16 *__restrict__  expr_70_out_0 = Args->expr_70_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_70_out_0: (64, 40, 40) expr_70_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_70_in_0: f16
        // expr_70_out_0 = Mul(expr_70_in_0, FastFloatSigmoid(expr_70_in_0))
        expr_70_out_0[i0] = (expr_70_in_0[i0]*fastsigmoid_f16(expr_70_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_71multiple_1(expr_71multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_71_in_0 = Args->expr_71_in_0;
    f16 *__restrict__  expr_71_out_0 = Args->expr_71_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_71_out_0: (64, 40, 40) expr_71_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_71_in_0: f16
        // expr_71_out_0 = Mul(expr_71_in_0, FastFloatSigmoid(expr_71_in_0))
        expr_71_out_0[i0] = (expr_71_in_0[i0]*fastsigmoid_f16(expr_71_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_72multiple_1(expr_72multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_72_in_0 = Args->expr_72_in_0;
    f16 *__restrict__  expr_72_out_0 = Args->expr_72_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_72_out_0: (64, 40, 40) expr_72_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_72_in_0: f16
        // expr_72_out_0 = Mul(expr_72_in_0, FastFloatSigmoid(expr_72_in_0))
        expr_72_out_0[i0] = (expr_72_in_0[i0]*fastsigmoid_f16(expr_72_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_73multiple_1(expr_73multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_73_in_0 = Args->expr_73_in_0;
    f16 *__restrict__  expr_73_out_0 = Args->expr_73_out_0; // (64, 40, 40) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 40, 40) var shapes:
    // expr_73_out_0: (64, 40, 40) expr_73_in_0: (64, 40, 40)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_73_in_0: f16
        // expr_73_out_0 = Mul(expr_73_in_0, FastFloatSigmoid(expr_73_in_0))
        expr_73_out_0[i0] = (expr_73_in_0[i0]*fastsigmoid_f16(expr_73_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_88multiple_1(expr_88multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_88_in_0 = Args->expr_88_in_0;
    f16 *__restrict__  expr_88_out_0 = Args->expr_88_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_88_out_0: (64, 20, 20) expr_88_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_88_in_0: f16
        // expr_88_out_0 = Mul(expr_88_in_0, FastFloatSigmoid(expr_88_in_0))
        expr_88_out_0[i0] = (expr_88_in_0[i0]*fastsigmoid_f16(expr_88_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_89multiple_1(expr_89multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_89_in_0 = Args->expr_89_in_0;
    f16 *__restrict__  expr_89_out_0 = Args->expr_89_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_89_out_0: (64, 20, 20) expr_89_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_89_in_0: f16
        // expr_89_out_0 = Mul(expr_89_in_0, FastFloatSigmoid(expr_89_in_0))
        expr_89_out_0[i0] = (expr_89_in_0[i0]*fastsigmoid_f16(expr_89_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_90multiple_1(expr_90multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_90_in_0 = Args->expr_90_in_0;
    f16 *__restrict__  expr_90_out_0 = Args->expr_90_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_90_out_0: (64, 20, 20) expr_90_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_90_in_0: f16
        // expr_90_out_0 = Mul(expr_90_in_0, FastFloatSigmoid(expr_90_in_0))
        expr_90_out_0[i0] = (expr_90_in_0[i0]*fastsigmoid_f16(expr_90_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_91multiple_1(expr_91multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_91_in_0 = Args->expr_91_in_0;
    f16 *__restrict__  expr_91_out_0 = Args->expr_91_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_91_out_0: (64, 20, 20) expr_91_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_91_in_0: f16
        // expr_91_out_0 = Mul(expr_91_in_0, FastFloatSigmoid(expr_91_in_0))
        expr_91_out_0[i0] = (expr_91_in_0[i0]*fastsigmoid_f16(expr_91_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_92multiple_1(expr_92multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_92_in_0 = Args->expr_92_in_0;
    f16 *__restrict__  expr_92_out_0 = Args->expr_92_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_92_out_0: (64, 20, 20) expr_92_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_92_in_0: f16
        // expr_92_out_0 = Mul(expr_92_in_0, FastFloatSigmoid(expr_92_in_0))
        expr_92_out_0[i0] = (expr_92_in_0[i0]*fastsigmoid_f16(expr_92_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_93multiple_1(expr_93multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_93_in_0 = Args->expr_93_in_0;
    f16 *__restrict__  expr_93_out_0 = Args->expr_93_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_93_out_0: (64, 20, 20) expr_93_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_93_in_0: f16
        // expr_93_out_0 = Mul(expr_93_in_0, FastFloatSigmoid(expr_93_in_0))
        expr_93_out_0[i0] = (expr_93_in_0[i0]*fastsigmoid_f16(expr_93_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_94multiple_1(expr_94multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_94_in_0 = Args->expr_94_in_0;
    f16 *__restrict__  expr_94_out_0 = Args->expr_94_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_94_out_0: (64, 20, 20) expr_94_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_94_in_0: f16
        // expr_94_out_0 = Mul(expr_94_in_0, FastFloatSigmoid(expr_94_in_0))
        expr_94_out_0[i0] = (expr_94_in_0[i0]*fastsigmoid_f16(expr_94_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_95multiple_1(expr_95multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_95_in_0 = Args->expr_95_in_0;
    f16 *__restrict__  expr_95_out_0 = Args->expr_95_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_95_out_0: (64, 20, 20) expr_95_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_95_in_0: f16
        // expr_95_out_0 = Mul(expr_95_in_0, FastFloatSigmoid(expr_95_in_0))
        expr_95_out_0[i0] = (expr_95_in_0[i0]*fastsigmoid_f16(expr_95_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_96multiple_1(expr_96multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_96_in_0 = Args->expr_96_in_0;
    f16 *__restrict__  expr_96_out_0 = Args->expr_96_out_0; // (64, 20, 20) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 20, 20) var shapes:
    // expr_96_out_0: (64, 20, 20) expr_96_in_0: (64, 20, 20)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_96_in_0: f16
        // expr_96_out_0 = Mul(expr_96_in_0, FastFloatSigmoid(expr_96_in_0))
        expr_96_out_0[i0] = (expr_96_in_0[i0]*fastsigmoid_f16(expr_96_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_109multiple_1(expr_109multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_109_in_0 = Args->expr_109_in_0;
    f16 *__restrict__  expr_109_out_0 = Args->expr_109_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_109_out_0: (64, 10, 10) expr_109_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_109_in_0: f16
        // expr_109_out_0 = Mul(expr_109_in_0, FastFloatSigmoid(expr_109_in_0))
        expr_109_out_0[i0] = (expr_109_in_0[i0]*fastsigmoid_f16(expr_109_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_110multiple_1(expr_110multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_110_in_0 = Args->expr_110_in_0;
    f16 *__restrict__  expr_110_out_0 = Args->expr_110_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_110_out_0: (64, 10, 10) expr_110_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_110_in_0: f16
        // expr_110_out_0 = Mul(expr_110_in_0, FastFloatSigmoid(expr_110_in_0))
        expr_110_out_0[i0] = (expr_110_in_0[i0]*fastsigmoid_f16(expr_110_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_111multiple_1(expr_111multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_111_in_0 = Args->expr_111_in_0;
    f16 *__restrict__  expr_111_out_0 = Args->expr_111_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_111_out_0: (64, 10, 10) expr_111_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_111_in_0: f16
        // expr_111_out_0 = Mul(expr_111_in_0, FastFloatSigmoid(expr_111_in_0))
        expr_111_out_0[i0] = (expr_111_in_0[i0]*fastsigmoid_f16(expr_111_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_112multiple_1(expr_112multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_112_in_0 = Args->expr_112_in_0;
    f16 *__restrict__  expr_112_out_0 = Args->expr_112_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_112_out_0: (64, 10, 10) expr_112_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_112_in_0: f16
        // expr_112_out_0 = Mul(expr_112_in_0, FastFloatSigmoid(expr_112_in_0))
        expr_112_out_0[i0] = (expr_112_in_0[i0]*fastsigmoid_f16(expr_112_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_113multiple_1(expr_113multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_113_in_0 = Args->expr_113_in_0;
    f16 *__restrict__  expr_113_out_0 = Args->expr_113_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_113_out_0: (64, 10, 10) expr_113_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_113_in_0: f16
        // expr_113_out_0 = Mul(expr_113_in_0, FastFloatSigmoid(expr_113_in_0))
        expr_113_out_0[i0] = (expr_113_in_0[i0]*fastsigmoid_f16(expr_113_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_114multiple_1(expr_114multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_114_in_0 = Args->expr_114_in_0;
    f16 *__restrict__  expr_114_out_0 = Args->expr_114_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_114_out_0: (64, 10, 10) expr_114_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_114_in_0: f16
        // expr_114_out_0 = Mul(expr_114_in_0, FastFloatSigmoid(expr_114_in_0))
        expr_114_out_0[i0] = (expr_114_in_0[i0]*fastsigmoid_f16(expr_114_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_115multiple_1(expr_115multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_115_in_0 = Args->expr_115_in_0;
    f16 *__restrict__  expr_115_out_0 = Args->expr_115_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_115_out_0: (64, 10, 10) expr_115_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_115_in_0: f16
        // expr_115_out_0 = Mul(expr_115_in_0, FastFloatSigmoid(expr_115_in_0))
        expr_115_out_0[i0] = (expr_115_in_0[i0]*fastsigmoid_f16(expr_115_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_116multiple_1(expr_116multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_116_in_0 = Args->expr_116_in_0;
    f16 *__restrict__  expr_116_out_0 = Args->expr_116_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_116_out_0: (64, 10, 10) expr_116_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_116_in_0: f16
        // expr_116_out_0 = Mul(expr_116_in_0, FastFloatSigmoid(expr_116_in_0))
        expr_116_out_0[i0] = (expr_116_in_0[i0]*fastsigmoid_f16(expr_116_in_0[i0]));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_117multiple_1(expr_117multiple_1_args_t * __restrict__ Args) {
    f16 *__restrict__  expr_117_in_0 = Args->expr_117_in_0;
    f16 *__restrict__  expr_117_out_0 = Args->expr_117_out_0; // (64, 10, 10) f16
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (64, 10, 10) var shapes:
    // expr_117_out_0: (64, 10, 10) expr_117_in_0: (64, 10, 10)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_117_in_0: f16
        // expr_117_out_0 = Mul(expr_117_in_0, FastFloatSigmoid(expr_117_in_0))
        expr_117_out_0[i0] = (expr_117_in_0[i0]*fastsigmoid_f16(expr_117_in_0[i0]));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop