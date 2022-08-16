#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000000c0,0x0000008801010001,0x0000000000000038\n"
".quad 0x0000004000000036,0x0000004b00060004,0x0000000000000000,0x0000000000002011\n"
".quad 0x0000000000000000,0x0000000000000038,0x0000003e00000048,0x3d6567616d692d2d\n"
".quad 0x3d656c69666f7270,0x69662c35375f6d73,0x432f706d743d656c,0x41445543656b614d\n"
".quad 0x72656c69706d6f43,0x35375f6d732e6449,0x00006e696275632e,0x762e21f000010a13\n"
".quad 0x36206e6f69737265,0x677261742e0a342e,0x35375f6d73207465,0x7365726464612e0a\n"
".quad 0x3620657a69735f73, 0x0000000a0a0a0a34\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[26];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif
