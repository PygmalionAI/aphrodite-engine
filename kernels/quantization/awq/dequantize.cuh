#pragma once

__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t* h = reinterpret_cast<u_int32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x00f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to
    // the register packing format and the fact that we force out integers to be unsigned,
    // and account for this in the fp16 subtractions.
    // We exploit the fact that sub and fma have the same throughput in order to convert
    // elt_32 and elt_67 to fp16 without having to shift them to the bottom bits beforehand.

    // shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency
    // immediately before required.
    const uin32_t top_i4s = i4s >> 8;
    // extract elt_01 - (i4s & 0x00f00f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut);
    // extract elt_32 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    :"=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    :"=r"(h[2])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // extract elt_64 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    :"=r"(h[3])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // convert elt_23
    asm volatile("sub.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;

}