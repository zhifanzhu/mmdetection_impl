#define THREADS_PER_BLOCK 1024

/* ceiling division with integers. */
inline int ceilDivide(const int numerator, const int denominator) {
    return (numerator + denominator - 1) / denominator;
}

/* clamp to [0, 1] */
template<typename scalar_t>
__device__ __forceinline__ scalar_t clamp(scalar_t x)
{
    return max(static_cast<scalar_t>(0), min(static_cast<scalar_t>(1), x));
}

