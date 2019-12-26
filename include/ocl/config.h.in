#if !defined (__OCL_CONFIG_H__)
#define __OCL_CONFIG_H__ 1

#if defined (__GNUC__) || defined (__clang__)
// clang++ -dM -E -x c++ -march=native /dev/null
// g++ -dM -E -x c++ -march=native /dev/null
#define __thread_local __thread
#define __likely(a) __builtin_expect(!!(a), 1)
#define __unlikely(a) __builtin_expect(!!(a), 0)
#define __assume_aligned(p, x) __builtin_assume_aligned(p, x)
#endif

#if defined (_MSC_VER)
#define __thread_local __declspec(thread)
#endif

#if !defined (__assume_aligned)
#define __assume_aligned(p, x) p
#endif
#if !defined (__likely)
#define __likely(a) a
#endif
#if !defined (__unlikely)
#define __unlikely(a) a
#endif
#if !defined(__GNUC__) && !defined (__CLANG__)
#define __restrict
#endif

#endif
