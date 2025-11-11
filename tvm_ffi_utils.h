#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "driver_types.h"

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
    return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t float64_code = encode_dlpack_dtype(dl_float64);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);

#define _DISPATCH_CASE_F16(c_type, ...) \
    case float16_code: {                \
        using c_type = nv_half;         \
        __VA_ARGS__();                  \
        return true;                    \
    }
#define _DISPATCH_CASE_F32(c_type, ...) \
    case float32_code: {                \
        using c_type = float;           \
        __VA_ARGS__();                  \
        return true;                    \
    }
#define _DISPATCH_CASE_F64(c_type, ...) \
    case float64_code: {                \
        using c_type = double;          \
        __VA_ARGS__();                  \
        return true;                    \
    }
#define _DISPATCH_CASE_BF16(c_type, ...) \
    case bfloat16_code: {                \
        using c_type = nv_bfloat16;      \
        __VA_ARGS__();                   \
        return true;                     \
    }

#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FLOAT(dlpack_dtype, c_type, ...)                               \
    [&]() -> bool {                                                                                   \
        switch (encode_dlpack_dtype(dlpack_dtype)) {                                                  \
            _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                   \
            _DISPATCH_CASE_F32(c_type, __VA_ARGS__)                                                   \
            _DISPATCH_CASE_F64(c_type, __VA_ARGS__)                                                   \
            _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                  \
            default:                                                                                  \
                TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type "      \
                                      << (int)(dlpack_dtype).code << " " << (int)(dlpack_dtype).bits; \
                return false;                                                                         \
        }                                                                                             \
    }()

#define TVM_FFI_GET_CUDA_STREAM(data) \
    static_cast<cudaStream_t>(TVMFFIEnvGetStream(data.device().device_type, data.device().device_id))

#define CHECK_CUDA_SUCCESS(err)                                                            \
    do {                                                                                   \
        TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA Failure: " << cudaGetErrorString(err); \
    } while (0)
