#ifndef _CA_ACCESSOR_H_
#define _CA_ACCESSOR_H_
#include "caconst.h"
#include "legion.h"

#if defined(CA_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(CA_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(CA_USE_HIP_ROCM)
#include <hidapi/hip_fp16.h>
#endif

// using namespace Legion;

namespace astarte {
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRO =
    Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRW = Legion::
    FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorWO = Legion::
    FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

template <typename DT, int dim>
struct TensorAccessorR {
    TensorAccessorR(Legion::PhysicalRegion region,
                    Legion::RegionRequirement req,
                    Legion::FieldID fid,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);
    TensorAccessorR();
    Legion::Rect<dim> rect;
    Legion::Memory memory;
    const DT *ptr;
};

template <typename DT, int dim>
struct TensorAccessorW {
    TensorAccessorW(Legion::PhysicalRegion region,
                    Legion::RegionRequirement req,
                    Legion::FieldID fid,
                    Legion::Context ctx,
                    Legion::Runtime *runtime.
                    bool readOutput = false);
    TensorAccessorW();
    Legion::Rect<dim> rect;
    Legion:: Memory memory;
    DT *ptr;
};

class GenericTensorAccessorW {
    public:
    GenericTensorAccessorW();
    GenericTensorAccessorW(DataType data_type, Legion::Domain domain, void *ptr);
    int32_t *get_int32_ptr() const;
    int64_t *get_int64_ptr() const;
    float *get_float_ptr() const;
    double *get_double_ptr() const;
    half *get_half_ptr() const;
    char *get_byte_ptr() const;
    DataType data_type;
    Legion::Domain domain;
    void *ptr;
};

class GenericTensorAccessorR {
    public:
    GenericTensorAccessorR();
    GenericTensorAccessorR(DataType data_type, Legion::Domain domain, void const *ptr);
    GenericTensorAccessorR(GenericTensorAccessorW const &acc);
    int32_t *get_int32_ptr() const;
    int64_t *get_int64_ptr() const;
    float *get_float_ptr() const;
    double *get_double_ptr() const;
    half *get_half_ptr() const;
    char *get_byte_ptr() const;
    DataType data_type;
    Legion::Domain domain;
    void *ptr;
};

template <typename DT>
const DT *helperGetTensorPointerRO(Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);

template <typename DT>
const DT *helperGetTensorPointerRW(Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);

GenericTensorAccessorR
    helperGetGenericTensorAccessorRO(DataType datatype,
                                   Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);
GenericTensorAccessorW
    helperGetGenericTensorAccessorWO(DataType datatype,
                                   Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);

GenericTensorAccessorR
    helperGetGenericTensorAccessorRW(DataType datatype,
                                   Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);                          
}; // namespace astarte

#endif
