#define _cuda_buffers_cu
#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};

CudaBuffers::CudaBuffers
(
    int _device,
    half* _temp_state,
    int _temp_state_size,
    half* _temp_dq,
) :
    device(_device),
    temp_state(_temp_state),
    temp_state_size(_temp_state_size),
    temp_dq(_temp_dq),
{
    cudaSetDevice(_device);

    cudaStreamCreate(&alt_stream_1);
    cudaStreamCreate(&alt_stream_2);
    cudaStreamCreate(&alt_stream_3);
    cudaEventCreate(&alt_stream_1_done);
    cudaEventCreate(&alt_stream_2_done);
    cudaEventCreate(&alt_stream_3_done);
}

CudaBuffers::~CudaBuffers()
{
    cudaStreamDestroy(alt_stream_1);
    cudaStreamDestroy(alt_stream_2);
    cudaStreamDestroy(alt_stream_3);
    cudaEventDestroy(alt_stream_1_done);
    cudaEventDestroy(alt_stream_2_done);
    cudaEventDestroy(alt_stream_3_done);
}

CudaBuffers* get_buffers(const int device_index)
{
    return g_buffers[device_index];
}

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    int _temp_state_size,
    half* _temp_dq,
)
{
    CudaBuffers* buffers = new CudaBuffers
    (
        _device,
        _temp_state,
        _temp_state_size,
        _temp_dq,
    );

    g_buffers[_device] = buffers;
}

void cleanup_buffers_cuda()
{
    for (int i = 0; i < CUDA_MAX_DEVICES; i++)
    {
        if (!g_buffers[i]) continue;
        delete g_buffers[i];
        g_buffers[i] = NULL;
    }
}