import torch
from aphrodite.quantization.quip_utils import (
    hadamard_transform,
    get_hadK,
    matmul_hadU_cuda,
    matmul_hadUt_cuda
)

# Example 1: Basic Hadamard Transform
def example_hadamard():
    # Create a random tensor
    x = torch.randn(4, 4)  # Must be power of 2 dimensions
    # Apply Hadamard transform
    transformed = hadamard_transform(x, scale=1.0)
    # Inverse transform (using the same function with appropriate scale)
    inverse = hadamard_transform(transformed, scale=1.0)
    print("Original shape:", x.shape)
    print("Transformed shape:", transformed.shape)
    print("Reconstruction error:", torch.norm(x - inverse))

# Example 2: Using Hadamard-based matrix multiplication
def example_hadamard_matmul():
    # Create input tensor
    batch_size = 2
    n = 16  # dimension size (power of 2)
    x = torch.randn(batch_size, n)
    
    # Get Hadamard matrices and parameters
    hadK, K, padded_n = get_hadK(n, use_rand=True)
    
    # Forward transform
    transformed = matmul_hadU_cuda(x, hadK, K, padded_n)
    
    # Backward transform
    reconstructed = matmul_hadUt_cuda(transformed, hadK, K, padded_n)
    
    print("Input shape:", x.shape)
    print("Transformed shape:", transformed.shape)
    print("Reconstruction error:", torch.norm(x - reconstructed))

# Example 3: Working with non-power-of-2 dimensions
def example_non_power_2():
    # Create tensor with non-power-of-2 dimension
    x = torch.randn(3, 10)
    
    # Get appropriate Hadamard matrices and padding
    hadK, K, padded_n = get_hadK(x.shape[-1], use_rand=True)
    
    # Forward transform (will handle padding automatically)
    transformed = matmul_hadU_cuda(x, hadK, K, padded_n)
    
    # Backward transform
    reconstructed = matmul_hadUt_cuda(transformed, hadK, K, padded_n)
    
    # Remove padding from result
    reconstructed = reconstructed[..., :x.shape[-1]]
    
    print("Original shape:", x.shape)
    print("Padded transformed shape:", transformed.shape)
    print("Final reconstructed shape:", reconstructed.shape)
    print("Reconstruction error:", torch.norm(x - reconstructed))

if __name__ == "__main__":
    print("Example 1: Basic Hadamard Transform")
    example_hadamard()
    print("\nExample 2: Hadamard Matrix Multiplication")
    example_hadamard_matmul()
    print("\nExample 3: Non-power-of-2 Dimensions")
    example_non_power_2()
