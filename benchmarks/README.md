# Benchmarks

This directory contains Sonar's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Kernel benchmarks

The CUTLASS FP8, FP4, and grouped-GEMM benchmarks use the modular MoE expert
interfaces and initialize the shared workspace before running kernels. The
fork's `kernels/benchmark_cutlass_fp4_moe.py` remains the FP4 entry point; an
additional copy under the upstream NVFP4 filename is not required.

`kernels/benchmark_marlin.py` targets the current dense Marlin GEMM and repack
interfaces. The removed sparse Marlin24 runtime is not a benchmark target.
The permutation benchmark likewise uses the current scratch-based API rather
than comparing against removed legacy permutation kernels.

The retained `cutlass_benchmarks/`, RMSNorm, and hashing benchmarks remain
available. `kernels/benchmark_bitblas.py` is a standalone benchmark of the
external `bitblas` package, not an engine quantization backend.
