---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Aphrodite Engine"
  text: "User and Developer Documentation"
  tagline: Breathing Life Into Language
  actions:
    - theme: brand
      text: Get Started
      link: /pages/installation/installation.html
    - theme: alt
      text: View Source Code
      link: https://github.com/PygmalionAI/aphrodite-engine

features:
  - title: Paged Attention
    details: Efficiently manage KV cache using vLLM's Paged Attention kernels.
  - title: Continuous Batching
    details: Continuously batch incoming requests in the Async server.
  - title: Hugging Face Integration
    details: Run almost any Hugging Face format LLM seamlessly.
  - title: Quantization Support
    details: Support for almost all quantization formats, with optimized kernels for
             efficient deployment.
  - title: OpenAI-compatible API
    details: Quickly deploy models with the integrated OpenAI API, supporting Text/Chat Completions, Vision, and Batch API.
  - title: Speculative Decoding
    details: Accelerate inference using various state-of-the-art spec-decoding methods.
  - title: Adapters
    details: Deploy hundreds or thousands of LoRAs efficiently using Punica, and PEFT-style
             Prompt adapters.
  - title: Hardware support
    details: Aphrodite supports NVIDIA & AMD GPUs, Intel XPUs, Google TPUs, AWS
             Inferentia/Trainium, AVX2/AVX512/ppc64le CPUs. 
---


