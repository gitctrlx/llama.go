# llama.go

A pure Go implementation of the LLaMA model for inference and educational purposes. Supports LLaMA 1, 2, and 3 architectures.

This repository demonstrates how to run LLaMA inference using only Go's standard library, making it ideal for learning and understanding transformer internals without heavy dependencies.

## Features

- **HF-aligned Architecture** – Matches HuggingFace reference implementation with clean, structured codebase matching official model layouts
- **Concurrent MHA** – Multi-head attention parallelized across goroutines for 2-4x speedup on multi-core systems  
- **Int8 Quantization** – Post-training quantization reduces model size by 4x while maintaining inference quality
- **Zero Dependencies** – 100% pure Go standard library, no external packages or CGO
- **Educational** – Line-by-line readable transformer implementation with inline documentation

## Usage

```sh
go run cmd/llama/main.go ./cmd/llama/stories15M.bin
```

The examples use small models trained by [Andrej Karpathy](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) for demonstration.

## Acknowledgments

Inspired by [llama2.c](https://github.com/karpathy/llama2.c) and [go-llama2](https://github.com/tmc/go-llama2). Licensed under their respective terms.

## License

MIT
