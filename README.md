# llama.go

A pure Go implementation of the LLaMA model for inference and educational purposes. Supports LLaMA 1, 2, and 3 architectures.

This repository demonstrates how to run LLaMA inference using only Go's standard library, making it ideal for learning and understanding transformer internals without heavy dependencies.

## Usage

```sh
go run cmd/llama/main.go ./cmd/llama/stories15M.bin
```

The example uses a small model trained by [Andrej Karpathy](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) for demonstration.

## Acknowledgments

Inspired by [llama2.c](https://github.com/karpathy/llama2.c) and [go-llama2](https://github.com/tmc/go-llama2). Licensed under their respective terms.

## License

MIT
