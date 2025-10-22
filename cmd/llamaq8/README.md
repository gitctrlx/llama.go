# llama.go - Int8 Quantization

Int8 quantized LLaMA inference for faster execution and reduced memory usage.

## Usage

```bash
go run cmd/llamaq8/main.go ./cmd/llamaq8/stories15M_q8.bin
```

## Int8 Quantization

For creating quantized models and detailed documentation, see: [llama2.c int8 quantization](https://github.com/karpathy/llama2.c/tree/master?tab=readme-ov-file#int8-quantization)

## Acknowledgments

Based on quantization techniques from [llama2.c](https://github.com/karpathy/llama2.c) and [llama.cpp](https://github.com/ggerganov/llama.cpp).
