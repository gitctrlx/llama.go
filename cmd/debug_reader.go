package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
)

// 只保留 Config 结构体，和官方 C 代码完全对应
type Config struct {
	Dim       int32
	HiddenDim int32
	NLayers   int32
	NHeads    int32
	NKvHeads  int32
	VocabSize int32
	SeqLen    int32
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("请提供模型文件路径，例如：go run debug_reader.go ./stories15M.bin")
	}
	filePath := os.Args[1]

	// 获取文件总大小，用于对比
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		log.Fatalf("无法获取文件信息: %v", err)
	}
	fmt.Printf("文件总大小: %d 字节\n", fileInfo.Size())
	fmt.Println("---------------------------------")

	f, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("无法打开文件: %v", err)
	}
	defer f.Close()

	var config Config
	// 只读取 Config 结构体
	if err := binary.Read(f, binary.LittleEndian, &config); err != nil {
		log.Fatalf("读取 config 失败: %v", err)
	}

	fmt.Println("成功读取 Config 结构体，内容如下:")
	fmt.Printf("  Dim (维度):       %d\n", config.Dim)
	fmt.Printf("  HiddenDim (隐藏层): %d\n", config.HiddenDim)
	fmt.Printf("  NLayers (层数):     %d\n", config.NLayers)
	fmt.Printf("  NHeads (查询头):    %d\n", config.NHeads)
	fmt.Printf("  NKvHeads (键值头):  %d\n", config.NKvHeads)
	fmt.Printf("  VocabSize (词汇表): %d\n", config.VocabSize)
	fmt.Printf("  SeqLen (序列长度):  %d\n", config.SeqLen)
	fmt.Println("---------------------------------")

	// 让我们手动计算一下 stories15M.bin 的理论大小
	// Config 本身是 7 个 int32 = 28 字节
	// 我们知道 stories15M 的正确 VocabSize 是 32000
	// 如果读出的 VocabSize 是个奇怪的负数，这里就会很明显
	trueVocabSize := config.VocabSize
	// sharedWeights := false
	// if trueVocabSize < 0 {
	// 	trueVocabSize = -trueVocabSize
	// 	sharedWeights = true
	// }

	// 权重参数量计算 (来自 llama2.c 的 `build_transformer` 函数)
	tokenEmbeddingTable := int64(trueVocabSize) * int64(config.Dim)
	rmsAttWeight := int64(config.NLayers) * int64(config.Dim)
	wq := int64(config.NLayers) * int64(config.Dim) * int64(config.Dim)
	wk := int64(config.NLayers) * int64(config.Dim) * int64(config.Dim)
	wv := int64(config.NLayers) * int64(config.Dim) * int64(config.Dim)
	wo := int64(config.NLayers) * int64(config.Dim) * int64(config.Dim)
	rmsFfnWeight := int64(config.NLayers) * int64(config.Dim)
	w1 := int64(config.NLayers) * int64(config.Dim) * int64(config.HiddenDim)
	w2 := int64(config.NLayers) * int64(config.HiddenDim) * int64(config.Dim)
	w3 := int64(config.NLayers) * int64(config.Dim) * int64(config.HiddenDim)
	rmsFinalWeight := int64(config.Dim)
	freqCisReal := int64(config.SeqLen) * int64(config.Dim) / 2
	freqCisImag := int64(config.SeqLen) * int64(config.Dim) / 2

	totalParams := tokenEmbeddingTable + rmsAttWeight + wq + wk + wv + wo + rmsFfnWeight + w1 + w2 + w3 + rmsFinalWeight + freqCisReal + freqCisImag

	// 如果权重共享，最后的分类器权重是不存在的
	// 在 llama2.c 中，这个权重和 tokenEmbeddingTable 是同一个，但文件里只存一份
	// 实际上，最终的分类器权重并没有被单独写进文件，而是直接用了 embedding table
	// 所以我们不需要从 totalParams 中减去它

	// 总字节数 = 头部(28字节) + 参数量 * 4字节/参数
	expectedSize := 28 + totalParams*4

	fmt.Printf("根据读取到的 Config，计算出的理论文件大小应为: %d 字节\n", expectedSize)

	if fileInfo.Size() < expectedSize {
		fmt.Println("\n结论: 文件实际大小 < 理论大小。文件确实不完整或Config读取有误。")
	} else {
		fmt.Println("\n结论: 文件大小看起来是足夠的。问题可能更复杂，比如Go的内存分配或读取逻辑。")
	}
}
