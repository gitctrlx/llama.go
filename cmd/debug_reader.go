package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"

	"github.com/qntx/llama2.go"
)

func main() {
	// Check if model file path is provided
	if len(os.Args) < 2 {
		log.Fatal("Please provide the model file path, e.g.,: go run debug_reader.go ./stories15M.bin")
	}
	filePath := os.Args[1]

	// Get file size for comparison
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		log.Fatalf("Failed to get file info: %v", err)
	}
	fmt.Printf("Total file size: %d bytes\n", fileInfo.Size())
	fmt.Println("---------------------------------")

	// Open the model file
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer f.Close()

	// Read the Config struct
	var config llama2.Config
	if err := binary.Read(f, binary.LittleEndian, &config); err != nil {
		log.Fatalf("Failed to read config: %v", err)
	}

	// Print the Config struct contents
	fmt.Println("Successfully read Config struct, contents:")
	fmt.Printf("  Dim (dimension):       %d\n", config.Dim)
	fmt.Printf("  HiddenDim (hidden layer): %d\n", config.HiddenDim)
	fmt.Printf("  NLayers (layers):     %d\n", config.NLayers)
	fmt.Printf("  NHeads (heads):       %d\n", config.NHeads)
	fmt.Printf("  NKvHeads (key-value heads): %d\n", config.NKvHeads)
	fmt.Printf("  VocabSize (vocabulary): %d\n", config.VocabSize)
	fmt.Printf("  SeqLen (sequence length): %d\n", config.SeqLen)
	fmt.Println("---------------------------------")

	// Calculate theoretical size of stories15M.bin
	// Config itself is 7 int32 = 28 bytes
	// Known correct VocabSize for stories15M is 32000
	// Negative VocabSize would indicate an issue
	trueVocabSize := config.VocabSize
	// sharedWeights := false
	// if trueVocabSize < 0 {
	// 	trueVocabSize = -trueVocabSize
	// 	sharedWeights = true
	// }

	// Calculate parameter count (based on llama2.c's `build_transformer` function)
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

	// Sum all parameters
	totalParams := tokenEmbeddingTable + rmsAttWeight + wq + wk + wv + wo + rmsFfnWeight + w1 + w2 + w3 + rmsFinalWeight + freqCisReal + freqCisImag

	// If weights are shared, the final classifier weights are not stored separately
	// In llama2.c, these reuse the tokenEmbeddingTable, so no separate storage in the file
	// No need to subtract from totalParams

	// Total bytes = header (28 bytes) + parameters * 4 bytes/parameter
	expectedSize := 28 + totalParams*4

	fmt.Printf("Calculated theoretical file size based on Config: %d bytes\n", expectedSize)

	// Compare actual and expected file sizes
	if fileInfo.Size() < expectedSize {
		fmt.Println("\nConclusion: Actual file size < theoretical size. File is likely incomplete or Config is misread.")
	} else {
		fmt.Println("\nConclusion: File size appears sufficient. Issue may be more complex, e.g., Go memory allocation or read logic.")
	}
}
