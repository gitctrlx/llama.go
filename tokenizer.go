package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// LoadTokenizer reads vocab and scores from a binary file.
func LoadTokenizer(path string, vocabSize int) ([]string, []float32, uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()

	var maxLen uint32
	if err := binary.Read(f, binary.LittleEndian, &maxLen); err != nil {
		return nil, nil, 0, err
	}

	vocab := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		var score float32
		if err := binary.Read(f, binary.LittleEndian, &score); err != nil {
			return nil, nil, 0, fmt.Errorf("failed to read score for index %d: %w", i, err)
		}
		scores[i] = score

		var len int32
		if err := binary.Read(f, binary.LittleEndian, &len); err != nil {
			return nil, nil, 0, fmt.Errorf("failed to read token length for index %d: %w", i, err)
		}

		buf := make([]byte, len)
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, nil, 0, fmt.Errorf("failed to read token for index %d: %w", i, err)
		}
		vocab[i] = string(buf)
	}

	return vocab, scores, maxLen, nil
}

// BPEEncode tokenizes text using byte-pair encoding, aligning with the C implementation.
func BPEEncode(text string, vocab []string, scores []float32, vocabMap map[string]int32, bos bool, eos bool) ([]int32, error) {
	// 1. Initial tokenization from string to IDs
	tokens := make([]int32, 0, len(text)+3)

	// Add BOS token if requested
	if bos {
		tokens = append(tokens, 1)
	}

	// llama tokenizer behavior: Add a dummy prefix space if the string is not empty.
	if text != "" {
		dummyPrefix, ok := vocabMap[" "]
		if !ok {
			return nil, fmt.Errorf("dummy prefix ' ' not found in vocabulary")
		}
		tokens = append(tokens, dummyPrefix)
	}

	// Process the string rune by rune (handles UTF-8 correctly)
	for _, r := range text {
		charStr := string(r)
		id, ok := vocabMap[charStr]
		if ok {
			// Character is in the vocabulary
			tokens = append(tokens, id)
		} else {
			// Byte-level fallback for unknown characters
			// This matches the C code's `(unsigned char)str_buffer[i] + 3`
			for _, b := range []byte(charStr) {
				tokens = append(tokens, int32(b)+3)
			}
		}
	}

	// 2. Iteratively merge the best pair
	builder := strings.Builder{}
	for {
		bestScore := float32(math.Inf(-1))
		bestID := int32(-1)
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			// Form the merged token string
			builder.Reset()
			builder.WriteString(vocab[tokens[i]])
			builder.WriteString(vocab[tokens[i+1]])
			merged := builder.String()

			if id, ok := vocabMap[merged]; ok && scores[id] > bestScore {
				bestScore = scores[id]
				bestID = id
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // No more merges possible
		}

		// Merge the best pair
		tokens[bestIdx] = bestID
		// and delete the second token of the pair
		tokens = append(tokens[:bestIdx+1], tokens[bestIdx+2:]...)
	}

	// Add EOS token if requested
	if eos {
		tokens = append(tokens, 2)
	}

	return tokens, nil
}
