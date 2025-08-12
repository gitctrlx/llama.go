package llama2

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// LoadTokenizer reads vocab and scores from a binary file.
func LoadTokenizer(path string) ([]string, []float32, uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()

	var maxLen uint32
	if err := binary.Read(f, binary.LittleEndian, &maxLen); err != nil {
		return nil, nil, 0, err
	}

	vocab := make([]string, 0, 32000) // Typical size hint
	scores := make([]float32, 0, 32000)
	for {
		var score float32
		if err := binary.Read(f, binary.LittleEndian, &score); err == io.EOF {
			break
		} else if err != nil {
			return nil, nil, 0, err
		}

		var len int32
		if err := binary.Read(f, binary.LittleEndian, &len); err != nil {
			return nil, nil, 0, err
		}

		buf := make([]byte, len)
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, nil, 0, err
		}

		scores = append(scores, score)
		vocab = append(vocab, string(buf))
	}

	return vocab, scores, maxLen, nil
}

// BPEEncode tokenizes text using byte-pair encoding.
func BPEEncode(text string, vocab []string, scores []float32, vocabMap map[string]int32, maxLen uint32) ([]int32, error) {
	tokens := make([]int32, 0, len(text))
	for _, ch := range text {
		id, ok := vocabMap[string(ch)]
		if !ok {
			return nil, fmt.Errorf("unknown byte: %s", string(ch))
		}
		tokens = append(tokens, id)
	}

	builder := strings.Builder{}
	builder.Grow(int(maxLen) * 2)

	for len(tokens) > 1 {
		bestScore := float32(math.Inf(-1))
		bestID := int32(-1)
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
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
			break // No more merges
		}

		tokens[bestIdx] = bestID
		tokens = append(tokens[:bestIdx+1], tokens[bestIdx+2:]...)
	}

	return tokens, nil
}
