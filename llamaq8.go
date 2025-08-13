package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
)

// QuantizedTensor holds the int8 quantized data and float32 scaling factors.
type QuantizedTensor struct {
	Q []int8    // Quantized values
	S []float32 // Scaling factors
}

// QuantizedConfig defines the transformer hyperparameters.
type QuantizedConfig struct {
	Dim       int32 // Transformer dimension
	HiddenDim int32 // FFN hidden dimension
	NLayers   int32 // Number of transformer layers
	NHeads    int32 // Number of attention query heads
	NKvHeads  int32 // Number of key/value heads (GQA)
	VocabSize int32 // Vocabulary size
	SeqLen    int32 // Maximum sequence length
}

// QuantizedTransformerWeights holds the quantized model parameters.
type QuantizedTransformerWeights struct {
	TokenEmbeddingTable []float32
	QTokens             *QuantizedTensor
	RmsAttWeight        []float32
	RmsFfnWeight        []float32
	RmsFinalWeight      []float32
	Wq                  []QuantizedTensor
	Wk                  []QuantizedTensor
	Wv                  []QuantizedTensor
	Wo                  []QuantizedTensor
	W1                  []QuantizedTensor
	W2                  []QuantizedTensor
	W3                  []QuantizedTensor
	Wcls                *QuantizedTensor
}

// QuantizedRunState holds runtime buffers for a quantized inference run.
type QuantizedRunState struct {
	X          []float32
	Xb         []float32
	Xb2        []float32
	Hb         []float32
	Hb2        []float32
	Q          []float32
	K          []float32
	V          []float32
	Att        []float32
	Logits     []float32
	KeyCache   []float32
	ValueCache []float32
	Xq         *QuantizedTensor
	Hq         *QuantizedTensor
}

const qMax float32 = 127.0

// Quantize converts a float32 slice into a QuantizedTensor.
func Quantize(qt *QuantizedTensor, x []float32, gs int) {
	numGroups := len(x) / gs
	for group := 0; group < numGroups; group++ {
		wmax := float32(0.0)
		offset := group * gs
		for i := range gs {
			val := float32(math.Abs(float64(x[offset+i])))
			if val > wmax {
				wmax = val
			}
		}

		scale := wmax / qMax
		if scale == 0 {
			scale = 1.0
		}

		qt.S[group] = scale
		for i := range gs {
			quantValue := x[offset+i] / scale
			qt.Q[offset+i] = int8(math.Round(float64(quantValue)))
		}
	}
}

// Dequantize converts a QuantizedTensor back to a float32 slice.
func Dequantize(qt *QuantizedTensor, x []float32, gs int) {
	for i := range x {
		x[i] = float32(qt.Q[i]) * qt.S[i/gs]
	}
}

// LoadQuantizedCheckpoint reads a quantized model file (version 2).
func LoadQuantizedCheckpoint(path string) (*QuantizedConfig, *QuantizedTransformerWeights, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()

	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, nil, 0, err
	}
	if magic != 0x616b3432 { // "ak42" in ASCII
		return nil, nil, 0, fmt.Errorf("bad magic number: %x", magic)
	}

	var version int32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, nil, 0, err
	}
	if version != 2 {
		return nil, nil, 0, fmt.Errorf("bad version: %d, expected 2", version)
	}

	var config QuantizedConfig
	if err := binary.Read(f, binary.LittleEndian, &config); err != nil {
		return nil, nil, 0, err
	}

	var sharedClassifier uint8
	if err := binary.Read(f, binary.LittleEndian, &sharedClassifier); err != nil {
		return nil, nil, 0, err
	}

	var groupSize int32
	if err := binary.Read(f, binary.LittleEndian, &groupSize); err != nil {
		return nil, nil, 0, err
	}
	gs := int(groupSize)

	const headerTotalSize = 256
	if _, err := f.Seek(headerTotalSize, io.SeekStart); err != nil {
		return nil, nil, 0, err
	}

	weights := &QuantizedTransformerWeights{}
	kvDim := (config.Dim * config.NKvHeads) / config.NHeads

	// Helper function to read a single QuantizedTensor from the file
	readQT := func(size int32, currentGs int) (*QuantizedTensor, error) {
		t := &QuantizedTensor{
			Q: make([]int8, size),
			S: make([]float32, size/int32(currentGs)),
		}
		// The binary file stores Quantized values (Q) BEFORE scaling factors (S).
		if err := binary.Read(f, binary.LittleEndian, &t.Q); err != nil {
			return nil, fmt.Errorf("failed reading quantized values: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &t.S); err != nil {
			return nil, fmt.Errorf("failed reading scales: %w", err)
		}
		return t, nil
	}

	readQTLayered := func(count, size int32, currentGs int) ([]QuantizedTensor, error) {
		tensors := make([]QuantizedTensor, count)
		for i := range tensors {
			t, err := readQT(size, currentGs)
			if err != nil {
				return nil, fmt.Errorf("layer %d: %w", i, err)
			}
			tensors[i] = *t
		}
		return tensors, nil
	}

	// Read float32 RMS weights (these are not quantized)
	weights.RmsAttWeight = make([]float32, config.NLayers*config.Dim)
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsAttWeight); err != nil {
		return nil, nil, 0, err
	}
	weights.RmsFfnWeight = make([]float32, config.NLayers*config.Dim)
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsFfnWeight); err != nil {
		return nil, nil, 0, err
	}
	weights.RmsFinalWeight = make([]float32, config.Dim)
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsFinalWeight); err != nil {
		return nil, nil, 0, err
	}

	// Read quantized weights
	weights.QTokens, err = readQT(config.VocabSize*config.Dim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.TokenEmbeddingTable = make([]float32, config.VocabSize*config.Dim)
	Dequantize(weights.QTokens, weights.TokenEmbeddingTable, gs)

	weights.Wq, err = readQTLayered(config.NLayers, config.Dim*config.Dim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.Wk, err = readQTLayered(config.NLayers, config.Dim*kvDim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.Wv, err = readQTLayered(config.NLayers, config.Dim*kvDim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.Wo, err = readQTLayered(config.NLayers, config.Dim*config.Dim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.W1, err = readQTLayered(config.NLayers, config.Dim*config.HiddenDim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.W2, err = readQTLayered(config.NLayers, config.HiddenDim*config.Dim, gs)
	if err != nil {
		return nil, nil, 0, err
	}
	weights.W3, err = readQTLayered(config.NLayers, config.Dim*config.HiddenDim, gs)
	if err != nil {
		return nil, nil, 0, err
	}

	if sharedClassifier != 0 {
		weights.Wcls = weights.QTokens
	} else {
		weights.Wcls, err = readQT(config.Dim*config.VocabSize, gs)
		if err != nil {
			return nil, nil, 0, err
		}
	}

	return &config, weights, gs, nil
}

// NewQuantizedRunState allocates buffers for a quantized run.
func NewQuantizedRunState(c *QuantizedConfig, gs int) *QuantizedRunState {
	kvDim := (c.Dim * c.NKvHeads) / c.NHeads
	s := &QuantizedRunState{
		X:          make([]float32, c.Dim),
		Xb:         make([]float32, c.Dim),
		Xb2:        make([]float32, c.Dim),
		Hb:         make([]float32, c.HiddenDim),
		Hb2:        make([]float32, c.HiddenDim),
		Q:          make([]float32, c.Dim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, c.NHeads*c.SeqLen),
		Logits:     make([]float32, c.VocabSize),
		KeyCache:   make([]float32, c.NLayers*c.SeqLen*kvDim),
		ValueCache: make([]float32, c.NLayers*c.SeqLen*kvDim),
		Xq:         &QuantizedTensor{Q: make([]int8, c.Dim), S: make([]float32, int(c.Dim)/gs)},
		Hq:         &QuantizedTensor{Q: make([]int8, c.HiddenDim), S: make([]float32, int(c.HiddenDim)/gs)},
	}
	return s
}

// TransformerQ8 performs a full forward pass using quantized weights.
func TransformerQ8(token, pos int32, c *QuantizedConfig, s *QuantizedRunState, w *QuantizedTransformerWeights, gs int) {
	copy(s.X, w.TokenEmbeddingTable[token*c.Dim:(token+1)*c.Dim])

	for l := int32(0); l < c.NLayers; l++ {
		AttentionQ8(l, pos, c, s, w, gs)
		FeedForwardQ8(l, c, s, w, gs)
	}

	RMSNorm(s.X, s.X, w.RmsFinalWeight)

	Quantize(s.Xq, s.X, gs)
	MatmulQ8(s.Logits, s.Xq, w.Wcls, gs)
}

// AttentionQ8 performs the multi-head attention for a single quantized layer.
func AttentionQ8(l, pos int32, c *QuantizedConfig, s *QuantizedRunState, w *QuantizedTransformerWeights, gs int) {
	dim, heads := c.Dim, c.NHeads
	kvDim := (c.Dim * c.NKvHeads) / c.NHeads
	kvMul := heads / c.NKvHeads
	headSize := dim / heads

	RMSNorm(s.Xb, s.X, w.RmsAttWeight[l*dim:(l+1)*dim])

	Quantize(s.Xq, s.Xb, gs)
	MatmulQ8(s.Q, s.Xq, &w.Wq[l], gs)
	MatmulQ8(s.K, s.Xq, &w.Wk[l], gs)
	MatmulQ8(s.V, s.Xq, &w.Wv[l], gs)

	for i := int32(0); i < dim; i += 2 {
		headDim := i % headSize
		freq := 1.0 / float32(math.Pow(10000.0, float64(headDim)/float64(headSize)))
		val := float64(pos) * float64(freq)
		fcr, fci := float32(math.Cos(val)), float32(math.Sin(val))
		q0, q1 := s.Q[i], s.Q[i+1]
		s.Q[i], s.Q[i+1] = q0*fcr-q1*fci, q0*fci+q1*fcr
	}
	for i := int32(0); i < kvDim; i += 2 {
		headDim := i % headSize
		freq := 1.0 / float32(math.Pow(10000.0, float64(headDim)/float64(headSize)))
		val := float64(pos) * float64(freq)
		fcr, fci := float32(math.Cos(val)), float32(math.Sin(val))
		k0, k1 := s.K[i], s.K[i+1]
		s.K[i], s.K[i+1] = k0*fcr-k1*fci, k0*fci+k1*fcr
	}

	cacheLoff := l * c.SeqLen * kvDim
	copy(s.KeyCache[cacheLoff+pos*kvDim:(cacheLoff+(pos+1)*kvDim)], s.K)
	copy(s.ValueCache[cacheLoff+pos*kvDim:(cacheLoff+(pos+1)*kvDim)], s.V)

	var wg sync.WaitGroup
	for h_idx := range heads {
		h := int32(h_idx)
		wg.Go(func() {
			q := s.Q[h*headSize : (h+1)*headSize]
			att := s.Att[h*c.SeqLen : (h+1)*c.SeqLen]
			for t := int32(0); t <= pos; t++ {
				kvHead := h / kvMul
				k := s.KeyCache[cacheLoff+t*kvDim+kvHead*headSize : cacheLoff+t*kvDim+(kvHead+1)*headSize]
				score := float32(0)
				for i := range headSize {
					score += q[i] * k[i]
				}
				att[t] = score / float32(math.Sqrt(float64(headSize)))
			}
			Softmax(att[:pos+1])
			xb_head := s.Xb[h*headSize : (h+1)*headSize]
			for i := range xb_head {
				xb_head[i] = 0
			}
			for t := int32(0); t <= pos; t++ {
				kvHead := h / kvMul
				v := s.ValueCache[cacheLoff+t*kvDim+kvHead*headSize : cacheLoff+t*kvDim+(kvHead+1)*headSize]
				a := att[t]
				for i := range headSize {
					xb_head[i] += a * v[i]
				}
			}
		})
	}
	wg.Wait()

	Quantize(s.Xq, s.Xb, gs)
	MatmulQ8(s.Xb2, s.Xq, &w.Wo[l], gs)

	for i := range s.X {
		s.X[i] += s.Xb2[i]
	}
}

// FeedForwardQ8 performs the feed-forward network for a single quantized layer.
func FeedForwardQ8(l int32, c *QuantizedConfig, s *QuantizedRunState, w *QuantizedTransformerWeights, gs int) {
	dim := c.Dim
	RMSNorm(s.Xb, s.X, w.RmsFfnWeight[l*dim:(l+1)*dim])

	Quantize(s.Xq, s.Xb, gs)
	MatmulQ8(s.Hb, s.Xq, &w.W1[l], gs)
	MatmulQ8(s.Hb2, s.Xq, &w.W3[l], gs)

	SwiGLUQ8(s)

	Quantize(s.Hq, s.Hb, gs)
	MatmulQ8(s.Xb, s.Hq, &w.W2[l], gs)

	for i := range s.X {
		s.X[i] += s.Xb[i]
	}
}

// SwiGLUQ8 performs the SwiGLU non-linearity on float32 activations.
func SwiGLUQ8(s *QuantizedRunState) {
	for i := range s.Hb {
		val := s.Hb[i]
		val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		val *= s.Hb2[i]
		s.Hb[i] = val
	}
}

// MatmulQ8 performs quantized matrix-vector multiplication.
func MatmulQ8(xout []float32, xq *QuantizedTensor, wq *QuantizedTensor, gs int) {
	n := len(xq.Q)
	d := len(xout)

	for i := range d {
		var val float32 = 0.0
		wOffset := i * n

		for j := 0; j < n; j += gs {
			var ival int32 = 0
			xOffset := j
			currentWOffset := wOffset + j
			for k := range gs {
				ival += int32(xq.Q[xOffset+k]) * int32(wq.Q[currentWOffset+k])
			}
			val += float32(ival) * wq.S[currentWOffset/gs] * xq.S[xOffset/gs]
		}
		xout[i] = val
	}
}
