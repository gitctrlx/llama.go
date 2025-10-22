package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
)

// QuantizedTensor represents int8 quantized data with scales.
type QuantizedTensor struct {
	Q []int8    // Quantized values
	S []float32 // Per-group scales
}

// LlamaQuantizedLayerWeights holds quantized weights for a decoder layer.
type LlamaQuantizedLayerWeights struct {
	AttnNorm []float32       // input_layernorm
	QProj    QuantizedTensor // self_attn.q_proj
	KProj    QuantizedTensor // self_attn.k_proj
	VProj    QuantizedTensor // self_attn.v_proj
	OProj    QuantizedTensor // self_attn.o_proj
	FFNNorm  []float32       // post_attention_layernorm
	GateProj QuantizedTensor // mlp.gate_proj
	UpProj   QuantizedTensor // mlp.up_proj
	DownProj QuantizedTensor // mlp.down_proj
}

// LlamaQuantizedWeights holds all quantized model parameters.
type LlamaQuantizedWeights struct {
	EmbedTokens []float32                    // model.embed_tokens
	Layers      []LlamaQuantizedLayerWeights // model.layers
	Norm        []float32                    // model.norm
	Output      QuantizedTensor              // lm_head (Wcls)
}

// LlamaQuantizedState holds runtime buffers for quantized inference.
type LlamaQuantizedState struct {
	X          []float32       // hidden_states
	Xb         []float32       // temp for attention
	Xb2        []float32       // temp for attention output
	Hb         []float32       // mlp gate activation
	Hb2        []float32       // mlp up activation
	Q          []float32       // query
	K          []float32       // key
	V          []float32       // value
	Logits     []float32       // output logits
	Att        [][]float32     // attention scores [n_heads][seq_len]
	KeyCache   [][]float32     // key cache [n_layers][seq_len * kv_dim]
	ValueCache [][]float32     // value cache [n_layers][seq_len * kv_dim]
	Xq         QuantizedTensor // quantized X
	Hq         QuantizedTensor // quantized Hb
}

const qMax float32 = 127.0

// LoadLlamaQuantizedModel loads a quantized checkpoint (version 2).
func LoadLlamaQuantizedModel(path string) (*LlamaConfig, *LlamaQuantizedWeights, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()

	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, nil, 0, err
	}
	if magic != 0x616b3432 {
		return nil, nil, 0, fmt.Errorf("invalid magic: %x", magic)
	}

	var version int32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, nil, 0, err
	}
	if version != 2 {
		return nil, nil, 0, fmt.Errorf("unsupported version: %d", version)
	}

	var c LlamaConfig
	if err := binary.Read(f, binary.LittleEndian, &c); err != nil {
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

	const headerSize = 256
	if _, err := f.Seek(headerSize, io.SeekStart); err != nil {
		return nil, nil, 0, err
	}

	dim := int(c.Dim)
	hdim := int(c.HiddenDim)
	layersNum := int(c.NLayers)
	vocab := int(c.VocabSize)
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)

	w := &LlamaQuantizedWeights{
		EmbedTokens: make([]float32, vocab*dim),
		Layers:      make([]LlamaQuantizedLayerWeights, layersNum),
		Norm:        make([]float32, dim),
	}

	readQT := func(size int) (*QuantizedTensor, error) {
		t := &QuantizedTensor{
			Q: make([]int8, size),
			S: make([]float32, size/gs),
		}
		if err := binary.Read(f, binary.LittleEndian, t.Q); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, t.S); err != nil {
			return nil, err
		}
		return t, nil
	}

	readQTLayered := func(count, size int) ([]QuantizedTensor, error) {
		tensors := make([]QuantizedTensor, count)
		for i := range tensors {
			t, err := readQT(size)
			if err != nil {
				return nil, err
			}
			tensors[i] = *t
		}
		return tensors, nil
	}

	rmsAttFlat := make([]float32, layersNum*dim)
	if err := binary.Read(f, binary.LittleEndian, rmsAttFlat); err != nil {
		return nil, nil, 0, err
	}

	rmsFfnFlat := make([]float32, layersNum*dim)
	if err := binary.Read(f, binary.LittleEndian, rmsFfnFlat); err != nil {
		return nil, nil, 0, err
	}

	if err := binary.Read(f, binary.LittleEndian, w.Norm); err != nil {
		return nil, nil, 0, err
	}

	qTokens, err := readQT(vocab * dim)
	if err != nil {
		return nil, nil, 0, err
	}
	dequantize(qTokens, w.EmbedTokens, gs)

	wq, err := readQTLayered(layersNum, dim*dim)
	if err != nil {
		return nil, nil, 0, err
	}
	wk, err := readQTLayered(layersNum, dim*kvDim)
	if err != nil {
		return nil, nil, 0, err
	}
	wv, err := readQTLayered(layersNum, dim*kvDim)
	if err != nil {
		return nil, nil, 0, err
	}
	wo, err := readQTLayered(layersNum, dim*dim)
	if err != nil {
		return nil, nil, 0, err
	}
	w1, err := readQTLayered(layersNum, hdim*dim)
	if err != nil {
		return nil, nil, 0, err
	}
	w2, err := readQTLayered(layersNum, dim*hdim)
	if err != nil {
		return nil, nil, 0, err
	}
	w3, err := readQTLayered(layersNum, hdim*dim)
	if err != nil {
		return nil, nil, 0, err
	}

	for l := 0; l < layersNum; l++ {
		layer := &w.Layers[l]
		off := l * dim
		layer.AttnNorm = make([]float32, dim)
		copy(layer.AttnNorm, rmsAttFlat[off:off+dim])
		layer.FFNNorm = make([]float32, dim)
		copy(layer.FFNNorm, rmsFfnFlat[off:off+dim])

		layer.QProj = wq[l]
		layer.KProj = wk[l]
		layer.VProj = wv[l]
		layer.OProj = wo[l]
		layer.GateProj = w1[l]
		layer.DownProj = w2[l]
		layer.UpProj = w3[l]
	}

	if sharedClassifier != 0 {
		w.Output = *qTokens
	} else {
		qt, err := readQT(dim * vocab)
		if err != nil {
			return nil, nil, 0, err
		}
		w.Output = *qt
	}

	return &c, w, gs, nil
}

// NewLlamaQuantizedState allocates buffers for quantized inference.
func NewLlamaQuantizedState(c *LlamaConfig, gs int) *LlamaQuantizedState {
	dim := int(c.Dim)
	hdim := int(c.HiddenDim)
	heads := int(c.NHeads)
	seq := int(c.SeqLen)
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)

	s := &LlamaQuantizedState{
		X:          make([]float32, dim),
		Xb:         make([]float32, dim),
		Xb2:        make([]float32, dim),
		Hb:         make([]float32, hdim),
		Hb2:        make([]float32, hdim),
		Q:          make([]float32, dim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Logits:     make([]float32, c.VocabSize),
		Att:        make([][]float32, heads),
		KeyCache:   make([][]float32, c.NLayers),
		ValueCache: make([][]float32, c.NLayers),
		Xq:         QuantizedTensor{Q: make([]int8, dim), S: make([]float32, dim/gs)},
		Hq:         QuantizedTensor{Q: make([]int8, hdim), S: make([]float32, hdim/gs)},
	}

	for i := range s.Att {
		s.Att[i] = make([]float32, seq)
	}
	for i := range s.KeyCache {
		s.KeyCache[i] = make([]float32, seq*kvDim)
		s.ValueCache[i] = make([]float32, seq*kvDim)
	}

	return s
}

// LlamaForwardQuantized performs a single-token forward pass with quantization.
func LlamaForwardQuantized(token int32, pos int32, c *LlamaConfig, s *LlamaQuantizedState, w *LlamaQuantizedWeights, gs int) {
	dim := c.Dim

	copy(s.X, w.EmbedTokens[token*dim:(token+1)*dim])

	for l := int32(0); l < c.NLayers; l++ {
		llamaAttentionQuantized(l, pos, c, s, w.Layers[l], gs)
		llamaMLPQuantized(s, w.Layers[l], gs)
	}

	rmsNorm(s.X, s.X, w.Norm)

	quantize(&s.Xq, s.X, gs)
	matmulQuantized(s.Logits, &s.Xq, &w.Output, gs)
}

// llamaAttentionQuantized computes quantized self-attention for one layer.
func llamaAttentionQuantized(layerIdx int32, pos int32, c *LlamaConfig, s *LlamaQuantizedState, lw LlamaQuantizedLayerWeights, gs int) {
	dim := int(c.Dim)
	heads := int(c.NHeads)
	headSize := dim / heads
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)
	groupSize := heads / int(c.NKvHeads)

	rmsNorm(s.Xb, s.X, lw.AttnNorm)

	quantize(&s.Xq, s.Xb, gs)
	matmulQuantized(s.Q, &s.Xq, &lw.QProj, gs)
	matmulQuantized(s.K, &s.Xq, &lw.KProj, gs)
	matmulQuantized(s.V, &s.Xq, &lw.VProj, gs)

	applyRotaryEmb(s.Q, pos, int32(headSize))
	applyRotaryEmb(s.K[:kvDim], pos, int32(headSize)) // Apply to K up to kvDim

	layerKeyCache := s.KeyCache[layerIdx]
	copy(layerKeyCache[pos*int32(kvDim):(pos+1)*int32(kvDim)], s.K)
	layerValueCache := s.ValueCache[layerIdx]
	copy(layerValueCache[pos*int32(kvDim):(pos+1)*int32(kvDim)], s.V)

	var wg sync.WaitGroup
	wg.Add(heads)
	for h := range heads {
		go func(h int) {
			defer wg.Done()
			h32 := int32(h)

			qOff := h * headSize
			q := s.Q[qOff : qOff+headSize]

			att := s.Att[h][:pos+1]

			kvH := int(h32) / groupSize
			for t := int32(0); t <= pos; t++ {
				kOff := int(t)*kvDim + int(kvH)*headSize
				k := layerKeyCache[kOff : kOff+headSize]

				var score float32
				for i := 0; i < headSize; i++ {
					score += q[i] * k[i]
				}
				score /= float32(math.Sqrt(float64(headSize)))

				att[t] = score
			}

			softmax(att)

			xbOff := h * headSize
			xbHead := s.Xb[xbOff : xbOff+headSize]
			for i := range xbHead {
				xbHead[i] = 0
			}
			for t := int32(0); t <= pos; t++ {
				vOff := int(t)*kvDim + int(kvH)*headSize
				v := layerValueCache[vOff : vOff+headSize]

				a := att[t]
				for i := 0; i < headSize; i++ {
					xbHead[i] += a * v[i]
				}
			}
		}(h)
	}
	wg.Wait()

	quantize(&s.Xq, s.Xb, gs)
	matmulQuantized(s.Xb2, &s.Xq, &lw.OProj, gs)

	accum(s.X, s.Xb2)
}

// llamaMLPQuantized computes quantized FFN for one layer.
func llamaMLPQuantized(s *LlamaQuantizedState, lw LlamaQuantizedLayerWeights, gs int) {
	rmsNorm(s.Xb, s.X, lw.FFNNorm)

	quantize(&s.Xq, s.Xb, gs)
	matmulQuantized(s.Hb, &s.Xq, &lw.GateProj, gs)
	matmulQuantized(s.Hb2, &s.Xq, &lw.UpProj, gs)

	swiGLU(s)

	quantize(&s.Hq, s.Hb, gs)
	matmulQuantized(s.Xb, &s.Hq, &lw.DownProj, gs)

	accum(s.X, s.Xb)
}

// quantize converts float32 slice to QuantizedTensor.
func quantize(qt *QuantizedTensor, x []float32, gs int) {
	numGroups := len(x) / gs
	for g := 0; g < numGroups; g++ {
		maxVal := float32(0)
		off := g * gs
		for i := 0; i < gs; i++ {
			val := float32(math.Abs(float64(x[off+i])))
			if val > maxVal {
				maxVal = val
			}
		}

		scale := maxVal / qMax
		if scale == 0 {
			scale = 1
		}
		qt.S[g] = scale

		for i := range gs {
			qt.Q[off+i] = int8(math.Round(float64(x[off+i] / scale)))
		}
	}
}

// dequantize converts QuantizedTensor to float32 slice.
func dequantize(qt *QuantizedTensor, x []float32, gs int) {
	for i := range x {
		x[i] = float32(qt.Q[i]) * qt.S[i/gs]
	}
}

// matmulQuantized computes quantized matrix-vector mul.
func matmulQuantized(xout []float32, xq *QuantizedTensor, wq *QuantizedTensor, gs int) {
	inDim := len(xq.Q)
	outDim := len(xout)
	for i := 0; i < outDim; i++ {
		var val float32
		wOff := i * inDim
		for j := 0; j < inDim; j += gs {
			var ival int32
			for k := 0; k < gs; k++ {
				ival += int32(xq.Q[j+k]) * int32(wq.Q[wOff+j+k])
			}
			val += float32(ival) * xq.S[j/gs] * wq.S[(wOff+j)/gs]
		}
		xout[i] = val
	}
}

// swiGLU applies SwiGLU activation.
func swiGLU(s *LlamaQuantizedState) {
	for i := range s.Hb {
		val := s.Hb[i]
		val *= 1 / (1 + float32(math.Exp(float64(-val))))
		val *= s.Hb2[i]
		s.Hb[i] = val
	}
}
