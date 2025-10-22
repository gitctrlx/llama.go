package llama

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

// ProbIndex is used for sorting probabilities in top-p sampling.
type ProbIndex struct {
	Prob  float32
	Index int
}

// LlamaConfig defines the transformer hyperparameters, aligning with LlamaConfig in Hugging Face Transformers.
type LlamaConfig struct {
	Dim       int32 // Transformer embedding dimension (hidden_size)
	HiddenDim int32 // FFN intermediate dimension (intermediate_size)
	NLayers   int32 // Number of decoder layers (num_hidden_layers)
	NHeads    int32 // Number of query attention heads (num_attention_heads)
	NKvHeads  int32 // Number of key/value heads for GQA (num_key_value_heads)
	VocabSize int32 // Vocabulary size (vocab_size)
	SeqLen    int32 // Maximum context length (max_position_embeddings)
}

// LlamaLayerWeights groups weights for a single decoder layer.
type LlamaLayerWeights struct {
	// Self-attention
	AttnNorm []float32 // Input RMSNorm weights (input_layernorm)
	QProj    []float32 // Query projection (self_attn.q_proj.weight)
	KProj    []float32 // Key projection (self_attn.k_proj.weight)
	VProj    []float32 // Value projection (self_attn.v_proj.weight)
	OProj    []float32 // Output projection (self_attn.o_proj.weight)
	// FFN
	FFNNorm  []float32 // Post-attention RMSNorm weights (post_attention_layernorm)
	GateProj []float32 // Gate projection in MLP (mlp.gate_proj.weight)
	UpProj   []float32 // Up projection in MLP (mlp.up_proj.weight)
	DownProj []float32 // Down projection in MLP (mlp.down_proj.weight)
}

// LlamaWeights holds all model parameters, aligning with LlamaModel weights in Transformers.
type LlamaWeights struct {
	EmbedTokens []float32           // Token embeddings (model.embed_tokens.weight)
	Layers      []LlamaLayerWeights // Decoder layers (model.layers)
	Norm        []float32           // Final RMSNorm (model.norm.weight)
}

// LlamaState holds runtime buffers for inference, aligning with forward pass states.
type LlamaState struct {
	X          []float32   // Current hidden state (hidden_states)
	Xb         []float32   // Buffer for attention output before projection
	Xb2        []float32   // Temp buffer for attention projection output
	Hb         []float32   // FFN gate activation buffer
	Hb2        []float32   // FFN up activation buffer
	Q          []float32   // Query vector
	K          []float32   // Key vector
	V          []float32   // Value vector
	Att        [][]float32 // Attention scores per head [n_heads][seq_len]
	Logits     []float32   // Output logits
	KeyCache   [][]float32 // Key cache [n_layers][seq_len * kv_dim]
	ValueCache [][]float32 // Value cache [n_layers][seq_len * kv_dim]
}

const rmsEps = 1e-5 // RMSNorm epsilon, aligning with rms_norm_eps

// LoadLlamaModel reads config and weights from a binary checkpoint file.
func LoadLlamaModel(path string) (*LlamaConfig, *LlamaWeights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var c LlamaConfig
	if err := binary.Read(f, binary.LittleEndian, &c); err != nil {
		return nil, nil, err
	}

	dim := int(c.Dim)
	hdim := int(c.HiddenDim)
	layers := int(c.NLayers)
	vocab := int(c.VocabSize)
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)

	// Allocate structured weights
	w := &LlamaWeights{
		EmbedTokens: make([]float32, vocab*dim),
		Layers:      make([]LlamaLayerWeights, layers),
		Norm:        make([]float32, dim),
	}

	// Read flat buffers
	if err := binary.Read(f, binary.LittleEndian, w.EmbedTokens); err != nil {
		return nil, nil, err
	}

	rmsAttFlat := make([]float32, layers*dim)
	if err := binary.Read(f, binary.LittleEndian, rmsAttFlat); err != nil {
		return nil, nil, err
	}

	wqFlat := make([]float32, layers*dim*dim)
	if err := binary.Read(f, binary.LittleEndian, wqFlat); err != nil {
		return nil, nil, err
	}

	wkFlat := make([]float32, layers*dim*kvDim)
	if err := binary.Read(f, binary.LittleEndian, wkFlat); err != nil {
		return nil, nil, err
	}

	wvFlat := make([]float32, layers*dim*kvDim)
	if err := binary.Read(f, binary.LittleEndian, wvFlat); err != nil {
		return nil, nil, err
	}

	woFlat := make([]float32, layers*dim*dim)
	if err := binary.Read(f, binary.LittleEndian, woFlat); err != nil {
		return nil, nil, err
	}

	rmsFfnFlat := make([]float32, layers*dim)
	if err := binary.Read(f, binary.LittleEndian, rmsFfnFlat); err != nil {
		return nil, nil, err
	}

	gateFlat := make([]float32, layers*hdim*dim)
	if err := binary.Read(f, binary.LittleEndian, gateFlat); err != nil {
		return nil, nil, err
	}

	downFlat := make([]float32, layers*dim*hdim)
	if err := binary.Read(f, binary.LittleEndian, downFlat); err != nil {
		return nil, nil, err
	}

	upFlat := make([]float32, layers*hdim*dim)
	if err := binary.Read(f, binary.LittleEndian, upFlat); err != nil {
		return nil, nil, err
	}

	if err := binary.Read(f, binary.LittleEndian, w.Norm); err != nil {
		return nil, nil, err
	}

	// Assign to per-layer structures
	for l := 0; l < layers; l++ {
		layer := &w.Layers[l]

		off := l * dim
		layer.AttnNorm = make([]float32, dim)
		copy(layer.AttnNorm, rmsAttFlat[off:off+dim])

		off = l * dim * dim
		layer.QProj = make([]float32, dim*dim)
		copy(layer.QProj, wqFlat[off:off+dim*dim])

		off = l * dim * kvDim
		layer.KProj = make([]float32, dim*kvDim)
		copy(layer.KProj, wkFlat[off:off+dim*kvDim])

		layer.VProj = make([]float32, dim*kvDim)
		copy(layer.VProj, wvFlat[off:off+dim*kvDim])

		off = l * dim * dim
		layer.OProj = make([]float32, dim*dim)
		copy(layer.OProj, woFlat[off:off+dim*dim])

		off = l * dim
		layer.FFNNorm = make([]float32, dim)
		copy(layer.FFNNorm, rmsFfnFlat[off:off+dim])

		off = l * hdim * dim
		layer.GateProj = make([]float32, hdim*dim)
		copy(layer.GateProj, gateFlat[off:off+hdim*dim])

		off = l * dim * hdim
		layer.DownProj = make([]float32, dim*hdim)
		copy(layer.DownProj, downFlat[off:off+dim*hdim])

		off = l * hdim * dim
		layer.UpProj = make([]float32, hdim*dim)
		copy(layer.UpProj, upFlat[off:off+hdim*dim])
	}

	return &c, w, nil
}

// NewLlamaState allocates inference buffers based on config.
func NewLlamaState(c *LlamaConfig) *LlamaState {
	dim := int(c.Dim)
	hdim := int(c.HiddenDim)
	heads := int(c.NHeads)
	seq := int(c.SeqLen)
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)

	s := &LlamaState{
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

// LlamaForward performs a single-token forward pass, aligning with LlamaModel.forward.
func LlamaForward(token int32, pos int32, c *LlamaConfig, s *LlamaState, w *LlamaWeights) {
	dim := c.Dim

	// Token embedding
	copy(s.X, w.EmbedTokens[token*dim:(token+1)*dim])

	// Decoder layers
	for l := int32(0); l < c.NLayers; l++ {
		// Self-attention block
		llamaAttention(l, pos, c, s, w.Layers[l])

		// FFN block
		llamaMLP(c, s, w.Layers[l])
	}

	// Final norm
	rmsNorm(s.X, s.X, w.Norm)

	// Logits (using tied embeddings)
	matmul(s.Logits, s.X, w.EmbedTokens)
}

// llamaAttention computes self-attention for one layer, aligning with LlamaAttention.forward.
func llamaAttention(layerIdx int32, pos int32, c *LlamaConfig, s *LlamaState, lw LlamaLayerWeights) {
	dim := c.Dim
	heads := c.NHeads
	headSize := dim / heads
	kvDim := int((c.Dim * c.NKvHeads) / c.NHeads)
	groupSize := heads / c.NKvHeads

	// Input norm
	rmsNorm(s.Xb, s.X, lw.AttnNorm)

	// QKV projections
	matmul(s.Q, s.Xb, lw.QProj)
	matmul(s.K, s.Xb, lw.KProj)
	matmul(s.V, s.Xb, lw.VProj)

	// Apply RoPE
	applyRotaryEmb(s.Q, pos, headSize)
	applyRotaryEmb(s.K, pos, headSize)

	// Cache K and V
	layerKeyCache := s.KeyCache[layerIdx]
	copy(layerKeyCache[pos*int32(kvDim):(pos+1)*int32(kvDim)], s.K)
	layerValueCache := s.ValueCache[layerIdx]
	copy(layerValueCache[pos*int32(kvDim):(pos+1)*int32(kvDim)], s.V)

	// Multi-head attention (parallelized)
	var wg sync.WaitGroup
	wg.Add(int(heads))
	for h := range heads {
		go func(h int32) {
			defer wg.Done()

			// Query slice for head
			qOff := int(h * headSize)
			q := s.Q[qOff : qOff+int(headSize)]

			// Attention scores
			att := s.Att[h]

			kvH := h / groupSize
			for t := int32(0); t <= pos; t++ {
				kOff := int(t)*kvDim + int(kvH*headSize)
				k := layerKeyCache[kOff : kOff+int(headSize)]

				var score float32
				for i := range headSize {
					score += q[i] * k[i]
				}
				score /= float32(math.Sqrt(float64(headSize)))

				att[t] = score
			}

			// Softmax
			softmax(att[:pos+1])

			// Weighted values (output to Xb per head)
			xbOff := int(h * headSize)
			xbHead := s.Xb[xbOff : xbOff+int(headSize)]
			for i := range xbHead {
				xbHead[i] = 0
			}
			for t := int32(0); t <= pos; t++ {
				vOff := int(t)*kvDim + int(kvH*headSize)
				v := layerValueCache[vOff : vOff+int(headSize)]

				a := att[t]
				for i := range headSize {
					xbHead[i] += a * v[i]
				}
			}
		}(h)
	}
	wg.Wait()

	// Output projection
	matmul(s.Xb2, s.Xb, lw.OProj)

	// Residual add
	accum(s.X, s.Xb2)
}

// llamaMLP computes the FFN for one layer, aligning with LlamaMLP.forward.
func llamaMLP(c *LlamaConfig, s *LlamaState, lw LlamaLayerWeights) {
	// Input norm
	rmsNorm(s.Xb, s.X, lw.FFNNorm)

	// Gate and up projections
	matmul(s.Hb, s.Xb, lw.GateProj)
	matmul(s.Hb2, s.Xb, lw.UpProj)

	// SwiGLU activation
	for i := int32(0); i < c.HiddenDim; i++ {
		val := s.Hb[i]
		val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		val *= s.Hb2[i]
		s.Hb[i] = val
	}

	// Down projection
	matmul(s.Xb, s.Hb, lw.DownProj)

	// Residual add
	accum(s.X, s.Xb)
}

// applyRotaryEmb applies rotary positional embeddings, aligning with apply_rotary_pos_emb.
func applyRotaryEmb(x []float32, pos int32, headSize int32) {
	for i := int32(0); i < int32(len(x)); i += 2 {
		headDim := i % headSize
		freq := 1.0 / float32(math.Pow(10000.0, float64(headDim)/float64(headSize)))
		val := float64(pos) * float64(freq)
		fcr := float32(math.Cos(val))
		fci := float32(math.Sin(val))

		x0, x1 := x[i], x[i+1]
		x[i] = x0*fcr - x1*fci
		x[i+1] = x0*fci + x1*fcr
	}
}

// matmul computes matrix-vector multiplication: xout = x @ w.T (assuming w is row-major flattened).
func matmul(xout, x, w []float32) {
	inDim := len(x)
	outDim := len(xout)
	for i := 0; i < outDim; i++ {
		var val float32
		off := i * inDim
		for j := 0; j < inDim; j++ {
			val += w[off+j] * x[j]
		}
		xout[i] = val
	}
}

// accum adds b to a element-wise.
func accum(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// rmsNorm applies RMS normalization, aligning with LlamaRMSNorm.forward.
func rmsNorm(dest, src, weight []float32) {
	n := len(src)
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	ss = ss/float32(n) + rmsEps
	inv := 1 / float32(math.Sqrt(float64(ss)))
	for i := range dest {
		dest[i] = weight[i] * (inv * src[i])
	}
}

// softmax computes softmax in-place.
func softmax(x []float32) {
	n := len(x)
	if n == 0 {
		return
	}
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	var sum float32
	for i := range n {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	for i := range n {
		x[i] /= sum
	}
}

// Sample selects a token from logits with temperature and top-p.
func Sample(logits []float32, temp float64, topp float64, rng *rand.Rand) int32 {
	if temp == 0 {
		maxIdx := 0
		maxVal := logits[0]
		for i := 1; i < len(logits); i++ {
			if logits[i] > maxVal {
				maxVal = logits[i]
				maxIdx = i
			}
		}
		return int32(maxIdx)
	}

	// Scale by temperature
	for i := range logits {
		logits[i] /= float32(temp)
	}
	softmax(logits)

	r := rng.Float32()

	if topp <= 0 || topp >= 1 {
		var cdf float32
		for i, p := range logits {
			cdf += p
			if r < cdf {
				return int32(i)
			}
		}
		return int32(len(logits) - 1)
	}

	// Top-p sampling
	probIndex := make([]ProbIndex, len(logits))
	for i, p := range logits {
		probIndex[i] = ProbIndex{Prob: p, Index: i}
	}
	sort.Slice(probIndex, func(i, j int) bool {
		return probIndex[i].Prob > probIndex[j].Prob
	})

	var cumProb float32
	lastIdx := len(probIndex) - 1
	for i, pi := range probIndex {
		cumProb += pi.Prob
		if cumProb > float32(topp) {
			lastIdx = i
			break
		}
	}

	r *= cumProb
	var cdf float32
	for i := 0; i <= lastIdx; i++ {
		cdf += probIndex[i].Prob
		if r < cdf {
			return int32(probIndex[i].Index)
		}
	}
	return int32(probIndex[lastIdx].Index)
}
