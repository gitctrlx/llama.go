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

// Config defines the transformer hyperparameters.
type Config struct {
	Dim       int32 // Embedding dimension
	HiddenDim int32 // FFN hidden dimension
	NLayers   int32 // Number of transformer layers
	NHeads    int32 // Number of attention query heads
	NKvHeads  int32 // Number of key/value heads (assumed == NHeads; multiquery not fully implemented)
	VocabSize int32 // Vocabulary size
	SeqLen    int32 // Maximum sequence length
}

// TransformerWeights holds the model parameters.
type TransformerWeights struct {
	TokenEmbeddingTable []float32 // (vocab_size, dim)
	RmsAttWeight        []float32 // (n_layers, dim)
	RmsFfnWeight        []float32 // (n_layers, dim)
	Wq                  []float32 // (n_layers, dim, dim)
	Wk                  []float32 // (n_layers, dim, dim)
	Wv                  []float32 // (n_layers, dim, dim)
	Wo                  []float32 // (n_layers, dim, dim)
	W1                  []float32 // (n_layers, hidden_dim, dim)
	W2                  []float32 // (n_layers, dim, hidden_dim)
	W3                  []float32 // (n_layers, hidden_dim, dim)
	RmsFinalWeight      []float32 // (dim)
}

// RunState holds runtime buffers for inference.
type RunState struct {
	X          []float32 // Current activation (dim)
	Xb         []float32 // Residual branch buffer (dim)
	Xb2        []float32 // Additional residual buffer (dim)
	Hb         []float32 // FFN hidden buffer (hidden_dim)
	Hb2        []float32 // FFN hidden buffer (hidden_dim)
	Q          []float32 // Query vector (dim)
	K          []float32 // Key vector (dim)
	V          []float32 // Value vector (dim)
	Att        []float32 // Attention scores (n_heads, seq_len)
	Logits     []float32 // Output logits (vocab_size)
	KeyCache   []float32 // KV cache for keys (n_layers, seq_len, dim)
	ValueCache []float32 // KV cache for values (n_layers, seq_len, dim)
}

const rmsEps = 1e-5 // Epsilon for RMS norm stability

// LoadCheckpoint reads config and weights from a binary file.
func LoadCheckpoint(path string) (*Config, *TransformerWeights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var config Config
	if err := binary.Read(f, binary.LittleEndian, &config); err != nil {
		return nil, nil, err
	}

	weights := NewTransformerWeights(&config)
	if err := binary.Read(f, binary.LittleEndian, &weights.TokenEmbeddingTable); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsAttWeight); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.Wq); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.Wk); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.Wv); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.Wo); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsFfnWeight); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.W1); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.W2); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.W3); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &weights.RmsFinalWeight); err != nil {
		return nil, nil, err
	}

	return &config, weights, nil
}

// NewTransformerWeights allocates slices for weights based on config.
func NewTransformerWeights(c *Config) *TransformerWeights {
	dim := c.Dim
	hdim := c.HiddenDim
	kv_dim := c.Dim
	layers := c.NLayers
	vocab := c.VocabSize

	return &TransformerWeights{
		TokenEmbeddingTable: make([]float32, vocab*dim),
		RmsAttWeight:        make([]float32, layers*dim),
		RmsFfnWeight:        make([]float32, layers*dim),
		Wq:                  make([]float32, layers*dim*dim),
		Wk:                  make([]float32, layers*dim*kv_dim),
		Wv:                  make([]float32, layers*dim*kv_dim),
		Wo:                  make([]float32, layers*dim*dim),
		W1:                  make([]float32, layers*hdim*dim),
		W2:                  make([]float32, layers*dim*hdim),
		W3:                  make([]float32, layers*hdim*dim),
		RmsFinalWeight:      make([]float32, dim),
	}
}

// NewRunState allocates slices for run state based on config.
func NewRunState(c *Config) *RunState {
	dim := c.Dim
	hdim := c.HiddenDim
	kv_dim := c.Dim
	layers := c.NLayers
	vocab := c.VocabSize
	seq := c.SeqLen
	heads := c.NHeads

	return &RunState{
		X:          make([]float32, dim),
		Xb:         make([]float32, dim),
		Xb2:        make([]float32, dim),
		Hb:         make([]float32, hdim),
		Hb2:        make([]float32, hdim),
		Q:          make([]float32, dim),
		K:          make([]float32, kv_dim),
		V:          make([]float32, kv_dim),
		Att:        make([]float32, heads*seq),
		Logits:     make([]float32, vocab),
		KeyCache:   make([]float32, layers*seq*kv_dim),
		ValueCache: make([]float32, layers*seq*kv_dim),
	}
}

// Transformer performs a forward pass for one token at position pos.
func Transformer(token, pos int32, c *Config, s *RunState, w *TransformerWeights) {
	// Embed token into the main activation vector s.X
	copy(s.X, w.TokenEmbeddingTable[token*c.Dim:(token+1)*c.Dim])

	// Process all transformer layers
	for l := int32(0); l < c.NLayers; l++ {
		// Attention sub-layer
		Attention(l, pos, c, s, w)

		// Feed-forward sub-layer
		FeedForward(l, c, s, w)
	}

	// Final RMS norm on the output
	RMSNorm(s.X, s.X, w.RmsFinalWeight)

	// Classifier output
	Matmul(s.Logits, s.X, w.TokenEmbeddingTable)
}

// Attention performs the multi-head attention for a single layer.
func Attention(l int32, pos int32, c *Config, s *RunState, w *TransformerWeights) {
	dim := c.Dim
	heads := c.NHeads
	kv_dim := (c.Dim * c.NKvHeads) / c.NHeads
	kv_mul := heads / c.NKvHeads
	headSize := dim / heads

	// 1. Attention RMSNorm
	RMSNorm(s.Xb, s.X, w.RmsAttWeight[l*dim:(l+1)*dim])

	// 2. Q, K, V Projections
	wq_loff := l * dim * dim
	wk_loff := l * dim * kv_dim
	wv_loff := l * dim * kv_dim
	Matmul(s.Q, s.Xb, w.Wq[wq_loff:wq_loff+dim*dim])
	Matmul(s.K, s.Xb, w.Wk[wk_loff:wk_loff+dim*kv_dim])
	Matmul(s.V, s.Xb, w.Wv[wv_loff:wv_loff+dim*kv_dim])

	// 3. RoPE (Rotary Positional Encoding)
	// Apply to Q vector
	for i := int32(0); i < dim; i += 2 {
		headDim := i % headSize
		freq := 1.0 / float32(math.Pow(10000.0, float64(headDim)/float64(headSize)))
		val := float64(pos) * float64(freq)
		fcr := float32(math.Cos(val))
		fci := float32(math.Sin(val))

		q0, q1 := s.Q[i], s.Q[i+1]
		s.Q[i] = q0*fcr - q1*fci
		s.Q[i+1] = q0*fci + q1*fcr
	}
	// Apply to K vector (only up to kv_dim)
	for i := int32(0); i < kv_dim; i += 2 {
		headDim := i % headSize
		freq := 1.0 / float32(math.Pow(10000.0, float64(headDim)/float64(headSize)))
		val := float64(pos) * float64(freq)
		fcr := float32(math.Cos(val))
		fci := float32(math.Sin(val))

		k0, k1 := s.K[i], s.K[i+1]
		s.K[i] = k0*fcr - k1*fci
		s.K[i+1] = k0*fci + k1*fcr
	}

	// 4. Cache K and V for future timesteps
	cache_loff := l * c.SeqLen * kv_dim
	copy(s.KeyCache[cache_loff+pos*kv_dim:cache_loff+(pos+1)*kv_dim], s.K)
	copy(s.ValueCache[cache_loff+pos*kv_dim:cache_loff+(pos+1)*kv_dim], s.V)

	// 5. Multi-head Attention Calculation (parallelized with goroutines)
	var wg sync.WaitGroup
	wg.Add(int(heads))
	for h := range heads {
		go func(h int32) {
			defer wg.Done()

			// Get the query vector for this head
			q := s.Q[h*headSize : (h+1)*headSize]
			// Attention scores buffer for this head
			att := s.Att[h*c.SeqLen : (h+1)*c.SeqLen]

			// Iterate over all timesteps up to the current one
			for t := int32(0); t <= pos; t++ {
				// Get the key vector for this timestep
				// GQA: Map query head 'h' to the correct KV head 'kv_h'
				kv_h := h / kv_mul
				k_loff := cache_loff + t*kv_dim + kv_h*headSize
				k := s.KeyCache[k_loff : k_loff+headSize]

				// Calculate the attention score (dot product of Q and K)
				score := float32(0)
				for i := range headSize {
					score += q[i] * k[i]
				}
				score /= float32(math.Sqrt(float64(headSize)))
				att[t] = score
			}

			// Softmax the scores to get attention weights
			Softmax(att[:pos+1])

			// Weighted sum of the values
			xb_head := s.Xb[h*headSize : (h+1)*headSize]
			for i := range xb_head {
				xb_head[i] = 0
			}
			for t := int32(0); t <= pos; t++ {
				// Get the value vector for this timestep
				// GQA: Map query head 'h' to the correct KV head 'kv_h'
				kv_h := h / kv_mul
				v_loff := cache_loff + t*kv_dim + kv_h*headSize
				v := s.ValueCache[v_loff : v_loff+headSize]

				// Get the attention weight for this timestep
				a := att[t]
				// Accumulate the weighted value
				for i := range headSize {
					xb_head[i] += a * v[i]
				}
			}
		}(h)
	}
	wg.Wait()

	// 6. Output Projection
	wo_loff := l * dim * dim
	Matmul(s.Xb2, s.Xb, w.Wo[wo_loff:wo_loff+dim*dim])

	// 7. Residual Connection
	for i := range dim {
		s.X[i] += s.Xb2[i]
	}
}

// FeedForward performs the feed-forward network for a single layer.
func FeedForward(l int32, c *Config, s *RunState, w *TransformerWeights) {
	dim := c.Dim
	hdim := c.HiddenDim

	// FFN RMS norm
	RMSNorm(s.Xb, s.X, w.RmsFfnWeight[l*dim:(l+1)*dim])

	// FFN
	w1_loff := l * hdim * dim
	w3_loff := l * hdim * dim
	Matmul(s.Hb, s.Xb, w.W1[w1_loff:w1_loff+hdim*dim])
	Matmul(s.Hb2, s.Xb, w.W3[w3_loff:w3_loff+hdim*dim])

	// Apply SwiGLU non-linearity by calling our new helper function
	SwiGLU(s, c)

	// FFN output projection
	w2_loff := l * dim * hdim
	Matmul(s.Xb, s.Hb, w.W2[w2_loff:w2_loff+dim*hdim])

	// Residual connection
	for i := range dim {
		s.X[i] += s.Xb[i]
	}
}

// SwiGLU performs the SwiGLU non-linearity.
// val = (x * sigmoid(x)) * w3(x)
func SwiGLU(s *RunState, c *Config) {
	hdim := c.HiddenDim
	for i := range hdim {
		val := s.Hb[i]
		// silu(x) = x * sigmoid(x)
		val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
		// elementwise multiply with w3(x)
		val *= s.Hb2[i]
		s.Hb[i] = val
	}
}

// Matmul computes xout = w * x (matrix-vector multiply).
func Matmul(xout, x, w []float32) {
	n := len(x)     // Input dim
	d := len(w) / n // Output dim
	for i := range d {
		var val float32
		for j := range n {
			val += w[i*n+j] * x[j]
		}
		xout[i] = val
	}
}

// Accum adds b to a element-wise.
func Accum(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// RMSNorm normalizes src into dest using weight (RMS normalization).
func RMSNorm(dest, src, weight []float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	ss = ss/float32(len(src)) + rmsEps
	ss = 1 / float32(math.Sqrt(float64(ss)))
	for i, v := range src {
		dest[i] = weight[i] * (ss * v)
	}
}

// Softmax normalizes x in-place to probabilities.
func Softmax(x []float32) {
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

// Sample selects a token from logits using temperature, top-p, or greedy sampling.
func Sample(logits []float32, temp float64, topp float64, rng *rand.Rand) int32 {
	if temp == 0 {
		// Greedy sampling: return the token with the highest logit.
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

	// Temperature scaling.
	for i := range logits {
		logits[i] /= float32(temp)
	}
	// Compute softmax in-place to get probabilities.
	Softmax(logits)

	r := rng.Float32()

	// If topp is not used, perform simple temperature sampling.
	if topp <= 0 || topp >= 1 {
		var cdf float32
		for i, p := range logits {
			cdf += p
			if r < cdf {
				return int32(i)
			}
		}
		return int32(len(logits) - 1) // Fallback.
	}

	// Top-p (nucleus) sampling.
	// 1. Create and sort probabilities along with their original indices.
	probIndex := make([]ProbIndex, len(logits))
	for i, p := range logits {
		probIndex[i] = ProbIndex{Prob: p, Index: i}
	}
	sort.Slice(probIndex, func(i, j int) bool {
		return probIndex[i].Prob > probIndex[j].Prob
	})

	// 2. Find the nucleus: the smallest set of tokens whose cumulative probability exceeds topp.
	var cumProb float32
	lastIdx := 0
	for i, pi := range probIndex {
		cumProb += pi.Prob
		if cumProb > float32(topp) {
			lastIdx = i
			break // The nucleus is probIndex[:lastIdx+1]
		}
	}

	// 3. Sample from the nucleus.
	r = r * cumProb // Rescale random number to the range [0, cumProb).
	var cdf float32
	for i := 0; i <= lastIdx; i++ {
		cdf += probIndex[i].Prob
		if r < cdf {
			return int32(probIndex[i].Index)
		}
	}

	return int32(probIndex[lastIdx].Index) // Fallback for rounding errors.
}
