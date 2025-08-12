package llama2

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
)

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
	FreqCisReal         []float32 // (seq_len, dim/2)
	FreqCisImag         []float32 // (seq_len, dim/2)
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
	// if err := binary.Read(f, binary.LittleEndian, &weights.FreqCisReal); err != nil {
	// 	return nil, nil, err
	// }
	// if err := binary.Read(f, binary.LittleEndian, &weights.FreqCisImag); err != nil {
	// 	return nil, nil, err
	// }

	return &config, weights, nil
}

// NewTransformerWeights allocates slices for weights based on config.
func NewTransformerWeights(c *Config) *TransformerWeights {
	dim := c.Dim
	hdim := c.HiddenDim
	layers := c.NLayers
	vocab := c.VocabSize
	seq := c.SeqLen

	return &TransformerWeights{
		TokenEmbeddingTable: make([]float32, vocab*dim),
		RmsAttWeight:        make([]float32, layers*dim),
		RmsFfnWeight:        make([]float32, layers*dim),
		Wq:                  make([]float32, layers*dim*dim),
		Wk:                  make([]float32, layers*dim*dim),
		Wv:                  make([]float32, layers*dim*dim),
		Wo:                  make([]float32, layers*dim*dim),
		W1:                  make([]float32, layers*hdim*dim),
		W2:                  make([]float32, layers*dim*hdim),
		W3:                  make([]float32, layers*hdim*dim),
		RmsFinalWeight:      make([]float32, dim),
		FreqCisReal:         make([]float32, seq*dim/2),
		FreqCisImag:         make([]float32, seq*dim/2),
	}
}

// NewRunState allocates slices for run state based on config.
func NewRunState(c *Config) *RunState {
	dim := c.Dim
	hdim := c.HiddenDim
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
		K:          make([]float32, dim),
		V:          make([]float32, dim),
		Att:        make([]float32, heads*seq),
		Logits:     make([]float32, vocab),
		KeyCache:   make([]float32, layers*seq*dim),
		ValueCache: make([]float32, layers*seq*dim),
	}
}

// Transformer performs a forward pass for one token at position pos.
func Transformer(token, pos int32, c *Config, s *RunState, w *TransformerWeights) {
	x := s.X
	dim := c.Dim
	hdim := c.HiddenDim
	heads := c.NHeads
	headSize := dim / heads

	// Embed token
	copy(x, w.TokenEmbeddingTable[token*dim:(token+1)*dim])

	// Precomputed RoPE frequencies
	fcr := w.FreqCisReal[pos*(headSize/2) : (pos+1)*(headSize/2)]
	fci := w.FreqCisImag[pos*(headSize/2) : (pos+1)*(headSize/2)]

	// Layer loop
	for l := int32(0); l < c.NLayers; l++ {
		// Attention RMS norm
		RMSNorm(s.Xb, x, w.RmsAttWeight[l*dim:(l+1)*dim])

		// QKV projections
		Matmul(s.Q, s.Xb, w.Wq[l*dim*dim:(l+1)*dim*dim])
		Matmul(s.K, s.Xb, w.Wk[l*dim*dim:(l+1)*dim*dim])
		Matmul(s.V, s.Xb, w.Wv[l*dim*dim:(l+1)*dim*dim])

		// Apply RoPE to Q and K per head
		for h := int32(0); h < heads; h++ {
			q := s.Q[h*headSize : (h+1)*headSize]
			k := s.K[h*headSize : (h+1)*headSize]
			for i := int32(0); i < headSize; i += 2 {
				q0, q1 := q[i], q[i+1]
				k0, k1 := k[i], k[i+1]
				r, im := fcr[i/2], fci[i/2]
				q[i] = q0*r - q1*im
				q[i+1] = q0*im + q1*r
				k[i] = k0*r - k1*im
				k[i+1] = k0*im + k1*r
			}
		}

		// Cache K and V
		loff := l * c.SeqLen * dim
		copy(s.KeyCache[loff+pos*dim:loff+(pos+1)*dim], s.K)
		copy(s.ValueCache[loff+pos*dim:loff+(pos+1)*dim], s.V)

		// Multi-head attention
		for h := int32(0); h < heads; h++ {
			q := s.Q[h*headSize : (h+1)*headSize]
			att := s.Att[h*c.SeqLen : (h+1)*c.SeqLen]

			// Compute scores
			for t := int32(0); t <= pos; t++ {
				k := s.KeyCache[loff+t*dim+h*headSize : loff+t*dim+(h+1)*headSize]
				score := float32(0)
				for i := int32(0); i < headSize; i++ {
					score += q[i] * k[i]
				}
				att[t] = score / float32(math.Sqrt(float64(headSize)))
			}

			// Softmax attention weights
			Softmax(att[:pos+1])

			// Weighted sum of values
			for i := int32(0); i < headSize; i++ {
				var val float64 // Use float64 for accumulation precision
				for t := int32(0); t <= pos; t++ {
					val += float64(att[t]) * float64(s.ValueCache[loff+t*dim+h*headSize+i])
				}
				s.Xb[h*headSize+i] = float32(val)
			}
		}

		// Output projection
		Matmul(s.Xb2, s.Xb, w.Wo[l*dim*dim:(l+1)*dim*dim])

		// Residual
		Accum(x, s.Xb2)

		// FFN RMS norm
		RMSNorm(s.Xb, x, w.RmsFfnWeight[l*dim:(l+1)*dim])

		// FFN: w1 and w3
		Matmul(s.Hb, s.Xb, w.W1[l*dim*hdim:(l+1)*dim*hdim])
		Matmul(s.Hb2, s.Xb, w.W3[l*dim*hdim:(l+1)*dim*hdim])

		// SiLU activation on hb
		for i := int32(0); i < hdim; i++ {
			s.Hb[i] *= 1 / (1 + float32(math.Exp(-float64(s.Hb[i]))))
		}

		// Element-wise multiply hb *= hb2
		for i := int32(0); i < hdim; i++ {
			s.Hb[i] *= s.Hb2[i]
		}

		// FFN output projection
		Matmul(s.Xb, s.Hb, w.W2[l*dim*hdim:(l+1)*dim*hdim])

		// Residual
		Accum(x, s.Xb)
	}

	// Final RMS norm
	RMSNorm(x, x, w.RmsFinalWeight)

	// Classifier to logits (reuse embedding table)
	Matmul(s.Logits, x, w.TokenEmbeddingTable)
}

// Matmul computes xout = w * x (matrix-vector multiply).
func Matmul(xout, x, w []float32) {
	n := len(x)     // Input dim
	d := len(w) / n // Output dim
	for i := 0; i < d; i++ {
		var val float32
		for j := 0; j < n; j++ {
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
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	for i := 0; i < n; i++ {
		x[i] /= sum
	}
}

// Sample selects a token from logits (greedy if temp=0, else softmax sample).
func Sample(logits []float32, temp float64, rng *rand.Rand) int32 {
	if temp == 0 {
		// Greedy: argmax
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

	// Temperature scaling
	probs := make([]float32, len(logits))
	copy(probs, logits)
	for i := range probs {
		probs[i] /= float32(temp)
	}
	Softmax(probs)

	// Sample from distribution
	r := rng.Float32()
	var cdf float32
	for i, p := range probs {
		cdf += p
		if r < cdf {
			return int32(i)
		}
	}
	return int32(len(probs) - 1) // Fallback for rounding
}
