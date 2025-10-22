package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/gitctrlx/llama.go"
)

// AppConfig holds command-line parameters.
type AppConfig struct {
	CheckpointPath string
	TokenizerPath  string
	Temperature    float64
	Topp           float64
	Steps          int32
	Prompt         string
	Seed           int64
}

// QuantizedRunner encapsulates quantized model components.
type QuantizedRunner struct {
	config      *llama.LlamaConfig
	weights     *llama.LlamaQuantizedWeights
	state       *llama.LlamaQuantizedState
	rng         *rand.Rand
	vocab       []string
	vocabMap    map[string]int32
	vocabScores []float32
	groupSize   int
}

func main() {
	var (
		tokenizer   = flag.String("tokenizer", "./tokenizer.bin", "tokenizer path")
		temperature = flag.Float64("temperature", 0.9, "temperature (0 for greedy)")
		topp        = flag.Float64("topp", 0.9, "top-p threshold")
		steps       = flag.Int("steps", 256, "max steps")
		prompt      = flag.String("prompt", "Once upon a time", "input prompt")
		seed        = flag.Int64("seed", time.Now().UnixNano(), "RNG seed")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <quantized_checkpoint>\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	cfg := AppConfig{
		CheckpointPath: flag.Arg(0),
		TokenizerPath:  *tokenizer,
		Temperature:    *temperature,
		Topp:           *topp,
		Steps:          int32(*steps),
		Prompt:         *prompt,
		Seed:           *seed,
	}

	r, err := setup(&cfg)
	if err != nil {
		log.Fatalf("setup: %v", err)
	}

	if err := generate(r, &cfg); err != nil {
		log.Fatalf("generate: %v", err)
	}
}

// setup loads quantized model and tokenizer.
func setup(cfg *AppConfig) (*QuantizedRunner, error) {
	fmt.Print("Loading quantized model... ")

	config, weights, gs, err := llama.LoadLlamaQuantizedModel(cfg.CheckpointPath)
	if err != nil {
		return nil, err
	}

	vocab, vocabScores, _, err := llama.LoadTokenizer(cfg.TokenizerPath, int(config.VocabSize))
	if err != nil {
		return nil, err
	}

	vocabMap := make(map[string]int32, len(vocab))
	for i, v := range vocab {
		vocabMap[v] = int32(i)
	}

	fmt.Println("done")

	return &QuantizedRunner{
		config:      config,
		weights:     weights,
		state:       llama.NewLlamaQuantizedState(config, gs),
		rng:         rand.New(rand.NewSource(cfg.Seed)),
		vocab:       vocab,
		vocabMap:    vocabMap,
		vocabScores: vocabScores,
		groupSize:   gs,
	}, nil
}

// generate runs quantized inference.
func generate(r *QuantizedRunner, cfg *AppConfig) error {
	steps := cfg.Steps
	if steps <= 0 || steps > r.config.SeqLen {
		steps = r.config.SeqLen
	}

	tokens, err := llama.BPEEncode(cfg.Prompt, r.vocab, r.vocabScores, r.vocabMap, true, false)
	if err != nil || len(tokens) == 0 {
		return fmt.Errorf("invalid prompt")
	}

	fmt.Printf("Prompt: %s\nGeneration: ", cfg.Prompt)

	var start time.Time
	token := tokens[0]
	pos := int32(0)

	for pos < steps {
		llama.LlamaForwardQuantized(token, pos, r.config, r.state, r.weights, r.groupSize)

		var next int32
		if pos < int32(len(tokens))-1 {
			next = tokens[pos+1]
		} else {
			if start.IsZero() {
				start = time.Now()
			}
			next = llama.Sample(r.state.Logits, cfg.Temperature, cfg.Topp, r.rng)
			if next == 1 {
				break
			}
			tokenStr := r.vocab[next]
			if token == 1 && len(tokenStr) > 0 && tokenStr[0] == ' ' {
				tokenStr = tokenStr[1:]
			}
			fmt.Print(tokenStr)
		}

		token = next
		pos++
	}

	genSteps := pos - int32(len(tokens))
	if genSteps > 1 {
		elapsed := time.Since(start).Seconds()
		if elapsed > 0 {
			fmt.Printf("\n%.2f tokens/sec\n", float64(genSteps)/elapsed)
		}
	} else {
		fmt.Println()
	}

	return nil
}
