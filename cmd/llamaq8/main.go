package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/gocnn/llama.go"
)

// --- UI Color Scheme ---
type noopColor struct{}

func newNoopColor() *noopColor { return &noopColor{} }

func (c *noopColor) Print(a ...interface{})                 { fmt.Print(a...) }
func (c *noopColor) Printf(format string, a ...interface{}) { fmt.Printf(format, a...) }
func (c *noopColor) Println(a ...interface{})               { fmt.Println(a...) }
func (c *noopColor) Sprint(a ...interface{}) string         { return fmt.Sprint(a...) }

var (
	titleColor    = newNoopColor()
	promptColor   = newNoopColor()
	generateColor = newNoopColor()
	infoColor     = newNoopColor()
	errorColor    = newNoopColor()
)

// AppConfig holds all configuration parameters from the command line.
type AppConfig struct {
	CheckpointPath string
	TokenizerPath  string
	Temperature    float64
	Topp           float64
	Steps          int32
	Prompt         string
	Seed           int64
}

// QuantizedRunner encapsulates all components for a QUANTIZED model run.
type QuantizedRunner struct {
	config      *llama.QuantizedConfig
	weights     *llama.QuantizedTransformerWeights
	state       *llama.QuantizedRunState
	rng         *rand.Rand
	vocab       []string
	vocabMap    map[string]int32
	vocabScores []float32
	// SYNC: Add groupSize to the runner to pass it around easily.
	groupSize int
}

func main() {
	var cfg AppConfig

	// Define flags using the standard library.
	tokenizer := flag.String("tokenizer", "./tokenizer.bin", "Path to tokenizer file")
	temperature := flag.Float64("temperature", 0.9, "Sampling temperature (0 for greedy)")
	topp := flag.Float64("topp", 0.9, "Top-p (nucleus) sampling threshold")
	steps := flag.Int("steps", 256, "Max generation steps")
	prompt := flag.String("prompt", "Once upon a time", "Input prompt")
	seed := flag.Int64("seed", time.Now().UnixNano(), "RNG seed")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <quantized_checkpoint>\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) != 1 {
		flag.Usage()
		os.Exit(2)
	}
	cfg.CheckpointPath = args[0]
	cfg.TokenizerPath = *tokenizer
	cfg.Temperature = *temperature
	cfg.Topp = *topp
	cfg.Steps = int32(*steps)
	cfg.Prompt = *prompt
	cfg.Seed = *seed

	if err := executeQ8(&cfg); err != nil {
		log.Fatalf("%s %v", errorColor.Sprint("Error:"), err)
	}
}

// executeQ8 orchestrates the main program flow for a quantized model.
func executeQ8(cfg *AppConfig) error {
	titleColor.Println("Llama.go (Q8 Quantized)")
	infoColor.Println("-----------------")

	runner, err := setupQ8(cfg)
	if err != nil {
		return err
	}

	infoColor.Println("Configuration:")
	infoColor.Printf("  - Model:        %s\n", filepath.Base(cfg.CheckpointPath))
	// SYNC: Display Group Size from the runner struct, not from the config.
	infoColor.Printf("  - Group Size:   %d\n", runner.groupSize)
	infoColor.Printf("  - Steps:        %d\n", cfg.Steps)
	infoColor.Printf("  - Temp:         %.2f\n", cfg.Temperature)
	infoColor.Printf("  - Top-p:        %.2f\n", cfg.Topp)
	infoColor.Printf("  - Seed:         %d\n", cfg.Seed)
	infoColor.Println("-----------------")

	return runQ8(runner, cfg)
}

// setupQ8 handles initialization for a QUANTIZED model.
func setupQ8(cfg *AppConfig) (*QuantizedRunner, error) {
	infoColor.Print("Initializing quantized model...")

	// SYNC: Update the function call to receive the new `groupSize` return value.
	config, weights, gs, err := llama.LoadQuantizedCheckpoint(cfg.CheckpointPath)
	if err != nil {
		return nil, fmt.Errorf("could not load quantized checkpoint: %w", err)
	}

	// Tokenizer loading remains the same
	vocab, vocabScores, _, err := llama.LoadTokenizer(cfg.TokenizerPath, int(config.VocabSize))
	if err != nil {
		return nil, fmt.Errorf("could not load tokenizer: %w", err)
	}

	vocabMap := make(map[string]int32, len(vocab))
	for i, v := range vocab {
		vocabMap[v] = int32(i)
	}

	runner := &QuantizedRunner{
		config:  config,
		weights: weights,
		// SYNC: Pass `gs` to the state allocator function.
		state:       llama.NewQuantizedRunState(config, gs),
		rng:         rand.New(rand.NewSource(cfg.Seed)),
		vocab:       vocab,
		vocabMap:    vocabMap,
		vocabScores: vocabScores,
		// SYNC: Store `gs` in our runner struct.
		groupSize: gs,
	}

	infoColor.Println("\rQuantized model loaded successfully.   ")
	return runner, nil
}

// runQ8 contains the core text generation loop for a QUANTIZED model.
func runQ8(r *QuantizedRunner, cfg *AppConfig) error {
	steps := cfg.Steps
	if steps <= 0 || steps > r.config.SeqLen {
		steps = r.config.SeqLen
	}

	promptTokens, err := llama.BPEEncode(cfg.Prompt, r.vocab, r.vocabScores, r.vocabMap, true, false)
	if err != nil {
		return fmt.Errorf("could not encode prompt: %w", err)
	}
	if len(promptTokens) < 1 {
		return fmt.Errorf("something is wrong, expected at least 1 prompt token")
	}

	promptColor.Print("▶ Prompt: ")
	fmt.Println(cfg.Prompt)
	generateColor.Print("▶ Generation: ")

	var start time.Time
	token := promptTokens[0]
	pos := int32(0)

	for pos < steps {
		// SYNC: Pass `r.groupSize` to the transformer forward pass function.
		llama.TransformerQ8(token, pos, r.config, r.state, r.weights, r.groupSize)

		var next int32
		if pos < int32(len(promptTokens)-1) {
			next = promptTokens[pos+1]
		} else {
			if start.IsZero() {
				start = time.Now()
			}
			// Sampling logic is identical, as it works on float32 logits
			next = llama.Sample(r.state.Logits, cfg.Temperature, cfg.Topp, r.rng)

			if next == 1 { // End of sequence token
				break
			}

			tokenStr := r.vocab[next]
			if token == 1 && len(tokenStr) > 0 && tokenStr[0] == ' ' {
				tokenStr = tokenStr[1:]
			}
			generateColor.Print(tokenStr)
			os.Stdout.Sync()
		}

		token = next
		pos++
	}

	generatedSteps := pos - int32(len(promptTokens))
	if generatedSteps > 1 {
		elapsed := time.Since(start)
		if elapsed.Seconds() > 0 {
			tokPerSec := float64(generatedSteps) / elapsed.Seconds()
			infoColor.Printf("\n-----------------\nAchieved %.2f tokens/sec\n", tokPerSec)
		}
	} else {
		fmt.Println()
	}

	return nil
}
