package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/fatih/color"
	"github.com/qntx/llama.go"
	"github.com/spf13/cobra"
)

// --- UI Color Scheme ---
var (
	titleColor    = color.New(color.FgCyan, color.Bold)
	promptColor   = color.New(color.FgYellow)
	generateColor = color.New(color.FgGreen)
	infoColor     = color.New(color.FgWhite, color.Faint)
	errorColor    = color.New(color.FgRed, color.Bold)
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

	rootCmd := &cobra.Command{
		Use:   "llamaq8 <quantized_checkpoint>",
		Short: "Run a Q8 quantized Llama model with an elegant CLI interface",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg.CheckpointPath = args[0]
			return executeQ8(&cfg)
		},
	}

	// Bind command-line flags to the AppConfig struct.
	rootCmd.Flags().StringVarP(&cfg.TokenizerPath, "tokenizer", "z", "./tokenizer.bin", "Path to tokenizer file")
	rootCmd.Flags().Float64VarP(&cfg.Temperature, "temperature", "t", 0.9, "Sampling temperature (0 for greedy)")
	rootCmd.Flags().Float64VarP(&cfg.Topp, "topp", "p", 0.9, "Top-p (nucleus) sampling threshold")
	rootCmd.Flags().Int32VarP(&cfg.Steps, "steps", "n", 256, "Max generation steps")
	rootCmd.Flags().StringVarP(&cfg.Prompt, "prompt", "i", "Once upon a time", "Input prompt")
	rootCmd.Flags().Int64VarP(&cfg.Seed, "seed", "s", time.Now().UnixNano(), "RNG seed")

	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("%s %v", errorColor.Sprint("Error:"), err)
	}
}

// executeQ8 orchestrates the main program flow for a quantized model.
func executeQ8(cfg *AppConfig) error {
	titleColor.Println("🚀 Llama.go (Q8 Quantized)")
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
	infoColor.Print("⏳ Initializing quantized model...")

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

	infoColor.Println("\r✅ Quantized model loaded successfully.   ")
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
			infoColor.Printf("\n-----------------\n🚀 Achieved %.2f tokens/sec\n", tokPerSec)
		}
	} else {
		fmt.Println()
	}

	return nil
}
