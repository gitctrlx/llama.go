package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/qntx/llama.go"
	"github.com/spf13/cobra"
)

// AppConfig holds all configuration parameters from the command line.
type AppConfig struct {
	CheckpointPath string
	TokenizerPath  string
	Temperature    float64
	Topp           float64 // Add this line
	Steps          int32
	Prompt         string
	Seed           int64
}

// Runner encapsulates all the components and state required to run the model.
// It bundles the model's parts together for easier management.
type Runner struct {
	config         *llama.Config
	weights        *llama.TransformerWeights
	state          *llama.RunState
	rng            *rand.Rand
	vocab          []string
	vocabMap       map[string]int32
	vocabScores    []float32
	maxTokenLength uint32
}

func main() {
	var cfg AppConfig

	rootCmd := &cobra.Command{
		Use:   "llama <checkpoint>",
		Short: "Run a llama model",
		Args:  cobra.ExactArgs(1), // Use checkpoint as a positional argument.
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg.CheckpointPath = args[0]
			return execute(&cfg)
		},
	}

	// Bind command-line flags to the AppConfig struct.
	rootCmd.Flags().StringVar(&cfg.TokenizerPath, "tokenizer", "./tokenizer.bin", "Path to tokenizer file")
	rootCmd.Flags().Float64Var(&cfg.Temperature, "temperature", 0.9, "Sampling temperature (0 for greedy)")
	rootCmd.Flags().Float64Var(&cfg.Topp, "topp", 0.9, "Top-p (nucleus) sampling threshold")
	rootCmd.Flags().Int32Var(&cfg.Steps, "steps", 256, "Max generation steps (0 uses seq_len)")
	rootCmd.Flags().StringVar(&cfg.Prompt, "prompt", "Once upon a time", "Input prompt")
	rootCmd.Flags().Int64Var(&cfg.Seed, "seed", time.Now().UnixNano(), "RNG seed")

	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("❌ Error: %v", err)
	}
}

// execute orchestrates the main program flow.
func execute(cfg *AppConfig) error {
	// 1. Set up and load the model.
	runner, err := setup(cfg)
	if err != nil {
		return err
	}

	// 2. Run the text generation.
	return run(runner, cfg)
}

// setup handles all initialization and setup tasks.
func setup(cfg *AppConfig) (*Runner, error) {
	fmt.Println("⏳ Initializing model...")

	// Load model config and weights.
	config, weights, err := llama.LoadCheckpoint(cfg.CheckpointPath)
	if err != nil {
		return nil, fmt.Errorf("could not load checkpoint: %w", err)
	}

	// Load the tokenizer.
	vocab, vocabScores, maxTokenLength, err := llama.LoadTokenizer(cfg.TokenizerPath, int(config.VocabSize))
	if err != nil {
		return nil, fmt.Errorf("could not load tokenizer: %w", err)
	}
	vocabMap := make(map[string]int32, len(vocab))
	for i, v := range vocab {
		vocabMap[v] = int32(i)
	}

	// Create a fully initialized Runner.
	runner := &Runner{
		config:         config,
		weights:        weights,
		state:          llama.NewRunState(config),
		rng:            rand.New(rand.NewSource(cfg.Seed)),
		vocab:          vocab,
		vocabMap:       vocabMap,
		vocabScores:    vocabScores,
		maxTokenLength: maxTokenLength,
	}

	fmt.Println("✅ Model loaded successfully.")
	return runner, nil
}

// run contains the core text generation loop.
func run(r *Runner, cfg *AppConfig) error {
	// Clamp the number of steps to the model's max sequence length.
	steps := cfg.Steps
	if steps <= 0 || steps > r.config.SeqLen {
		steps = r.config.SeqLen
	}

	// 1. Encode the prompt using the new, aligned BPEEncode function.
	// We want to prepend the BOS token (true) but not append an EOS token (false).
	promptTokens, err := llama.BPEEncode(cfg.Prompt, r.vocab, r.vocabScores, r.vocabMap, true, false)
	if err != nil {
		return fmt.Errorf("could not encode prompt: %w", err)
	}
	if len(promptTokens) < 1 {
		return fmt.Errorf("something is wrong, expected at least 1 prompt token")
	}

	fmt.Println("--- Generation Start ---")

	// 2. Start the generation loop.
	var start time.Time
	// The first token is the one from our encoded prompt (which is BOS).
	token := promptTokens[0]
	pos := int32(0)

	for pos < steps {
		// Run one forward pass of the transformer.
		llama.Transformer(token, pos, r.config, r.state, r.weights)

		var next int32
		if pos < int32(len(promptTokens)-1) {
			// If we are still processing the prompt, force the next token to be the next prompt token.
			// This is called "prompt processing".
			next = promptTokens[pos+1]
		} else {
			// Otherwise, we are in "generation" mode: sample from the model's output distribution.
			next = llama.Sample(r.state.Logits, cfg.Temperature, cfg.Topp, r.rng)
		}

		// Print the token, handling special cases.
		// NOTE: A proper `decode` function like in C would be better here.
		tokenStr := r.vocab[next]
		if token == 1 && len(tokenStr) > 0 && tokenStr[0] == ' ' {
			// The C code's `decode` function strips the leading space from the first token if it's a BOS.
			tokenStr = tokenStr[1:]
		}
		fmt.Print(tokenStr)
		os.Stdout.Sync() // Flush stdout for a streaming effect.

		// Advance to the next token.
		token = next
		pos++

		// Start the timer after the prompt is fully processed and generation begins.
		if start.IsZero() && pos >= int32(len(promptTokens)) {
			start = time.Now()
		}
	}

	fmt.Println("\n--- Generation End ---")

	// Report performance.
	// We calculate tokens/sec only for the generated part, not the prompt processing part.
	generatedSteps := pos - int32(len(promptTokens))
	if generatedSteps > 1 {
		elapsed := time.Since(start)
		if elapsed.Seconds() > 0 {
			tokPerSec := float64(generatedSteps) / elapsed.Seconds()
			fmt.Printf("🚀 Achieved tokens/sec: %.2f\n", tokPerSec)
		}
	}
	return nil
}
