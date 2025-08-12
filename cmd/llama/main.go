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
// We define our color scheme once, globally.
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

// Runner encapsulates all the components required to run the model.
type Runner struct {
	config      *llama.Config
	weights     *llama.TransformerWeights
	state       *llama.RunState
	rng         *rand.Rand
	vocab       []string
	vocabMap    map[string]int32
	vocabScores []float32
}

func main() {
	var cfg AppConfig

	rootCmd := &cobra.Command{
		Use:   "llama <checkpoint>",
		Short: "Run a llama model with an elegant CLI interface",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg.CheckpointPath = args[0]
			return execute(&cfg)
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
		// Using our error color for fatal logs.
		log.Fatalf("%s %v", errorColor.Sprint("Error:"), err)
	}
}

// execute orchestrates the main program flow and UI.
func execute(cfg *AppConfig) error {
	// --- Header ---
	titleColor.Println("🚀 Llama.go")
	infoColor.Println("-----------------")

	// 1. Set up and load the model.
	runner, err := setup(cfg)
	if err != nil {
		return err
	}

	// --- Configuration Summary ---
	infoColor.Println("Configuration:")
	infoColor.Printf("  - Model:   %s\n", filepath.Base(cfg.CheckpointPath))
	infoColor.Printf("  - Steps:   %d\n", cfg.Steps)
	infoColor.Printf("  - Temp:    %.2f\n", cfg.Temperature)
	infoColor.Printf("  - Top-p:   %.2f\n", cfg.Topp)
	infoColor.Printf("  - Seed:    %d\n", cfg.Seed)
	infoColor.Println("-----------------")

	// 2. Run the text generation.
	return run(runner, cfg)
}

// setup handles all initialization and setup tasks, now with styled output.
func setup(cfg *AppConfig) (*Runner, error) {
	infoColor.Print("⏳ Initializing model...")

	config, weights, err := llama.LoadCheckpoint(cfg.CheckpointPath)
	if err != nil {
		return nil, fmt.Errorf("could not load checkpoint: %w", err)
	}

	vocab, vocabScores, _, err := llama.LoadTokenizer(cfg.TokenizerPath, int(config.VocabSize))
	if err != nil {
		return nil, fmt.Errorf("could not load tokenizer: %w", err)
	}

	vocabMap := make(map[string]int32, len(vocab))
	for i, v := range vocab {
		vocabMap[v] = int32(i)
	}

	runner := &Runner{
		config:      config,
		weights:     weights,
		state:       llama.NewRunState(config),
		rng:         rand.New(rand.NewSource(cfg.Seed)),
		vocab:       vocab,
		vocabMap:    vocabMap,
		vocabScores: vocabScores,
	}

	// Use \r to move the cursor to the beginning of the line and overwrite the loading message.
	infoColor.Println("\r✅ Model loaded successfully.  ")
	return runner, nil
}

// run contains the core text generation loop.
func run(r *Runner, cfg *AppConfig) error {
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

	// --- Generation ---
	// Print the user's prompt first for context.
	promptColor.Print("▶ Prompt: ")
	fmt.Println(cfg.Prompt)

	// Start the generation output on a new line.
	generateColor.Print("▶ Generation: ")

	var start time.Time
	token := promptTokens[0]
	pos := int32(0)

	for pos < steps {
		llama.Transformer(token, pos, r.config, r.state, r.weights)

		var next int32
		if pos < int32(len(promptTokens)-1) {
			// Processing the prompt, don't print anything yet.
			next = promptTokens[pos+1]
		} else {
			// Start the timer only when actual generation begins.
			if start.IsZero() {
				start = time.Now()
			}
			// Sample the next token.
			next = llama.Sample(r.state.Logits, cfg.Temperature, cfg.Topp, r.rng)

			// When the model generates the BOS token (ID 1), it signals the end of the sequence.
			// We should stop here to prevent it from starting a new, unrelated sequence.
			if next == 1 {
				break
			}

			// Print the generated token.
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

	// --- Footer ---
	generatedSteps := pos - int32(len(promptTokens))
	if generatedSteps > 1 {
		elapsed := time.Since(start)
		if elapsed.Seconds() > 0 {
			tokPerSec := float64(generatedSteps) / elapsed.Seconds()
			infoColor.Printf("\n-----------------\n🚀 Achieved %.2f tokens/sec\n", tokPerSec)
		}
	} else {
		fmt.Println() // Ensure a newline at the end.
	}

	return nil
}
