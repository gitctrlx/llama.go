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
// We define our color scheme once, globally.
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

	// Define flags using the standard library.
	tokenizer := flag.String("tokenizer", "./tokenizer.bin", "Path to tokenizer file")
	temperature := flag.Float64("temperature", 0.9, "Sampling temperature (0 for greedy)")
	topp := flag.Float64("topp", 0.9, "Top-p (nucleus) sampling threshold")
	steps := flag.Int("steps", 256, "Max generation steps")
	prompt := flag.String("prompt", "Once upon a time", "Input prompt")
	seed := flag.Int64("seed", time.Now().UnixNano(), "RNG seed")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <checkpoint>\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	// Expect exactly one positional argument: the checkpoint path.
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

	if err := execute(&cfg); err != nil {
		// Using our error color for fatal logs.
		log.Fatalf("%s %v", errorColor.Sprint("Error:"), err)
	}
}

// execute orchestrates the main program flow and UI.
func execute(cfg *AppConfig) error {
	// --- Header ---
	titleColor.Println("Llama.go")
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
	infoColor.Print("Initializing model...")

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
	infoColor.Println("\rModel loaded successfully.  ")
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
			infoColor.Printf("\n-----------------\nAchieved %.2f tokens/sec\n", tokPerSec)
		}
	} else {
		fmt.Println() // Ensure a newline at the end.
	}

	return nil
}
