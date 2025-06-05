#!/bin/bash

# Default values
INPUT_FILE="/data/breast_cancer.csv"
TARGET_COL="target"
CONFIG_FILE="pipeline_config.yaml"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
POP_SIZE=10
GENERATIONS=50
CV=5
CROSSOVER_RATE=0.6
MUTATION_RATE=0.05
ELITE_SIZE=2


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_COL="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--pop-size)
            POP_SIZE="$2"
            shift 2
            ;;
        -g|--generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --cv)
            CV="$2"
            shift 2
            ;;
        --crossover)
            CROSSOVER_RATE="$2"
            shift 2
            ;;
        --mutation)
            MUTATION_RATE="$2"
            shift 2
            ;;
        --elite-size)
            ELITE_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$TARGET_COL" ]; then
    echo "Error: Input file and target column are required"
    show_help
fi

# Run the genetic feature selector
echo "----------------------------------------"
echo "Running genetic feature selection..."
echo "Input file: $INPUT_FILE"
echo "Target column: $TARGET_COL"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Population size: $POP_SIZE"
echo "Generations: $GENERATIONS"
echo "CV folds: $CV"
echo "Crossover rate: $CROSSOVER_RATE"
echo "Mutation rate: $MUTATION_RATE"
echo "Elite size: $ELITE_SIZE"
echo "----------------------------------------"

genetic-feature-selector \
    --input "$INPUT_FILE" \
    --target "$TARGET_COL" \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --pop-size "$POP_SIZE" \
    --generations "$GENERATIONS" \
    --cv "$CV" \
    --crossover-rate "$CROSSOVER_RATE" \
    --mutation-rate "$MUTATION_RATE" \
    --elite-size "$ELITE_SIZE"


echo "----------------------------------------"

echo "Done!"