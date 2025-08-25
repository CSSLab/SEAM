#!/bin/bash

set -e  # Exit on any error

cd "$(dirname "$0")"

VLLM_MODELS=(
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-Omni-7B"
    "OpenGVLab/InternVL3-78B"
    "OpenGVLab/InternVL3-8B"
    "OpenGVLab/InternVL2_5-78B"
    "OpenGVLab/InternVL2_5-8B"
    "meta-llama/Llama-3.2-90B-Vision-Instruct"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "google/gemma-3-27b-it"
    "google/gemma-3-12b-it"
)

OPENAI_MODELS=(
    "gpt-5"
    "gpt-5-mini"
    "gpt-5-nano"
    "gpt-4o"
    "gpt-4o-mini"
)

CLAUDE_MODELS=(
    "claude-opus-4-1-20250805"
    "claude-opus-4-20250514"
    "claude-sonnet-4-20250514"
    "claude-3-7-sonnet-20250219"
    "claude-3-5-haiku-20241022"
)

# Modes to run (language-only, vision-only, vision-language)
MODES="l,v,vl"

# Log directory
LOG_DIR="/datadrive/josephtang/SEAM/logs"
mkdir -p "$LOG_DIR"

# Calculate total models
TOTAL_MODELS=$((${#VLLM_MODELS[@]} + ${#OPENAI_MODELS[@]} + ${#CLAUDE_MODELS[@]}))

echo "🚀 Starting SEAM Benchmark Complete Pipeline for All Models"
echo "📁 Logs will be saved to: $LOG_DIR"
echo "🎯 Modes: $MODES"
echo "📊 Total models: $TOTAL_MODELS"
echo "   - vLLM models: ${#VLLM_MODELS[@]}"
echo "   - OpenAI models: ${#OPENAI_MODELS[@]}"
echo "   - Claude models: ${#CLAUDE_MODELS[@]}"
echo "🔄 Pipeline: Inference (01) -> Extraction (02) -> Metrics (03)"
echo ""

# Function to normalize model names (consistent with Python scripts)
normalize_model_name() {
    local model="$1"
    echo "$model" | sed 's/\//-/g' | sed 's/_/-/g' | tr '[:upper:]' '[:lower:]'
}

# Function to determine model type and inference script
get_inference_script() {
    local model="$1"
    
    # Check if it's an OpenAI model
    for openai_model in "${OPENAI_MODELS[@]}"; do
        if [[ "$model" == "$openai_model" ]]; then
            echo "01_inference_openai.py"
            return
        fi
    done
    
    # Check if it's a Claude model
    for claude_model in "${CLAUDE_MODELS[@]}"; do
        if [[ "$model" == "$claude_model" ]]; then
            echo "01_inference_claude.py"
            return
        fi
    done
    
    # Default to vLLM for local models
    echo "01_inference_vllm.py"
}

# Function to run complete pipeline for a single model
run_model_pipeline() {
    local model="$1"
    local model_safe_name=$(normalize_model_name "$model")
    local inference_script=$(get_inference_script "$model")
    local inference_log="$LOG_DIR/01_inference_${model_safe_name}.log"
    local extraction_log="$LOG_DIR/02_extract_${model_safe_name}.log"
    local metrics_log="$LOG_DIR/03_metrics_${model_safe_name}.log"
    
    echo "🔄 Starting complete pipeline for: $model"
    echo "📁 Normalized model name: $model_safe_name"
    echo "🛠️  Using inference script: $inference_script"
    echo "📝 Stage 1 (Inference) log: $inference_log"
    echo "📝 Stage 2 (Extraction) log: $extraction_log"
    echo "📝 Stage 3 (Metrics) log: $metrics_log"
    
    # Stage 1: Inference
    echo "🚀 Stage 1: Running inference..."
    if ! nohup python "$inference_script" \
        --model "$model" \
        --modes "$MODES" \
        > "$inference_log" 2>&1; then
        echo "❌ Stage 1 (Inference) failed for model: $model"
        echo "   Check log: $inference_log"
        return 1
    fi
    echo "✅ Stage 1 (Inference) completed for: $model"
    
    # Stage 2: Extraction
    echo "🔍 Stage 2: Running answer extraction..."
    if ! nohup python 02_extract.py \
        --model "$model_safe_name" \
        > "$extraction_log" 2>&1; then
        echo "❌ Stage 2 (Extraction) failed for model: $model"
        echo "   Check log: $extraction_log"
        return 1
    fi
    echo "✅ Stage 2 (Extraction) completed for: $model"
    
    # Stage 3: Metrics
    echo "📊 Stage 3: Computing metrics..."
    if ! nohup python 03_metric.py \
        --model "$model_safe_name" \
        > "$metrics_log" 2>&1; then
        echo "❌ Stage 3 (Metrics) failed for model: $model"
        echo "   Check log: $metrics_log"
        return 1
    fi
    echo "✅ Stage 3 (Metrics) completed for: $model"
    
    echo "🎉 Complete pipeline finished successfully for: $model"
    return 0
}

# Main execution
failed_models=()
successful_models=()

# Process vLLM models
if [ ${#VLLM_MODELS[@]} -gt 0 ]; then
    echo "🔄 Processing vLLM models..."
    for model in "${VLLM_MODELS[@]}"; do
        echo "════════════════════════════════════════════════════════════════"
        
        if run_model_pipeline "$model"; then
            successful_models+=("$model")
        else
            failed_models+=("$model")
        fi
        
        # Add a brief pause between model pipelines
        echo "⏸️  Waiting 30 seconds before next model..."
        sleep 30
    done
else
    echo "⏭️  No vLLM models configured"
fi

# Process OpenAI models
if [ ${#OPENAI_MODELS[@]} -gt 0 ]; then
    echo "🔄 Processing OpenAI models..."
    for model in "${OPENAI_MODELS[@]}"; do
        echo "════════════════════════════════════════════════════════════════"
        
        if run_model_pipeline "$model"; then
            successful_models+=("$model")
        else
            failed_models+=("$model")
        fi
        
        # Add a brief pause between model pipelines
        echo "⏸️  Waiting 30 seconds before next model..."
        sleep 30
    done
else
    echo "⏭️  No OpenAI models configured"
fi

# Process Claude models
if [ ${#CLAUDE_MODELS[@]} -gt 0 ]; then
    echo "🔄 Processing Claude models..."
    for model in "${CLAUDE_MODELS[@]}"; do
        echo "════════════════════════════════════════════════════════════════"
        
        if run_model_pipeline "$model"; then
            successful_models+=("$model")
        else
            failed_models+=("$model")
        fi
        
        # Add a brief pause between model pipelines
        echo "⏸️  Waiting 30 seconds before next model..."
        sleep 30
    done
else
    echo "⏭️  No Claude models configured"
fi

echo "════════════════════════════════════════════════════════════════"
echo "📊 FINAL SUMMARY"
echo "✅ Successfully completed pipeline: ${#successful_models[@]} models"
echo "❌ Failed pipeline: ${#failed_models[@]} models"
echo ""

if [ ${#successful_models[@]} -gt 0 ]; then
    echo "✅ Successfully completed models:"
    for model in "${successful_models[@]}"; do
        echo "   - $model"
    done
    echo ""
fi

if [ ${#failed_models[@]} -gt 0 ]; then
    echo "❌ Failed models:"
    for model in "${failed_models[@]}"; do
        echo "   - $model"
    done
    echo ""
fi

echo "📁 All logs saved in: $LOG_DIR"
echo "   Stage 1 logs: $LOG_DIR/01_inference_*.log"
echo "   Stage 2 logs: $LOG_DIR/02_extract_*.log"
echo "   Stage 3 logs: $LOG_DIR/03_metrics_*.log"
echo ""

if [ ${#successful_models[@]} -gt 1 ]; then
    echo "🎯 Generate final comparison:"
    # Generate normalized model names for comparison
    local normalized_models=""
    for model in "${successful_models[@]}"; do
        if [ -z "$normalized_models" ]; then
            normalized_models=$(normalize_model_name "$model")
        else
            normalized_models="$normalized_models,$(normalize_model_name "$model")"
        fi
    done
    echo "   python 03_metric.py --compare --models $normalized_models"
    echo ""
fi

echo "📊 Results are available in:"
echo "   /datadrive/josephtang/SEAM/results/"
echo ""
echo "🎉 Complete SEAM Benchmark Pipeline Finished!"
echo "   Total runtime: $(date)"