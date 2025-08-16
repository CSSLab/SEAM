#!/bin/bash
# SEAM Benchmark Setup Script

echo "Setting up SEAM Benchmark environment..."

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup API keys configuration
echo ""
echo "Setting up API keys configuration..."
if [ ! -f "api_keys.json" ]; then
    echo "Creating api_keys.json from template..."
    cp api_keys.json.template api_keys.json
    echo "✅ Created api_keys.json"
    echo ""
    echo "⚠️  IMPORTANT: You need to edit api_keys.json and add your API keys!"
    echo "   - Add your OpenAI API key for OpenAI batch processing"
    echo "   - Add your Anthropic API key for Claude batch processing"
    echo "   - The api_keys.json file is excluded from git for security"
    echo ""
else
    echo "✅ api_keys.json already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "🔑 Next steps for API configuration:"
echo "   1. Edit api_keys.json and add your API keys"
echo "   2. Your API keys will be kept secure and not committed to git"
echo ""
echo "Quick Start Examples:"
echo ""
echo "1. vLLM batch inference:"
echo "   cd ../run && python run_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l"
echo ""
echo "2. OpenAI batch:"
echo "   cd ../run && python run_batch.py --provider openai --action all --model gpt-4o-mini"
echo ""
echo "3. Claude batch:"
echo "   cd ../run && python run_batch.py --provider claude --model claude-3-5-sonnet-20241022"
echo ""
echo "For more examples, see README.md"
echo ""