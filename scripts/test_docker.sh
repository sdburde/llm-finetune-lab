#!/bin/bash
# Test All Methods with Docker
# Usage: ./scripts/test_docker.sh

set -e

echo "======================================================================"
echo "  LLM FINE-TUNING TOOLKIT - DOCKER TEST SUITE"
echo "======================================================================"
echo ""

# Configuration
IMAGE_NAME="llm-fine-tuning:test"
CONTAINER_NAME="llm-test-container"
TEST_SCRIPT="scripts/test_alpaca.py"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Step 1: Build Docker Image
print_header "STEP 1: Building Docker Image"
echo "Image name: $IMAGE_NAME"

docker build -t $IMAGE_NAME . 

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Step 2: Run Tests in Docker Container
print_header "STEP 2: Running Test Suite in Docker"

# Remove existing container if exists
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run tests
docker run --gpus all \
    --name $CONTAINER_NAME \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/test_results:/app/test_results \
    $IMAGE_NAME \
    python $TEST_SCRIPT

TEST_EXIT_CODE=$?

# Step 3: Extract Results
print_header "STEP 3: Extracting Test Results"

# Copy results from container
docker cp $CONTAINER_NAME:/app/alpaca_test_results.json ./test_results/ 2>/dev/null || true
docker cp $CONTAINER_NAME:/app/ALPACA_TEST_RESULTS.md ./test_results/ 2>/dev/null || true

# Cleanup container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Step 4: Display Results
print_header "STEP 4: TEST RESULTS"

if [ -f ./test_results/alpaca_test_results.json ]; then
    print_success "Test results saved to: ./test_results/alpaca_test_results.json"
    
    # Parse and display results
    echo ""
    echo "Summary:"
    cat ./test_results/alpaca_test_results.json | python -m json.tool | grep -A 5 '"summary"'
    
    echo ""
    echo "Detailed Results:"
    echo "----------------------------------------------------------------------"
    
    # Display method results
    python -c "
import json
with open('./test_results/alpaca_test_results.json') as f:
    data = json.load(f)
    for result in data['results']:
        method = result['method']
        status = result['status']
        time_m = result.get('time_minutes', 'N/A')
        vram = result.get('vram_gb', 'N/A')
        loss = result.get('training_loss', 'N/A')
        ppl = result.get('eval_perplexity', 'N/A')
        
        if isinstance(time_m, float):
            time_m = f'{time_m:.1f}m'
        if isinstance(vram, float):
            vram = f'{vram:.1f}GB'
        if isinstance(loss, float):
            loss = f'{loss:.2f}'
        if isinstance(ppl, float):
            ppl = f'{ppl:.2f}'
        
        icon = '✅' if status == 'PASS' else '❌'
        print(f'{icon} {method:<10} | Time: {time_m:>6} | VRAM: {vram:>6} | Loss: {loss:>8} | PPL: {ppl:>8}')
"
else
    print_error "Test results file not found"
fi

# Step 5: Final Status
print_header "FINAL STATUS"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All Docker tests passed!"
    print_info "Results saved to: ./test_results/"
else
    print_error "Some Docker tests failed"
    print_info "Check logs for details"
fi

echo ""
echo "======================================================================"
echo "  DOCKER TEST COMPLETE"
echo "======================================================================"
echo ""

exit $TEST_EXIT_CODE
