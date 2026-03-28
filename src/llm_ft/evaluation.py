"""
Evaluation & Benchmarking Suite for LLM Fine-Tuning

Provides metrics for evaluating fine-tuned models:
- Perplexity
- BLEU score
- ROUGE score
- Accuracy
- Response time
- VRAM usage
"""

import torch
import math
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import time


def evaluate_perplexity(
    model,
    tokenizer,
    test_data: List[str],
    batch_size: int = 1
) -> float:
    """
    Calculate perplexity on test data.
    
    Perplexity measures how well the model predicts the data.
    Lower is better!
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_data: List of text samples
        batch_size: Batch size for evaluation
        
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_data:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if not inputs["input_ids"].numel():
                continue
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Accumulate
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    
    # Calculate perplexity
    if total_tokens == 0:
        return float("inf")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def evaluate_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Calculate BLEU score for text generation.
    
    BLEU measures similarity between generated and reference text.
    Higher is better (0 to 1 scale).
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        max_n: Maximum n-gram order
        
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        # Fallback simple BLEU implementation
        return _simple_bleu(predictions, references, max_n)
    
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {f"BLEU-{i}": 0.0 for i in range(1, max_n + 1)}
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        
        # Calculate BLEU for each n-gram order
        for n in range(1, max_n + 1):
            weights = [0] * max_n
            weights[n - 1] = 1.0
            
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores[f"BLEU-{n}"] += score
    
    # Average over all samples
    n_samples = len(predictions)
    bleu_scores = {k: v / n_samples for k, v in bleu_scores.items()}
    
    return bleu_scores


def _simple_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """Simple BLEU implementation without nltk."""
    
    def count_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = count_ngrams(pred_tokens, n)
        ref_ngrams = count_ngrams(ref_tokens, n)
        
        matches = 0
        total = sum(pred_ngrams.values())
        
        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                matches += min(count, ref_ngrams[ngram])
        
        return matches / max(total, 1)
    
    bleu_scores = {}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        for n in range(1, max_n + 1):
            prec = precision(pred_tokens, ref_tokens, n)
            key = f"BLEU-{n}"
            bleu_scores[key] = bleu_scores.get(key, 0) + prec
    
    n_samples = len(predictions)
    bleu_scores = {k: v / n_samples for k, v in bleu_scores.items()}
    
    return bleu_scores


def evaluate_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for summarization.
    
    ROUGE measures overlap between generated and reference text.
    Higher is better (0 to 1 scale).
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_scores["ROUGE-1"] += scores["rouge1"].fmeasure
            rouge_scores["ROUGE-2"] += scores["rouge2"].fmeasure
            rouge_scores["ROUGE-L"] += scores["rougeL"].fmeasure
        
        n_samples = len(predictions)
        rouge_scores = {k: v / n_samples for k, v in rouge_scores.items()}
        
        return rouge_scores
        
    except ImportError:
        # Fallback simple implementation
        return _simple_rouge(predictions, references)


def _simple_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Simple ROUGE implementation without rouge_score."""
    
    def get_ngrams(tokens: List[str], n: int) -> set:
        return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    
    def f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    rouge_scores = {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        # ROUGE-1 (unigrams)
        pred_1 = set(pred_tokens)
        ref_1 = set(ref_tokens)
        overlap_1 = len(pred_1 & ref_1)
        prec_1 = overlap_1 / max(len(pred_1), 1)
        rec_1 = overlap_1 / max(len(ref_1), 1)
        rouge_scores["ROUGE-1"] += f1_score(prec_1, rec_1)
        
        # ROUGE-2 (bigrams)
        pred_2 = get_ngrams(pred_tokens, 2)
        ref_2 = get_ngrams(ref_tokens, 2)
        overlap_2 = len(pred_2 & ref_2)
        prec_2 = overlap_2 / max(len(pred_2), 1)
        rec_2 = overlap_2 / max(len(ref_2), 1)
        rouge_scores["ROUGE-2"] += f1_score(prec_2, rec_2)
        
        # ROUGE-L (longest common subsequence)
        def lcs_length(X, Y):
            m, n = len(X), len(Y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        prec_l = lcs / max(len(pred_tokens), 1)
        rec_l = lcs / max(len(ref_tokens), 1)
        rouge_scores["ROUGE-L"] += f1_score(prec_l, rec_l)
    
    n_samples = len(predictions)
    rouge_scores = {k: v / n_samples for k, v in rouge_scores.items()}
    
    return rouge_scores


def evaluate_accuracy(
    model,
    tokenizer,
    test_data: List[Dict[str, str]],
    task_type: str = "classification"
) -> float:
    """
    Calculate accuracy for classification tasks.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_data: List of dicts with 'input' and 'label' keys
        task_type: Type of task
        
    Returns:
        Accuracy (0 to 1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in test_data:
            input_text = example.get("input", "")
            label = example.get("label", "")
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if prediction matches label
            if label.lower() in prediction.lower():
                correct += 1
            total += 1
    
    return correct / max(total, 1)


def measure_inference_speed(
    model,
    tokenizer,
    prompt: str,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Measure inference speed and memory usage.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        prompt: Input prompt
        num_runs: Number of runs to average
        
    Returns:
        Dictionary with timing and memory metrics
    """
    model.eval()
    
    times = []
    memory_used = []
    
    # Warm up
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Measure
    for _ in range(num_runs):
        torch.cuda.empty_cache()
        
        # Measure time
        start = time.time()
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=100)
        
        end = time.time()
        times.append(end - start)
        
        # Measure memory
        if torch.cuda.is_available():
            memory_used.append(torch.cuda.memory_allocated(0) / 1e9)
    
    results = {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "tokens_per_second": 100 / (sum(times) / len(times)),
    }
    
    if memory_used:
        results["avg_memory_gb"] = sum(memory_used) / len(memory_used)
    
    return results


def generate_evaluation_report(
    model_name: str,
    metrics: Dict[str, Any],
    output_path: str = "./evaluation_report.json"
) -> str:
    """
    Generate evaluation report in JSON format.
    
    Args:
        model_name: Name of the evaluated model
        metrics: Dictionary of evaluation metrics
        output_path: Path to save report
        
    Returns:
        Path to saved report
    """
    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "summary": {
            "perplexity": metrics.get("perplexity", "N/A"),
            "bleu_score": metrics.get("BLEU-4", "N/A"),
            "rouge_score": metrics.get("ROUGE-L", "N/A"),
            "inference_speed": metrics.get("tokens_per_second", "N/A"),
        }
    }
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return output_path


def compare_models(
    models_data: List[Dict[str, Any]],
    output_path: str = "./model_comparison.json"
) -> str:
    """
    Compare multiple models and generate comparison report.
    
    Args:
        models_data: List of dicts with model name and metrics
        output_path: Path to save comparison report
        
    Returns:
        Path to saved report
    """
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models_data,
        "ranking": {
            "perplexity": sorted(models_data, key=lambda x: x.get("perplexity", float("inf"))),
            "bleu": sorted(models_data, key=lambda x: x.get("BLEU-4", 0), reverse=True),
            "speed": sorted(models_data, key=lambda x: x.get("tokens_per_second", 0), reverse=True),
        }
    }
    
    # Save comparison
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    return output_path


def evaluate_all(
    model,
    tokenizer,
    test_data: List[str],
    references: List[str] = None
) -> Dict[str, Any]:
    """
    Run complete evaluation suite on a model.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_data: Test samples
        references: Reference texts (for BLEU/ROUGE)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # Perplexity
    print("Evaluating perplexity...")
    results["perplexity"] = evaluate_perplexity(model, tokenizer, test_data)
    print(f"  Perplexity: {results['perplexity']:.2f}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    model.eval()
    with torch.no_grad():
        for text in test_data[:10]:  # Limit for speed
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
            )
            predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # BLEU
    if references:
        print("Evaluating BLEU...")
        bleu = evaluate_bleu(predictions, references[:len(predictions)])
        results.update(bleu)
        print(f"  BLEU-4: {results.get('BLEU-4', 0):.3f}")
    
    # ROUGE
    if references:
        print("Evaluating ROUGE...")
        rouge = evaluate_rouge(predictions, references[:len(predictions)])
        results.update(rouge)
        print(f"  ROUGE-L: {results.get('ROUGE-L', 0):.3f}")
    
    # Inference speed
    print("Measuring inference speed...")
    speed = measure_inference_speed(model, tokenizer, test_data[0])
    results.update(speed)
    print(f"  Tokens/sec: {results.get('tokens_per_second', 0):.1f}")
    
    return results
