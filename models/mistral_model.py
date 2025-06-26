#!/usr/bin/env python3
"""
Mistral 7B Model Wrapper

This module provides a wrapper for the local Mistral 7B model
for patent analysis and prediction tasks.
Optimized for Apple Silicon M3 with GPU acceleration.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
import platform

logger = logging.getLogger(__name__)

class MistralModel:
    """Wrapper for Mistral 7B model for patent analysis with GPU acceleration."""
    
    def __init__(self, model_path: str = "mistral-pytorch-7b-instruct-v0.1-hf-v1"):
        """
        Initialize the Mistral model with GPU acceleration.
        
        Args:
            model_path: Path to the local Mistral model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = self._get_optimal_device()
        
        logger.info(f"Loading Mistral model from {model_path}")
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for the current hardware.
        Prioritizes Apple Silicon GPU (MPS) for Mac M3, then CUDA, then CPU.
        """
        # Check for Apple Silicon (MPS)
        if (platform.system() == "Darwin" and 
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()):
            logger.info("Apple Silicon detected - using MPS (Metal Performance Shaders)")
            return "mps"
        
        # Check for CUDA
        elif torch.cuda.is_available():
            logger.info("CUDA detected - using GPU acceleration")
            return "cuda"
        
        # Fallback to CPU
        else:
            logger.info("No GPU acceleration available - using CPU")
            return "cpu"
    
    def _load_model(self):
        """Load the Mistral model and tokenizer with GPU optimization."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine optimal dtype and loading strategy
            if self.device == "mps":
                # Apple Silicon optimization
                torch_dtype = torch.float16  # Use FP16 for memory efficiency
                device_map = None  # Load to CPU first, then move to MPS
                low_cpu_mem_usage = True
            elif self.device == "cuda":
                # CUDA optimization
                torch_dtype = torch.float16
                device_map = "auto"
                low_cpu_mem_usage = True
            else:
                # CPU optimization
                torch_dtype = torch.float32
                device_map = None
                low_cpu_mem_usage = False
            
            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
            
            # Move model to device if not using device_map
            if device_map is None:
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Mistral model loaded successfully on {self.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            
            # Log memory usage
            if self.device == "mps":
                # For MPS, we can't easily get memory info, but we can log the device
                logger.info("Model loaded on Apple Silicon GPU (MPS)")
            elif self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            # Fallback to CPU if GPU loading fails
            if self.device != "cpu":
                logger.info("Falling back to CPU due to GPU loading error")
                self.device = "cpu"
                self._load_model_cpu_fallback()
            else:
                raise
    
    def _load_model_cpu_fallback(self):
        """Fallback method to load model on CPU if GPU loading fails."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            self.model = self.model.to("cpu")
            self.model.eval()
            
            logger.info("Mistral model loaded successfully on CPU (fallback)")
            
        except Exception as e:
            logger.error(f"Failed to load model on CPU: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate a response from the model with GPU acceleration.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Tokenize input with proper truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,  # Increased to handle longer patent data
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response using max_new_tokens with GPU optimization
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache for faster generation
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def predict_patent_decision(self, patent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict patent decision (accepted/rejected) based on patent data.
        
        Args:
            patent_data: Dictionary containing patent information
            
        Returns:
            Dictionary with prediction and reasoning
        """
        # Build prompt for decision prediction
        prompt = self._build_decision_prompt(patent_data)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Parse response to extract prediction and reasoning
        prediction, reasoning = self._parse_decision_response(response)
        
        return {
            'prediction': prediction,
            'reasoning': reasoning,
            'confidence': 0.8,  # Placeholder confidence score
            'raw_response': response
        }
    
    def _build_decision_prompt(self, patent_data: Dict[str, Any]) -> str:
        """Build a prompt for patent decision prediction."""
        # Truncate long text fields to prevent token overflow
        title = patent_data.get('title', 'N/A')[:200]  # Limit title length
        abstract = patent_data.get('abstract', 'N/A')[:500]  # Limit abstract length
        description = patent_data.get('description', 'N/A')[:800]  # Limit description length
        
        # Handle claims - take first few claims if there are many
        claims = patent_data.get('claims', [])
        if isinstance(claims, list) and len(claims) > 3:
            claims_text = " ".join(claims[:3])  # Take first 3 claims
        elif isinstance(claims, str):
            claims_text = claims[:300]  # Limit string claims
        else:
            claims_text = "N/A"
        
        date_published = patent_data.get('date_published', 'N/A')
        
        prompt = f"""<s>[INST] Analyze this patent and predict: ACCEPTED or REJECTED.

Title: {title}
Abstract: {abstract}
Description: {description}
Claims: {claims_text}
Date: {date_published}

Provide your prediction (ACCEPTED/REJECTED) and brief reasoning based on the patent's technical merit, novelty, and potential impact. [/INST]"""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract prediction and reasoning."""
        response_lower = response.lower()
        
        # Extract prediction
        if 'accepted' in response_lower:
            prediction = 'accepted'
        elif 'rejected' in response_lower:
            prediction = 'rejected'
        else:
            prediction = 'unknown'
        
        # Use the entire response as reasoning
        reasoning = response.strip()
        
        return prediction, reasoning
    
    def train_on_patent_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fine-tune the model on patent data (placeholder for future implementation).
        
        Args:
            training_data: List of patent data with decisions
            
        Returns:
            Training results
        """
        logger.info(f"Training on {len(training_data)} patent samples")
        
        # This is a placeholder for actual fine-tuning
        # In a real implementation, you would:
        # 1. Prepare the data for fine-tuning
        # 2. Set up training parameters
        # 3. Run the fine-tuning process
        # 4. Save the fine-tuned model
        
        return {
            'status': 'training_completed',
            'samples_processed': len(training_data),
            'note': 'Fine-tuning not implemented yet - using base model'
        }
    
    def evaluate_predictions(self, true_labels: List[str], predicted_labels: List[str], 
                           predicted_probs: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Evaluate model predictions using various metrics.
        
        Args:
            true_labels: True decision labels
            predicted_labels: Predicted decision labels
            predicted_probs: Predicted probabilities (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert labels to binary (accepted=1, rejected=0)
        label_map = {'accepted': 1, 'rejected': 0}
        
        y_true = [label_map.get(label.lower(), 0) for label in true_labels]
        y_pred = [label_map.get(label.lower(), 0) for label in predicted_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['rejected', 'accepted'])
        
        # ROC-AUC if probabilities are provided
        roc_auc = None
        if predicted_probs:
            try:
                roc_auc = roc_auc_score(y_true, predicted_probs)
            except ValueError:
                logger.warning("Could not calculate ROC-AUC - insufficient data")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report,
            'total_samples': len(y_true),
            'accepted_count': sum(y_true),
            'rejected_count': len(y_true) - sum(y_true)
        } 