#!/usr/bin/env python3
"""
Patent Analysis System

This system integrates NetworkX graph conversion with Mistral 7B model
for patent decision prediction and evaluation.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import networkx as nx
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from networkx_patent_converter import NetworkXPatentConverter
from models.mistral_model import MistralModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration
DEFAULT_TRAIN_RATIO = 0.9995
DEFAULT_RANDOM_SEED = 42

class PatentAnalysisSystem:
    """Main system for patent analysis using NetworkX graphs and Mistral 7B."""
    
    def __init__(self, model_path: str = "mistral-pytorch-7b-instruct-v0.1-hf-v1"):
        """
        Initialize the patent analysis system.
        
        Args:
            model_path: Path to the local Mistral model
        """
        self.model_path = model_path
        self.converter = NetworkXPatentConverter()
        self.model = None
        self.graph = None
        
        logger.info("Patent Analysis System initialized")
    
    def convert_json_to_graph(self, json_directory: str, output_path: str = "patent_graph.pkl") -> Dict[str, Any]:
        """
        Convert JSON patent files to NetworkX graph.
        
        Args:
            json_directory: Directory containing JSON patent files
            output_path: Path to save the graph
            
        Returns:
            Statistics about the conversion
        """
        logger.info(f"Converting JSON files from {json_directory} to graph")
        
        # Process the directory
        processed = self.converter.process_directory(json_directory)
        
        # Save the graph
        self.converter.save_graph(output_path)
        
        # Get statistics
        stats = self.converter.get_graph_stats()
        
        logger.info(f"Graph conversion completed: {processed} patents processed")
        return stats
    
    def load_graph(self, graph_path: str):
        """Load a previously created graph."""
        logger.info(f"Loading graph from {graph_path}")
        self.converter.load_graph(graph_path)
        self.graph = self.converter.graph
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def prepare_training_data(self, train_ratio: float = DEFAULT_TRAIN_RATIO, random_seed: int = DEFAULT_RANDOM_SEED) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and testing data from the graph.
        
        Args:
            train_ratio: Ratio of data to use for training (default: 0.8)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        logger.info("Preparing training and testing data")
        
        # Extract patent data from graph
        patent_data = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            patent_data.append({
                'patent_id': node,
                'title': node_data.get('title', ''),
                'decision': node_data.get('decision', ''),
                'date_published': node_data.get('date_published', ''),
                'abstract': node_data.get('abstract', ''),
                'claims': node_data.get('claims', []),
                'description': node_data.get('description', '')
            })
        
        logger.info(f"Extracted {len(patent_data)} patents from graph")
        
        # Split into train and test
        random.seed(random_seed)
        random.shuffle(patent_data)
        
        split_idx = int(len(patent_data) * train_ratio)
        train_data = patent_data[:split_idx]
        test_data = patent_data[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} training, {len(test_data)} testing")
        
        return train_data, test_data
    
    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the Mistral model on patent data.
        
        Args:
            training_data: List of patent data for training
            
        Returns:
            Training results
        """
        logger.info("Initializing Mistral model for training")
        
        # Initialize model
        self.model = MistralModel(self.model_path)
        
        # Train the model
        training_results = self.model.train_on_patent_data(training_data)
        
        logger.info("Model training completed")
        return training_results
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of patent data for testing
            
        Returns:
            Evaluation results with metrics
        """
        logger.info("Evaluating model on test data")
        
        if not self.model:
            raise ValueError("Model not initialized. Please train the model first.")
        
        true_labels = []
        predicted_labels = []
        predictions_with_reasoning = []
        
        # Make predictions on test data
        for patent in tqdm(test_data, desc="Making predictions"):
            # Get true label
            true_label = patent['decision'].lower()
            true_labels.append(true_label)
            
            # Make prediction
            prediction_result = self.model.predict_patent_decision(patent)
            predicted_label = prediction_result['prediction']
            predicted_labels.append(predicted_label)
            
            # Store detailed prediction
            predictions_with_reasoning.append({
                'patent_id': patent['patent_id'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'reasoning': prediction_result['reasoning'],
                'confidence': prediction_result['confidence']
            })
        
        # Calculate metrics
        evaluation_results = self.model.evaluate_predictions(
            true_labels, predicted_labels
        )
        
        # Add detailed predictions to results
        evaluation_results['detailed_predictions'] = predictions_with_reasoning
        
        logger.info(f"Evaluation completed: Accuracy = {evaluation_results['accuracy']:.3f}")
        
        return evaluation_results
    
    def run_complete_analysis(self, json_directory: str, output_dir: str = "results", 
                            train_ratio: float = DEFAULT_TRAIN_RATIO, random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, Any]:
        """
        Run the complete patent analysis pipeline.
        
        Args:
            json_directory: Directory containing JSON patent files
            output_dir: Directory to save results
            train_ratio: Ratio of data for training
            random_seed: Random seed for reproducibility
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete patent analysis pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if graph file already exists
        graph_path = os.path.join(output_dir, "patent_graph.pkl")
        
        if os.path.exists(graph_path):
            logger.info(f"Graph file found at {graph_path}. Skipping conversion and loading existing graph.")
            # Load existing graph and get stats
            self.load_graph(graph_path)
            conversion_stats = self.converter.get_graph_stats()
        else:
            logger.info("No existing graph found. Converting JSON files to graph.")
            # Step 1: Convert JSON to graph
            conversion_stats = self.convert_json_to_graph(json_directory, graph_path)
            # Load the newly created graph
            self.load_graph(graph_path)
        
        # Step 2: Prepare training and testing data
        train_data, test_data = self.prepare_training_data(train_ratio, random_seed)
        
        # Step 3: Train model
        training_results = self.train_model(train_data)
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_model(test_data)
        
        # Step 5: Save results
        results = {
            'conversion_stats': conversion_stats,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_split': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_ratio': train_ratio,
                'random_seed': random_seed
            }
        }
        
        # Save results to file
        results_path = os.path.join(output_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed predictions
        predictions_path = os.path.join(output_dir, "detailed_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(evaluation_results['detailed_predictions'], f, indent=2)
        
        logger.info(f"Complete analysis saved to {output_dir}")
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of the analysis results."""
        print("\n" + "="*60)
        print("PATENT ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        # Conversion stats
        print(f"\n📊 GRAPH CONVERSION:")
        print(f"   Total patents: {results['conversion_stats']['total_patents']}")
        print(f"   Decision breakdown:")
        for decision, count in results['conversion_stats']['decision_breakdown'].items():
            print(f"     {decision}: {count}")
        
        # Data split
        print(f"\n📈 DATA SPLIT:")
        print(f"   Training samples: {results['data_split']['train_samples']}")
        print(f"   Testing samples: {results['data_split']['test_samples']}")
        print(f"   Train ratio: {results['data_split']['train_ratio']}")
        
        # Evaluation metrics
        eval_results = results['evaluation_results']
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Accuracy: {eval_results['accuracy']:.3f}")
        if eval_results['roc_auc']:
            print(f"   ROC-AUC: {eval_results['roc_auc']:.3f}")
        print(f"   Total test samples: {eval_results['total_samples']}")
        print(f"   Accepted in test: {eval_results['accepted_count']}")
        print(f"   Rejected in test: {eval_results['rejected_count']}")
        
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(eval_results['classification_report'])
        
        print("="*60)
    
    def check_existing_results(self, output_dir: str) -> bool:
        """
        Check if analysis results already exist in the output directory.
        
        Args:
            output_dir: Directory to check for existing results
            
        Returns:
            True if results exist, False otherwise
        """
        results_path = os.path.join(output_dir, "analysis_results.json")
        predictions_path = os.path.join(output_dir, "detailed_predictions.json")
        graph_path = os.path.join(output_dir, "patent_graph.pkl")
        
        return all(os.path.exists(path) for path in [results_path, predictions_path, graph_path])
    
    def load_existing_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Load existing analysis results from the output directory.
        
        Args:
            output_dir: Directory containing existing results
            
        Returns:
            Loaded results dictionary
        """
        results_path = os.path.join(output_dir, "analysis_results.json")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found at {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded existing results from {results_path}")
        return results

    def run_model_analysis_only(self, graph_path: str, output_dir: str = "results",
                               train_ratio: float = DEFAULT_TRAIN_RATIO, random_seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, Any]:
        """
        Run only model training and evaluation using an existing graph file.
        
        Args:
            graph_path: Path to existing graph file
            output_dir: Directory to save results
            train_ratio: Ratio of data for training
            random_seed: Random seed for reproducibility
            
        Returns:
            Analysis results (without conversion stats)
        """
        logger.info("Running model analysis only (using existing graph)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing graph
        self.load_graph(graph_path)
        conversion_stats = self.converter.get_graph_stats()
        
        # Prepare training and testing data
        train_data, test_data = self.prepare_training_data(train_ratio, random_seed)
        
        # Train model
        training_results = self.train_model(train_data)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(test_data)
        
        # Save results
        results = {
            'conversion_stats': conversion_stats,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_split': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_ratio': train_ratio,
                'random_seed': random_seed
            }
        }
        
        # Save results to file
        results_path = os.path.join(output_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed predictions
        predictions_path = os.path.join(output_dir, "detailed_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(evaluation_results['detailed_predictions'], f, indent=2)
        
        logger.info(f"Model analysis results saved to {output_dir}")
        
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Patent Analysis System")
    parser.add_argument("--json-directory", required=True, 
                       help="Directory containing JSON patent files")
    parser.add_argument("--output-dir", default="results",
                       help="Directory to save results")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                       help=f"Ratio of data for training (default: {DEFAULT_TRAIN_RATIO})")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                       help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
    parser.add_argument("--model-path", default="mistral-pytorch-7b-instruct-v0.1-hf-v1",
                       help="Path to Mistral model")
    parser.add_argument("--skip-if-exists", action="store_true",
                       help="Skip analysis if results already exist")
    parser.add_argument("--force-conversion", action="store_true",
                       help="Force graph conversion even if graph file exists")
    
    args = parser.parse_args()
    
    # Initialize system
    system = PatentAnalysisSystem(args.model_path)
    
    # Check if results already exist
    if args.skip_if_exists and system.check_existing_results(args.output_dir):
        logger.info(f"Results already exist in {args.output_dir}. Loading existing results.")
        results = system.load_existing_results(args.output_dir)
    else:
        # Run complete analysis
        results = system.run_complete_analysis(
            args.json_directory,
            args.output_dir,
            args.train_ratio,
            args.random_seed
        )
    
    # Print summary
    system.print_results_summary(results)

if __name__ == "__main__":
    main() 