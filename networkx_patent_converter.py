#!/usr/bin/env python3
"""
NetworkX Patent Data Converter

This script converts patent JSON files to a NetworkX graph structure.
It filters patents based on decision status and creates a graph with patent nodes.
Includes: title, decision, date published, abstract, claims, and description.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import networkx as nx
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatentData:
    """Container for patent data extracted from JSON."""
    patent_id: str
    title: str
    decision: str
    date_published: Optional[str]
    abstract: str
    claims: List[str]
    description: str

class NetworkXPatentConverter:
    """Handles conversion of patent JSON files to NetworkX graph structure."""
    
    def __init__(self):
        """Initialize the converter with an empty NetworkX graph."""
        self.graph = nx.Graph()
        self.patent_count = 0
        self.processed_files = 0
        
    def extract_patent_data(self, json_data: Dict[str, Any]) -> Optional[PatentData]:
        """
        Extract patent data from JSON structure.
        
        Args:
            json_data: Raw JSON data from file
            
        Returns:
            PatentData object or None if invalid
        """
        try:
            # Extract basic patent information
            patent_id = (
                json_data.get('application_number') or 
                json_data.get('patent_id') or 
                json_data.get('id') or 
                json_data.get('patent_number')
            )
            if not patent_id:
                return None
            
            title = json_data.get('title', '').strip()
            if not title:
                return None
                
            decision = json_data.get('decision', '').strip()
            
            # Filter for accepted/rejected patents only
            if decision.upper() not in ['ACCEPTED', 'REJECTED', 'GRANTED', 'NOT GRANTED']:
                return None
            
            # Normalize decision to lowercase for consistency
            decision = decision.lower()
            
            # Extract date published (try multiple possible field names)
            date_published = (
                json_data.get('date_published') or 
                json_data.get('published_date') or 
                json_data.get('filing_date') or 
                json_data.get('application_date') or 
                json_data.get('grant_date') or 
                json_data.get('issue_date')
            )
            
            # Extract abstract
            abstract = json_data.get('abstract', '').strip()
            
            # Extract claims
            claims = []
            if 'claims' in json_data:
                if isinstance(json_data['claims'], list):
                    claims = [str(claim).strip() for claim in json_data['claims'] if claim]
                elif isinstance(json_data['claims'], str):
                    claims = [json_data['claims'].strip()]
            
            # Extract description (try multiple possible field names)
            description = (
                json_data.get('full_description') or 
                json_data.get('description') or 
                json_data.get('detailed_description') or 
                json_data.get('specification') or 
                ''
            ).strip()
            
            return PatentData(
                patent_id=str(patent_id),
                title=title,
                decision=decision,
                date_published=date_published,
                abstract=abstract,
                claims=claims,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract patent data: {e}")
            return None
    
    def add_patent_node(self, patent: PatentData):
        """Add a patent node to the NetworkX graph."""
        try:
            # Add node with patent data as attributes
            self.graph.add_node(
                patent.patent_id,
                title=patent.title,
                decision=patent.decision,
                date_published=patent.date_published,
                abstract=patent.abstract,
                claims_count=len(patent.claims),
                claims=patent.claims,
                description=patent.description
            )
            self.patent_count += 1
            
        except Exception as e:
            logger.error(f"Failed to add patent node {patent.patent_id}: {e}")
    
    def process_patent(self, patent: PatentData):
        """Process a single patent and add it to the graph."""
        try:
            self.add_patent_node(patent)
            
        except Exception as e:
            logger.error(f"Failed to process patent {patent.patent_id}: {e}")
    
    def process_json_file(self, file_path: str) -> int:
        """
        Process a single JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Number of patents processed
        """
        logger.info(f"Processing file: {file_path}")
        processed_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the entire file as a single JSON object
                try:
                    json_data = json.load(f)
                    
                    # Extract patent data
                    patent = self.extract_patent_data(json_data)
                    if patent:
                        self.process_patent(patent)
                        processed_count += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
        
        logger.info(f"✓ Processed {processed_count} patents from {file_path}")
        return processed_count
    
    def find_all_json_files(self, directory_path: str) -> List[str]:
        """
        Recursively find all JSON files in directory and subdirectories.
        
        Args:
            directory_path: Path to directory containing JSON files
            
        Returns:
            List of all JSON file paths
        """
        json_files = []
        
        # Walk through directory recursively
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        return json_files
    
    def process_directory(self, directory_path: str) -> int:
        """
        Process all JSON files in a directory recursively.
        
        Args:
            directory_path: Path to directory containing JSON files
            
        Returns:
            Total number of patents processed
        """
        logger.info(f"Processing directory recursively: {directory_path}")
        
        # Find all JSON files recursively
        json_files = self.find_all_json_files(directory_path)
        
        if not json_files:
            logger.warning(f"No JSON files found in {directory_path}")
            return 0
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        total_processed = 0
        
        # Process files with progress bar
        for file_path in tqdm(json_files, desc="Processing files"):
            processed = self.process_json_file(file_path)
            total_processed += processed
            self.processed_files += 1
        
        logger.info(f"✓ Total patents processed: {total_processed}")
        return total_processed
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded graph."""
        stats = {
            "total_patents": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "processed_files": self.processed_files
        }
        
        # Count by decision
        decision_counts = {}
        for node in self.graph.nodes():
            decision = self.graph.nodes[node].get('decision', 'unknown')
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        stats["decision_breakdown"] = decision_counts
        
        # Count patents with abstracts
        patents_with_abstract = sum(
            1 for node in self.graph.nodes() 
            if self.graph.nodes[node].get('abstract') and 
            self.graph.nodes[node]['abstract'].strip()
        )
        stats["patents_with_abstract"] = patents_with_abstract
        
        # Count patents with claims
        patents_with_claims = sum(
            1 for node in self.graph.nodes() 
            if self.graph.nodes[node].get('claims_count', 0) > 0
        )
        stats["patents_with_claims"] = patents_with_claims
        
        return stats
    
    def save_graph(self, file_path: str):
        """Save the graph to a file."""
        try:
            # Save as pickle for full NetworkX compatibility
            with open(file_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"✓ Graph saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def load_graph(self, file_path: str):
        """Load a graph from a file."""
        try:
            with open(file_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"✓ Graph loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
    
    def export_to_gexf(self, file_path: str):
        """Export graph to GEXF format for visualization tools like Gephi."""
        try:
            nx.write_gexf(self.graph, file_path)
            logger.info(f"✓ Graph exported to GEXF: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export to GEXF: {e}")
    
    def export_to_graphml(self, file_path: str):
        """Export graph to GraphML format."""
        try:
            nx.write_graphml(self.graph, file_path)
            logger.info(f"✓ Graph exported to GraphML: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export to GraphML: {e}")
    
    def create_similarity_edges(self, similarity_threshold: float = 0.5):
        """
        Create edges between patents based on title similarity.
        
        Args:
            similarity_threshold: Minimum similarity score to create an edge
        """
        logger.info("Creating similarity edges...")
        
        # Get all patent titles
        nodes = list(self.graph.nodes())
        edges_created = 0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                title1 = self.graph.nodes[node1].get('title', '').lower()
                title2 = self.graph.nodes[node2].get('title', '').lower()
                
                # Simple similarity based on common words
                words1 = set(title1.split())
                words2 = set(title2.split())
                
                if words1 and words2:
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    similarity = len(intersection) / len(union)
                    
                    if similarity >= similarity_threshold:
                        self.graph.add_edge(node1, node2, similarity=similarity)
                        edges_created += 1
        
        logger.info(f"✓ Created {edges_created} similarity edges")
    
    def analyze_graph(self) -> Dict[str, Any]:
        """Perform basic graph analysis."""
        analysis = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        if self.graph.number_of_nodes() > 0:
            # Connected components
            components = list(nx.connected_components(self.graph))
            analysis["connected_components"] = len(components)
            analysis["largest_component_size"] = max(len(comp) for comp in components) if components else 0
            
            # Degree statistics
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            analysis["average_degree"] = sum(degrees) / len(degrees) if degrees else 0
            analysis["max_degree"] = max(degrees) if degrees else 0
            analysis["min_degree"] = min(degrees) if degrees else 0
        
        return analysis

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Convert patent JSON files to NetworkX graph")
    parser.add_argument("--input-path", required=True, 
                       help="Path to JSON files directory (will process recursively)")
    parser.add_argument("--output-path", default="patent_graph.pkl", 
                       help="Output path for the graph file")
    parser.add_argument("--export-gexf", help="Export to GEXF format")
    parser.add_argument("--export-graphml", help="Export to GraphML format")
    parser.add_argument("--create-similarity-edges", action="store_true",
                       help="Create edges based on title similarity")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                       help="Similarity threshold for creating edges")
    parser.add_argument("--analyze", action="store_true",
                       help="Perform graph analysis")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = NetworkXPatentConverter()
    
    # Process input path
    input_path = Path(args.input_path)
    if input_path.is_file():
        # Single file
        processed = converter.process_json_file(str(input_path))
        logger.info(f"Processed {processed} patents from single file")
    elif input_path.is_dir():
        # Directory (recursive processing)
        processed = converter.process_directory(str(input_path))
        logger.info(f"Processed {processed} patents from directory (recursive)")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Create similarity edges if requested
    if args.create_similarity_edges:
        converter.create_similarity_edges(args.similarity_threshold)
    
    # Save graph
    converter.save_graph(args.output_path)
    
    # Export to other formats if requested
    if args.export_gexf:
        converter.export_to_gexf(args.export_gexf)
    
    if args.export_graphml:
        converter.export_to_graphml(args.export_graphml)
    
    # Print statistics
    stats = converter.get_graph_stats()
    logger.info("Graph Statistics:")
    for stat_name, count in stats.items():
        if stat_name == "decision_breakdown":
            logger.info(f"  {stat_name}:")
            for decision, count in count.items():
                logger.info(f"    {decision}: {count}")
        else:
            logger.info(f"  {stat_name}: {count}")
    
    # Perform analysis if requested
    if args.analyze:
        analysis = converter.analyze_graph()
        logger.info("Graph Analysis:")
        for metric, value in analysis.items():
            logger.info(f"  {metric}: {value}")

if __name__ == "__main__":
    main() 