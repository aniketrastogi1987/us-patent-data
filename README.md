
## Workflow Overview

The system follows a 3-step process:

1. **JSON to Graph Conversion**: Convert patent JSON files to NetworkX graph database
2. **Model Training**: Train Mistral 7B model on patent data with 99.95:0.05 train/test split
3. **Evaluation**: Generate predictions and calculate accuracy, ROC-AUC metrics

## System Architecture

```
JSON Files → NetworkX Graph → Train/Test Split → Mistral 7B (GPU) → Predictions & Metrics
```

### Components

- **`networkx_patent_converter.py`**: Converts JSON files to NetworkX graph
- **`models/mistral_model.py`**: Mistral 7B model wrapper with GPU acceleration
- **`patent_analysis_system.py`**: Main system that orchestrates the entire pipeline

## Performance Optimization

The system includes intelligent optimization features:


## Installation

1. Install dependencies optimized for Apple Silicon:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Mistral 7B model in the `mistral-pytorch-7b-instruct-v0.1-hf-v1/` directory
```

## Usage

### Quick Start

Run the complete analysis pipeline with GPU acceleration:

```bash
python patent_analysis_system.py --json-directory hupd_extracted --output-dir results
```

The system will automatically detect and use your Apple Silicon GPU for optimal performance.


```bash
# Skip entire analysis if results exist
python patent_analysis_system.py --json-directory hupd_extracted --output-dir results


### Step-by-Step Usage

#### 1. Convert JSON to Graph (if needed)
```bash
python networkx_patent_converter.py --input-path hupd_extracted --output-path patent_graph.pkl
```

#### 2. Run Complete Analysis
```bash
python patent_analysis_system.py \
    --json-directory hupd_extracted \
    --output-dir results \
    --train-ratio 0.9995 \
    --random-seed 42
```

### Parameters

- `--json-directory`: Directory containing patent JSON files (required)
- `--output-dir`: Directory to save results (default: "results")
- `--train-ratio`: Ratio for train/test split (default: 0.9995)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--model-path`: Path to Mistral model (default: "mistral-pytorch-7b-instruct-v0.1-hf-v1")
- `--skip-if-exists`: Skip analysis if results already exist
- `--force-conversion`: Force graph conversion even if graph file exists

## Output

The system generates:

1. **`patent_graph.pkl`**: NetworkX graph database
2. **`analysis_results.json`**: Complete analysis results with metrics
3. **`detailed_predictions.json`**: Individual predictions with reasoning


## Data Requirements

### JSON File Format

Each JSON file should contain a single patent with these fields:

```json
{
    "application_number": "patent_id",
    "title": "Patent Title",
    "decision": "ACCEPTED|REJECTED",
    "date_published": "YYYYMMDD",
    "abstract": "Patent abstract text",
    "claims": ["claim1", "claim2", ...],
    "full_description": "Detailed patent description text"
}
```

### Decision Values

Only patents with decisions `ACCEPTED`, `REJECTED`, `GRANTED`, or `NOT GRANTED` are processed.

### Required Fields

The system extracts and uses the following fields for analysis:
- **title**: Patent title (limited to 200 characters)
- **abstract**: Patent abstract (limited to 500 characters)
- **description**: Full patent description (limited to 800 characters)
- **claims**: Patent claims (first 3 claims or 300 characters)
- **date_published**: Publication date
- **decision**: Patent decision status

## Model Details

### Mistral 7B Configuration

#Model Performance
The system evaluates model performance using:

- **Accuracy**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Detailed Analysis**: Individual predictions with reasoning


# Test with small dataset
python patent_analysis_system.py --json-directory test_data --output-dir test_results
```