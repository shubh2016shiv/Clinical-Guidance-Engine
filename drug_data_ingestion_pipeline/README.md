# Pharmaceutical Data Ingestion Pipeline

## Overview

The Pharmaceutical Data Ingestion Pipeline is a comprehensive data processing system that transforms raw pharmaceutical datasets into a unified, searchable knowledge base with AI-powered semantic search capabilities. Think of it as a **pharmaceutical librarian** that organizes thousands of drug records, deduplicates similar entries, and creates an intelligent search system that helps healthcare professionals and AI systems find relevant drug information instantly.

### Business Value in Simple Terms

**For Healthcare Organizations:**
- **Faster Drug Research**: Instead of manually searching through thousands of drug files, researchers can find relevant medications using natural language queries like "blood pressure medications for elderly patients"
- **Reduced Errors**: Automated standardization prevents dangerous mix-ups between similar drug names
- **Better Decision Support**: AI can suggest drug alternatives and identify potential interactions based on chemical classifications

**For Developers:**
- **Production-Ready Data**: Clean, deduplicated pharmaceutical data ready for machine learning and AI applications
- **Vector Search**: Semantic search capabilities for building drug recommendation systems
- **Scalable Architecture**: Handles datasets from hundreds to millions of drug records

**For Researchers:**
- **Comprehensive Coverage**: Access to structured drug classification data including therapeutic categories, chemical families, and administration routes
- **High-Quality Data**: Professional-grade data cleaning and standardization processes

---

## Architecture Overview

### Data Flow Diagram

```
Raw Data Sources (CSV/Excel)
           ↓
    ┌─────────────────┐
    │   ETL Pipeline  │ ← Extract → Transform → Deduplicate
    └─────────────────┘
           ↓
    Standardized Drug Records
           ↓
    ┌─────────────────┐
    │ Embedding Gen   │ ← Google Gemini AI API
    └─────────────────┘
           ↓
    Vector Embeddings
           ↓
    ┌─────────────────┐
    │ Milvus Vector   │ ← Similarity Search Database
    │    Database     │
    └─────────────────┘
           ↓
    Semantic Search API
```

### Key Components

1. **ETL Pipeline** (`drugs_etl_pipeline.py`)
   - Extracts data from CSV and Excel files
   - Transforms drug names, dosages, and classifications
   - Deduplicates similar records intelligently

2. **Vector Database Ingestion** (`milvus_drug_ingestion_pipeline.py`)
   - Generates AI embeddings using Google Gemini
   - Stores data in Milvus vector database
   - Creates searchable index for semantic queries

3. **Configuration Management** (`config_loader.py`)
   - YAML-based configuration with validation
   - Environment variable support
   - Type-safe configuration loading

4. **Data Sources**
   - **RxNorm Classification**: Official US National Library of Medicine drug data
   - **Drug Class Reference**: Pharmaceutical classification mappings

---

## Quick Start

### Prerequisites

- **Python 3.7+** with pip
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **Milvus Vector Database** ([Setup instructions](#milvus-setup))

### Installation

1. **Clone and Setup**
```bash
# Navigate to pipeline directory
cd drug_data_ingestion_pipeline

# Install dependencies
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API key
# GEMINI_API_KEY=your_actual_api_key_here
```

3. **Configure Pipeline**
```bash
# Review and modify ingestion_config.yaml as needed
# Key settings: input files, embedding dimensions, Milvus connection
```

### Basic Usage

```python
from config_loader import ConfigLoader
from drugs_etl_pipeline import PharmaceuticalETLPipeline
from milvus_drug_ingestion_pipeline import VectorDBIngestionPipeline

# Load configuration
config = ConfigLoader("ingestion_config.yaml").load()

# Run ETL pipeline
etl_pipeline = PharmaceuticalETLPipeline(config=config)
records = etl_pipeline.run_pipeline(config.data_sources.input_files)

# Save standardized data
etl_pipeline.save_to_csv("standardized_drugs.csv")

# Ingest into vector database
vector_pipeline = VectorDBIngestionPipeline(config)
stats = vector_pipeline.run(records)

print(f"Ingestion complete: {stats['records_inserted']} records processed")
```

### Command Line Usage

```bash
# Run complete pipeline
python drugs_etl_pipeline.py

# Run vector ingestion only
python milvus_drug_ingestion_pipeline.py
```

---

## Data Schema Documentation

### Standardized Drug Record

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `drug_name` | str | Core drug name without dosage/route  | "Metformin" |
| `drug_class` | str | Primary classification | "Biguanides" |
| `drug_sub_class` | str | Specific chemical subgroup | "Non-selective beta blocking agents" |
| `therapeutic_category` | str | Medical condition treated | "Drugs used in diabetes" |
| `dosages` | List[str] | Available strengths | ["500 MG", "850 MG", "1000 MG"] |
| `route_of_administration` | str | How drug enters body | "Oral", "Intravenous" |
| `formulation` | str | Physical form | "Tablet", "Solution" |
| `search_text` | str | AI-generated searchable text | "Drug: Metformin \| Dosages: [500 MG, 850 MG] \| Class: Biguanides" |
| `source_files` | Set[str] | Origin file references | {"rxnorm_complete_drug_classification.csv"} |

### Database Schema (Milvus Collection)

```yaml
collection_name: "pharmaceutical_drugs"
fields:
  - name: "id"           # Auto-generated primary key
  - name: "drug_name"    # VARCHAR(500) - Original drug name
  - name: "drug_class"   # VARCHAR(200) - Drug class
  - name: "drug_sub_class" # VARCHAR(200) - Drug subclass
  - name: "therapeutic_category" # VARCHAR(200) - Medical use
  - name: "route_of_administration" # VARCHAR(100) - Administration route
  - name: "formulation"  # VARCHAR(100) - Physical form
  - name: "dosage_strengths" # VARCHAR(2000) - JSON array of dosages
  - name: "search_text"  # VARCHAR(2000) - Searchable text
  - name: "embedding"    # FLOAT_VECTOR(768) - AI embedding vector
```

---

## ⚙️ Configuration Details

### Configuration File Structure (`ingestion_config.yaml`)

#### Data Sources Section
```yaml
data_sources:
  input_files:
    - "../pharmaceutical_knowledge_base/rxnorm_complete_drug_classification.csv"
    - "../pharmaceutical_knowledge_base/drug_class_subclass_reference.xlsx"
  output_directory: "drug_data_ingestion_pipeline"
  standardized_data_file: "standardized_drugs.csv"
  encoding: "utf-8"
```

#### Embedding Configuration
```yaml
gemini:
  api_key: "${GEMINI_API_KEY}"  # Environment variable
  model_name: "models/text-embedding-004"
  task_type: "retrieval_document"
  embedding:
    dimensions: 768  # Options: 256, 512, 768, 1024
    normalize: true
```

#### Vector Database Configuration
```yaml
milvus:
  connection:
    host: "localhost"
    port: 19530
  collection:
    name: "pharmaceutical_drugs"
    embedding_dimension: 768  # Must match Gemini dimensions
  index:
    type: "HNSW"    # Index type: HNSW, IVF_FLAT, etc.
    metric_type: "IP"  # Similarity metric: IP (cosine) or L2
```

#### Search Text Generation
```yaml
search_text:
  template: "Drug: {drug_name} | Dosages: [{dosages}] | Form: {formulation} | Route: {route} | Class: {drug_class}"
  include_fields:
    drug_name: true
    dosages: true
    formulation: true
    route_of_administration: true
    drug_class: true
```

### Environment Variables

Required environment variables (set in `.env` file):

```bash
# Google Gemini API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Custom configuration
CUSTOM_CONFIG_PATH=path/to/custom_config.yaml
```

---

## Usage Examples

### Basic Drug Search Example

```python
# Search for diabetes medications
search_query = "diabetes medications for type 2 diabetes"

# Generate embedding for query
query_embedding = embedding_generator.generate_embedding(search_query)

# Search vector database
search_params = {
    "vector": query_embedding.tolist(),
    "param": {"ef": 64},  # Search parameter for HNSW index
    "limit": 10,          # Return top 10 results
    "expr": None
}

results = collection.search(**search_params)

# Display results
for result in results[0]:
    print(f"Drug: {result.entity.get('drug_name')}")
    print(f"Classification: {result.entity.get('drug_class')}")
    print(f"Score: {result.distance:.3f}")
```

### Custom Search with Filters

```python
# Search for oral medications only
search_params = {
    "vector": query_embedding.tolist(),
    "param": {"ef": 64},
    "limit": 5,
    "expr": "route_of_administration == 'oral'"
}

oral_drugs = collection.search(**search_params)
```

### Batch Processing Example

```python
# Process multiple drug files
input_files = [
    "drug_data_1.csv",
    "drug_data_2.xlsx",
    "drug_data_3.csv"
]

records = []
for file_path in input_files:
    pipeline = PharmaceuticalETLPipeline(config)
    file_records = pipeline.run_pipeline([file_path])
    records.extend(file_records)

# Remove duplicates
from drugs_etl_pipeline import DataDeduplicator
unique_records = DataDeduplicator.deduplicate_records(records)

print(f"Processed {len(records)} total records")
print(f"After deduplication: {len(unique_records)} unique records")
```

### Advanced Search Text Configuration

```yaml
# Custom search template for pharmaceutical research
search_text:
  template: "Drug: {drug_name} | Strength: [{dosages}] | Route: {route} | Category: {therapeutic_category} | Class: {drug_class} | Subclass: {drug_sub_class}"
  include_fields:
    drug_name: true
    dosages: true
    route_of_administration: true
    therapeutic_category: true
    drug_class: true
    drug_sub_class: true
  dosage_delimiter: " | "
  empty_field_placeholder: "N/A"
  include_empty_fields: true
```

---

## Milvus Setup

### Local Installation (Docker)

```bash
# Pull Milvus standalone
docker pull milvusdb/milvus:latest

# Start Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest milvus run standalone
```

### Using Milvus Lite (Python)

```bash
pip install pymilvus
```

```python
# Ingestion pipeline will automatically start Milvus Lite
from milvus import Milvus
client = Milvus(uri="http://localhost:19530")
```

### Cloud Setup (Zilliz)

```yaml
milvus:
  connection:
    host: "your-cluster.zillizcloud.com"
    port: 19530
    user: "your-username"
    password: "your-password"
    secure: true
```

---

## Performance Considerations

### Batch Size Optimization

```yaml
# Recommended settings for different dataset sizes
etl:
  batch_size: 1000  # For processing large CSV files

gemini:
  batch_size: 100   # Balance speed vs API limits
  requests_per_second: 2.0  # Respect API limits

milvus:
  insertion:
    batch_size: 100  # Optimal for most cases
```

### Index Performance Tuning

```yaml
# HNSW Index (recommended for most use cases)
index:
  type: "HNSW"
  params:
    M: 16              # Higher = better accuracy, more memory
    efConstruction: 200 # Higher = better index quality

# IVF Index (for very large datasets)
index:
  type: "IVF_FLAT"
  params:
    nlist: 128         # Number of clusters

search:
  params:
    ef: 64             # Search-time depth
    nprobe: 10         # Clusters to search
```

---

## Error Handling Procedures

### Common Error Scenarios

#### 1. Configuration Errors

**Problem**: Invalid YAML configuration
```python
try:
    config = ConfigLoader("config.yaml").load()
except ValidationError as e:
    print("Configuration validation failed:")
    print(e)
    # Fix: Check YAML syntax and required fields
```

**Solution**: Validate configuration before running
```bash
# Test configuration loading
python -c "from config_loader import ConfigLoader; ConfigLoader('ingestion_config.yaml').load()"
```

#### 2. API Rate Limiting

**Problem**: Gemini API rate limit exceeded
```python
# Error: "Rate limit exceeded"
# Solution: Reduce requests_per_second in config
gemini:
  requests_per_second: 1.0  # Reduce from 2.0 to 1.0
```

#### 3. Dimension Mismatch

**Problem**: Embedding dimension mismatch between Gemini and Milvus
```
ERROR: Embedding dimension mismatch: gemini.embedding.dimensions=768 
       but milvus.collection.embedding_dimension=512
```

**Solution**: Sync dimensions in configuration
```yaml
# Auto-sync in config_loader.py or manually set
gemini:
  embedding:
    dimensions: 768
milvus:
  collection:
    embedding_dimension: 768  # Must match
```

#### 4. File Not Found

**Problem**: Input files missing
```python
# Error: FileNotFoundError: Input file not found: drug_data.csv
# Solution: Check file paths in configuration
data_sources:
  input_files:
    - "path/to/actual_file.csv"  # Verify path exists
```

### Retry Logic Implementation

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

---

## API Reference

### PharmaceuticalETLPipeline Class

```python
class PharmaceuticalETLPipeline:
    def __init__(self, config: IngestionConfig)
    
    def run_pipeline(self, file_paths: List[str]) -> List[StandardizedDrugRecord]
        """Execute complete ETL pipeline"""
        
    def extract_from_file(self, file_path: str) -> pd.DataFrame
        """Extract data from single file"""
        
    def transform_dataframe(self, df: pd.DataFrame, source_file: str) -> List[StandardizedDrugRecord]
        """Transform dataframe to standardized records"""
        
    def export_to_dataframe(self) -> pd.DataFrame
        """Export standardized records to pandas DataFrame"""
        
    def save_to_csv(self, output_path: str)
        """Save standardized records to CSV"""
```

### VectorDBIngestionPipeline Class

```python
class VectorDBIngestionPipeline:
    def __init__(self, config: IngestionConfig)
    
    def run(self, records: List[StandardizedDrugRecord]) -> Dict[str, Any]
        """Execute complete ingestion pipeline"""
        
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]
        """Generate embeddings for search texts"""
        
    def search_similar_drugs(self, query: str, top_k: int = 10) -> List[Dict]
        """Search for similar drugs using semantic search"""
```

### Configuration Classes

```python
class IngestionConfig(BaseModel):
    """Complete ingestion configuration"""
    data_sources: DataSourceConfig
    etl: ETLConfig
    gemini: GeminiConfig
    milvus: MilvusConfig
    search_text: SearchTextConfig
    error_handling: ErrorHandlingConfig
```

---

## Troubleshooting Guide

### Installation Issues

**Problem**: Missing dependencies
```bash
# Solution: Install all requirements
pip install -r requirements.txt

# If using conda environment
conda install pandas pymilvus google-generativeai
```

**Problem**: Python version mismatch
```bash
# Check Python version
python --version  # Should be 3.7+

# Install specific Python version if needed
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

### Configuration Issues

**Problem**: Environment variable not found
```bash
# Solution: Check .env file exists and is loaded
cat .env
# Should show: GEMINI_API_KEY=your_key_here

# Debug environment loading
python -c "from config_loader import ConfigLoader; loader = ConfigLoader(); print('Env loaded:', loader._load_env_file())"
```

**Problem**: Invalid YAML syntax
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('ingestion_config.yaml'))"
```

### Runtime Issues

**Problem**: Milvus connection fails
```bash
# Check if Milvus is running
docker ps | grep milvus

# Check port accessibility
telnet localhost 19530

# Start Milvus if not running
docker start milvus-standalone
```

**Problem**: Gemini API errors
```bash
# Test API key validity
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"test"}]}]}' \
     -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
```

### Performance Issues

**Problem**: Slow processing
```yaml
# Solutions:
# 1. Increase batch sizes
gemini:
  batch_size: 200  # Increase from 100

# 2. Enable parallel processing
etl:
  parallel_processing: true
  max_workers: 8

# 3. Optimize index parameters
milvus:
  index:
    params:
      M: 32    # Increase for better performance
      efConstruction: 400
```

---

### Code Standards

- **Type Hints**: All functions must include type annotations
- **Docstrings**: Use Google-style docstrings
- **Testing**: Minimum 80% code coverage
- **Linting**: Pass flake8 and black formatting

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request with detailed description

---

## License

This project is licensed under the MIT License.

**Data Sources:**
- RxNorm data is provided by the U.S. National Library of Medicine
- Pharmaceutical classifications follow WHO ATC guidelines
- Free for educational and research use

---

## Support

### Getting Help

1. **Documentation**: Check this README and code comments
2. **Issues**: Open GitHub issues for bugs and feature requests
3. **Discussions**: Use GitHub Discussions for questions
4. **Email**: Contact maintainers for security issues

### Common Resources

- [Milvus Documentation](https://milvus.io/docs)
- [Google Gemini API](https://ai.google.dev/)
- [RxNav API](https://rxnav.nlm.nih.gov/)
- [ATC Classification](https://www.whocc.no/atc_ddd_index/)

---

*Last Updated: November 2025*

---
