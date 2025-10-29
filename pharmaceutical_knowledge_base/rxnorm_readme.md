# RxNorm Data Extraction Pipeline

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Medical Terminology](#understanding-the-medical-terminology)
3. [The ATC Classification System](#the-atc-classification-system)
4. [What This Script Does](#what-this-script-does)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output Format](#output-format)
8. [Technical Details](#technical-details)

---

## Introduction

Imagine you walk into a pharmacy and see thousands of medications on the shelves. How do pharmacists, doctors, and healthcare systems organize and understand this overwhelming amount of information? How do they know which medications treat heart conditions versus infections? How do they identify the active ingredients in brand-name drugs?

This script solves exactly that problem. It connects to a comprehensive medical database maintained by the U.S. National Library of Medicine and extracts a structured catalog of drugs, organizing them by their therapeutic purpose and chemical composition. The result is a clean, easy-to-understand spreadsheet that maps every drug ingredient to its medical classification.

Even if you have no medical background, by the end of this document, you'll understand how modern medicine organizes pharmaceutical knowledge and what this script accomplishes.

---

## Understanding the Medical Terminology

Before we dive into what the script does, let's understand the key medical terms involved. Think of this section as your medical dictionary.

### RxNorm: The Universal Language of Medications

**RxNorm** is like the "periodic table" for medications. Just as chemists use the periodic table to standardize how they talk about elements, healthcare professionals use RxNorm to standardize how they talk about medications.

Created and maintained by the U.S. National Library of Medicine, RxNorm provides a standardized naming system for clinical drugs and drug delivery devices. It's used by hospitals, pharmacies, insurance companies, and electronic health record systems across the United States to ensure everyone is speaking the same language when discussing medications.

**Real-world example:** You might know a medication as "Advil," while someone else calls it "ibuprofen," and a third person refers to it as "Motrin." RxNorm ensures that all these names are properly linked to the same underlying concept, preventing dangerous confusion in medical settings.

### RXCUI: The Unique Identifier

**RXCUI** stands for **RxNorm Concept Unique Identifier**. Think of it as a social security number for medications—every drug, ingredient, or medication concept gets its own unique numeric identifier.

For example:
- Ibuprofen (the ingredient) has RXCUI: **5640**
- Aspirin (the ingredient) has RXCUI: **1191**
- Acetaminophen (the ingredient) has RXCUI: **161**

Why is this important? Because medication names can be spelled differently, pronounced differently, or have multiple brand names. The RXCUI ensures that regardless of how someone refers to a medication, the computer system knows exactly which drug is being discussed.

**Analogy:** Imagine trying to organize a library where books could be called by different names in different languages. The RXCUI is like giving each book a unique barcode—no matter what language someone uses to request it, the system knows exactly which book they mean.

### TTY: The Type of Medication Concept

**TTY** stands for **Term Type**. It tells us what kind of medication concept we're looking at. Is it a pure ingredient? A branded product? A specific dose form?

Think of TTY as categories in a grocery store. Just as you have "produce," "dairy," and "frozen foods," medications have categories too:

- **IN (Ingredient)**: The pure, active chemical substance that provides the medical effect
  - Example: "Ibuprofen" is an ingredient (TTY=IN)
  
- **BN (Brand Name)**: The commercial name given by a manufacturer
  - Example: "Advil" is a brand name (TTY=BN)
  
- **SCD (Semantic Clinical Drug)**: A complete description including ingredient, strength, and form
  - Example: "Ibuprofen 200 MG Oral Tablet" (TTY=SCD)
  
- **PIN (Precise Ingredient)**: An ingredient with specific characteristics
  - Example: "Ibuprofen Sodium" (TTY=PIN)

**For our script's purpose**, we primarily focus on **IN (Ingredients)** because we want to understand the fundamental active substances, not the hundreds of different brand names and formulations built around them.

### ATC: The Filing System for Drugs

**ATC** stands for **Anatomical Therapeutic Chemical Classification System**. If RxNorm is the language of medications, then ATC is the filing cabinet that organizes them.

The ATC system was developed by the World Health Organization (WHO) and is used internationally. It classifies drugs based on:
1. **Anatomical** - Which organ system they target (heart, lungs, digestive system, etc.)
2. **Therapeutic** - What medical condition they treat (infection, pain, high blood pressure, etc.)
3. **Chemical** - What chemical family they belong to

This is the heart of how the script organizes medication data, so let's explore it in depth.

---

## The ATC Classification System

The ATC system is hierarchical, meaning it goes from broad categories to increasingly specific ones—like organizing files on your computer from main folders into subfolders.

### The Five Levels of ATC

The ATC system has **five levels** of classification. Our script focuses on the first four levels because they provide the most clinically relevant information.

#### Level 1: Anatomical Main Group (The Body System)

This is the broadest level—it tells us which major body system or general area the drug affects.

**Format:** A single letter (A through V)

**Examples:**
- **A** = Alimentary tract and metabolism (digestive system and metabolic disorders)
- **C** = Cardiovascular system (heart and blood vessels)
- **J** = Antiinfectives for systemic use (antibiotics and antivirals)
- **N** = Nervous system (brain and nerves)
- **R** = Respiratory system (lungs and airways)

**Real-world analogy:** Think of this like the section in a bookstore—"Fiction," "Biography," "Science." You know the general area, but not the specific book yet.

**Complete list of Level 1 categories:**
- **A**: Alimentary tract and metabolism
- **B**: Blood and blood forming organs
- **C**: Cardiovascular system
- **D**: Dermatologicals (skin)
- **G**: Genito-urinary system and sex hormones
- **H**: Systemic hormonal preparations
- **J**: Antiinfectives for systemic use
- **L**: Antineoplastic and immunomodulating agents (cancer drugs)
- **M**: Musculo-skeletal system
- **N**: Nervous system
- **P**: Antiparasitic products, insecticides and repellents
- **R**: Respiratory system
- **S**: Sensory organs (eyes and ears)
- **V**: Various (everything else)

#### Level 2: Therapeutic Subgroup (The Medical Purpose)

This level narrows down to the therapeutic purpose—what the drug is actually used to treat.

**Format:** Letter + two digits (e.g., C07, A10, J01)

**Examples:**
- **C07** = Beta blocking agents (used to treat high blood pressure and heart conditions)
- **A10** = Drugs used in diabetes
- **J01** = Antibacterials for systemic use
- **N02** = Analgesics (pain relievers)

**Real-world analogy:** This is like finding the right aisle in the bookstore section—within "Science," you've found "Biology," not "Physics" or "Chemistry."

**Example walkthrough:**
- **C** = Cardiovascular system (Level 1)
- **C07** = Beta blocking agents within the cardiovascular system (Level 2)

#### Level 3: Pharmacological Subgroup (How It Works)

This level describes the pharmacological mechanism—the specific way the drug works in the body.

**Format:** Letter + two digits + letter (e.g., C07A, A10B, J01C)

**Examples:**
- **C07A** = Beta blocking agents, non-selective (they block all beta receptors)
- **A10B** = Blood glucose lowering drugs, excluding insulins
- **J01C** = Beta-lactam antibacterials, penicillins (how penicillin-type antibiotics work)
- **N02A** = Opioids (pain relievers that work through opioid receptors)

**Real-world analogy:** This is like finding the specific shelf in the aisle—within "Biology," you've found "Marine Biology."

**Example walkthrough:**
- **C** = Cardiovascular system (Level 1)
- **C07** = Beta blocking agents (Level 2)
- **C07A** = Beta blocking agents that are non-selective (Level 3)

#### Level 4: Chemical Subgroup (The Chemical Family)

This is the most specific level we use—it identifies the precise chemical family or subgroup.

**Format:** Letter + two digits + two letters (e.g., C07AA, A10BA, J01CA)

**Examples:**
- **C07AA** = Non-selective beta blocking agents, plain (like propranolol)
- **A10BA** = Biguanides (a specific class of diabetes medications, like metformin)
- **J01CA** = Penicillins with extended spectrum
- **N02AA** = Natural opium alkaloids (like morphine and codeine)

**Real-world analogy:** This is like finding the specific book on the shelf—within "Marine Biology," you've found "Coral Reef Ecosystems of the Pacific."

**Example walkthrough:**
- **C** = Cardiovascular system (Level 1)
- **C07** = Beta blocking agents (Level 2)
- **C07A** = Beta blocking agents, non-selective (Level 3)
- **C07AA** = Non-selective beta blocking agents, plain (Level 4)

#### Level 5: Chemical Substance (The Specific Drug)

While our script doesn't focus on Level 5, it's worth mentioning for completeness. This level identifies the specific chemical substance or drug.

**Format:** Letter + two digits + two letters + two digits (e.g., C07AA01)

**Example:**
- **C07AA01** = Alfuzosin (a specific drug within the C07AA category)

### A Complete Real-World Example

Let's trace a common medication through all ATC levels:

**Metformin** - A medication used to treat type 2 diabetes

- **Level 1: A** = Alimentary tract and metabolism
- **Level 2: A10** = Drugs used in diabetes
- **Level 3: A10B** = Blood glucose lowering drugs, excluding insulins
- **Level 4: A10BA** = Biguanides (the chemical family metformin belongs to)
- **Level 5: A10BA02** = Metformin (the specific drug)

**Another example: Amoxicillin** - A common antibiotic

- **Level 1: J** = Antiinfectives for systemic use
- **Level 2: J01** = Antibacterials for systemic use
- **Level 3: J01C** = Beta-lactam antibacterials, penicillins
- **Level 4: J01CA** = Penicillins with extended spectrum
- **Level 5: J01CA04** = Amoxicillin

### Why This Hierarchical System Matters

The beauty of the ATC system is that it provides context at multiple levels:

1. **For clinicians**: They can quickly understand what a drug does and how it works
2. **For researchers**: They can study entire classes of drugs with similar properties
3. **For regulators**: They can track drug usage patterns across populations
4. **For data scientists**: They can analyze medication trends and patterns

Our script leverages this hierarchical structure to create a comprehensive map of how medications are organized in modern medicine.

---

## What This Script Does

Now that you understand the terminology, let's explore what the script actually accomplishes.

### The Big Picture

The script acts as an intelligent data harvester. It connects to the RxNav API (the online interface to RxNorm) and systematically:

1. **Navigates** through the entire ATC classification tree
2. **Identifies** all drug ingredients at the most specific level (Level 4)
3. **Extracts** their classification information
4. **Organizes** everything into a structured, easy-to-read spreadsheet

Think of it like a librarian creating a master catalog: they walk through every section, every aisle, every shelf, and every book, recording where each item belongs and what it's about.

### The Step-by-Step Process

#### Step 1: Connecting to the RxNav API

The script begins by establishing a connection to RxNav, the National Library of Medicine's web service that provides access to RxNorm data.

**What's an API?** Think of an API (Application Programming Interface) as a waiter in a restaurant. You (the script) don't go into the kitchen (the database) directly. Instead, you tell the waiter (the API) what you want, and they bring it to you in an organized format.

**Technical detail:** The script uses the HTTPS protocol to make secure requests to `https://rxnav.nlm.nih.gov/REST/`. It implements rate limiting (waiting between requests) to be respectful of the server and includes retry logic in case of temporary network issues.

#### Step 2: Traversing the ATC Tree

The script starts at the top of the ATC hierarchy and works its way down, like exploring a family tree from the oldest ancestors to the youngest descendants.

**Process:**
1. Request the classification tree for a Level 1 category (e.g., "C" for Cardiovascular)
2. Receive a nested structure containing all sub-categories
3. Recursively explore each branch:
   - Level 1 → Level 2 → Level 3 → Level 4
   - At each level, record the classification names
   - Continue deeper until reaching Level 4

**Analogy:** Imagine you're creating a map of a city:
- Level 1 = The city itself (New York)
- Level 2 = Boroughs (Manhattan, Brooklyn)
- Level 3 = Neighborhoods (Upper East Side, Williamsburg)
- Level 4 = Specific blocks

You'd walk through each borough, each neighborhood, and each block, recording everything you find.

#### Step 3: Finding Drug Ingredients

When the script reaches a Level 4 classification (the most specific), it asks the API: "What drug ingredients belong to this category?"

**For example**, for the ATC code **C07AA** (non-selective beta blocking agents):
- The script requests all members of this class
- The API returns ingredients like:
  - Propranolol (RXCUI: 8787)
  - Nadolol (RXCUI: 7241)
  - Timolol (RXCUI: 10600)

**Important filtering:** The script specifically requests ingredients (TTY=IN) rather than branded products or specific formulations. Why? Because we want the fundamental active substances, not the thousands of brand names and dose variations.

If "Advil 200mg tablet," "Advil 400mg capsule," "Motrin 200mg tablet," and "Generic Ibuprofen 200mg" were all listed separately, the data would be overwhelming and redundant. Instead, we capture "Ibuprofen" once as the ingredient, along with its classification.

#### Step 4: Handling API Complexity

The RxNav API has evolved over time, and different endpoints may return data in different formats. The script is designed to be robust and handle multiple response formats:

**Response Format 1 (Current API):**
```json
{
  "drugMemberGroup": {
    "drugMember": [
      {
        "minConcept": {
          "rxcui": "5640",
          "name": "Ibuprofen",
          "tty": "IN"
        }
      }
    ]
  }
}
```

**Response Format 2 (Legacy API):**
```json
{
  "rxclassDrugInfoList": {
    "rxclassDrugInfo": [
      {
        "drugMember": {
          "rxcui": "5640",
          "name": "Ibuprofen",
          "tty": "IN"
        }
      }
    ]
  }
}
```

The script checks all possible formats and extracts the data regardless of which structure is returned. This ensures the script continues to work even if the API changes.

#### Step 5: Assembling the Data

For each drug ingredient found, the script creates a complete record that includes:

1. **ingredient_rxcui**: The unique identifier for the ingredient
2. **primary_drug_name**: The standardized name
3. **therapeutic_class_l2**: What condition it treats (Level 2)
4. **drug_class_l3**: How it works pharmacologically (Level 3)
5. **drug_subclass_l4**: Its specific chemical family (Level 4)

**Example record:**
```
ingredient_rxcui: 8787
primary_drug_name: Propranolol
therapeutic_class_l2: Beta blocking agents
drug_class_l3: Beta blocking agents, non-selective
drug_subclass_l4: Non-selective beta blocking agents, plain
```

This record tells us: "Propranolol (ID: 8787) is a beta blocker used for cardiovascular conditions. Specifically, it's a non-selective beta blocker in its plain form."

#### Step 6: Preventing Duplicates

Some drug ingredients may appear in multiple ATC categories because they have multiple medical uses.

**Example:** Aspirin
- Appears in **B01AC06** (Antithrombotic agents - preventing blood clots)
- Also appears in **N02BA01** (Analgesics - pain relief)

The script tracks which ingredient-classification combinations it has already recorded (using the pair of RXCUI and ATC code) and skips duplicates. This ensures each unique classification is recorded once.

#### Step 7: Writing to CSV

As the script processes each ingredient, it immediately writes the record to a CSV (Comma-Separated Values) file—essentially a spreadsheet that can be opened in Excel, Google Sheets, or any data analysis tool.

The file is written incrementally, meaning:
- You can monitor progress by opening the file while the script runs
- If the script is interrupted, you don't lose already-processed data
- The file format is universally compatible with data analysis tools

### Performance and Reliability Features

The script includes several features to ensure reliable operation:

**Rate Limiting:** Waits 0.9 seconds between API requests by default (configurable). This prevents overwhelming the server and respects usage policies.

**Retry Logic:** If a request fails due to temporary network issues, the script automatically retries up to 3 times with exponential backoff (waiting longer between each retry).

**Error Handling:** If processing one ATC category fails, the script logs the error and continues with the next category, ensuring one failure doesn't stop the entire export.

**Logging:** All operations are logged to a timestamped log file for troubleshooting and audit purposes.

**Progress Reporting:** Clear console output shows which categories are being processed and how many records have been extracted.

---

## Installation

### Prerequisites

- **Python 3.7 or higher** - The script is written in Python 3
- **Internet connection** - Required to access the RxNav API

### Setup

1. **Download the script:**
   ```bash
   git clone https://github.com/yourusername/rxnorm-data-extraction.git
   cd rxnorm-data-extraction
   ```

2. **Run the script:**
   The script automatically installs required dependencies (httpx and tqdm) on first run:
   ```bash
   python rxnorm_data_extraction_pipeline.py
   ```

### Required Python Packages

The script requires two external packages:

- **httpx**: A modern HTTP client for making API requests
- **tqdm**: For displaying progress bars

These are automatically installed by the script if not already present.

---

## Usage

### Basic Usage

Extract all drug classifications (all ATC categories):

```bash
python rxnorm_data_extraction_pipeline.py
```

This creates a file named `rxnorm_complete_drug_classification.csv` in the current directory.

### Custom Output File

Specify a custom output filename:

```bash
python rxnorm_data_extraction_pipeline.py --output my_drug_data.csv
```

### Process Specific Categories

Extract only cardiovascular and respiratory drugs:

```bash
python rxnorm_data_extraction_pipeline.py --roots C,R
```

Extract only antibiotics and antivirals:

```bash
python rxnorm_data_extraction_pipeline.py --roots J
```

### Adjust Request Delay

Change the delay between API requests (in seconds):

```bash
python rxnorm_data_extraction_pipeline.py --delay 1.5
```

**Note:** Increasing the delay makes the script slower but reduces server load. Decreasing it speeds up extraction but may trigger rate limiting.

### Run API Test

Test connectivity without full extraction:

```bash
python rxnorm_data_extraction_pipeline.py --smoke-test
```

This queries a few known ATC codes and displays sample results to verify the API is accessible and returning expected data.

### Enable Debug Mode

Write diagnostic files for troubleshooting:

```bash
python rxnorm_data_extraction_pipeline.py --debug
```

When enabled, if any ATC code returns empty results, the script writes a JSON file containing the raw API response for investigation.

### Combined Options

You can combine multiple options:

```bash
python rxnorm_data_extraction_pipeline.py \
  --output cardiovascular_drugs.csv \
  --roots C \
  --delay 1.0 \
  --debug
```

---

## Output Format

### CSV Structure

The output CSV file contains five columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| **ingredient_rxcui** | Unique identifier for the drug ingredient | 8787 |
| **primary_drug_name** | Standardized name of the ingredient | Propranolol |
| **therapeutic_class_l2** | ATC Level 2 - Therapeutic category | Beta blocking agents |
| **drug_class_l3** | ATC Level 3 - Pharmacological category | Beta blocking agents, non-selective |
| **drug_subclass_l4** | ATC Level 4 - Chemical category | Non-selective beta blocking agents, plain |

### Sample Data

Here are actual examples from the output:

```csv
ingredient_rxcui,primary_drug_name,therapeutic_class_l2,drug_class_l3,drug_subclass_l4
8787,Propranolol,Beta blocking agents,Beta blocking agents, non-selective,Non-selective beta blocking agents, plain
6809,Metformin,Drugs used in diabetes,Blood glucose lowering drugs, excluding insulins,Biguanides
723,Amoxicillin,Antibacterials for systemic use,Beta-lactam antibacterials, penicillins,Penicillins with extended spectrum
5640,Ibuprofen,Anti-inflammatory and antirheumatic products,Anti-inflammatory and antirheumatic products, non-steroids,Propionic acid derivatives
161,Acetaminophen,Analgesics,Other analgesics and antipyretics,Anilides
```

### Understanding a Record

Let's decode one complete record:

```csv
6809,Metformin,Drugs used in diabetes,Blood glucose lowering drugs, excluding insulins,Biguanides
```

**Translation:**
- **6809** = This is metformin's unique identifier in RxNorm
- **Metformin** = The standard name for this ingredient
- **Drugs used in diabetes** = Level 2: This drug treats diabetes
- **Blood glucose lowering drugs, excluding insulins** = Level 3: It lowers blood sugar but isn't insulin
- **Biguanides** = Level 4: It belongs to the biguanide chemical family

**What this tells a healthcare professional:**
"Metformin (ID: 6809) is an antidiabetic medication that works by lowering blood glucose through a non-insulin mechanism, specifically as a member of the biguanide class of drugs."

### Data Volume

A complete extraction (all ATC categories) typically yields:

- **10,000 to 15,000 records** (depending on RxNorm database updates)
- **File size**: Approximately 1-2 MB
- **Processing time**: 2-4 hours (depending on network speed and API performance)

---

## Technical Details

### API Rate Limiting

The script implements a default 0.9-second delay between requests. This ensures:

1. **Respectful usage** of the free public API
2. **Compliance** with typical API rate limits
3. **Reliability** - avoiding throttling or temporary bans

The RxNav API is generally generous with rate limits for individual users, but aggressive querying may result in temporary blocks.

### Data Freshness

The RxNorm database is updated monthly by the National Library of Medicine. The script always fetches the most current data available through the API.

### Error Scenarios

The script handles several common error scenarios:

**Network timeouts:** Automatically retries with exponential backoff

**API errors:** Logs the error, continues with next category

**Empty results:** Normal for some ATC codes that have no ingredient-level members

**Malformed responses:** Attempts multiple parsing strategies before giving up

### Performance Optimization

The script optimizes performance through:

1. **Immediate writing**: Records are written to CSV as they're processed, not buffered in memory
2. **Single-pass traversal**: Each ATC tree is traversed only once
3. **Duplicate tracking**: Uses a set (hash table) for O(1) duplicate detection
4. **Connection reuse**: Maintains a persistent HTTP connection rather than creating new ones for each request

### Logging

All operations are logged to a timestamped file: `rxnorm_export_YYYYMMDD_HHMMSS.log`

The log includes:
- Start and end times for each root category
- Errors encountered during processing
- API response issues
- Final statistics

This log is invaluable for troubleshooting if issues occur.

---

## Use Cases

### Healthcare Analytics

Researchers and analysts can use this data to:
- Study medication usage patterns across therapeutic categories
- Identify gaps in drug development for specific conditions
- Analyze the diversity of available treatments for diseases
- Map relationships between drug classes

### Pharmaceutical Research

Drug developers can:
- Identify competitive landscape within therapeutic areas
- Discover potential drug repurposing opportunities
- Understand classification of new compounds
- Analyze market saturation in specific categories

### Clinical Decision Support

Healthcare IT systems can:
- Build drug classification modules
- Implement therapeutic substitution logic
- Create drug-drug interaction warnings based on classes
- Develop clinical pathways based on medication categories

### Education and Training

Medical and pharmacy students can:
- Learn drug classification systematically
- Understand relationships between medications
- Study therapeutic categories with concrete examples
- Build reference materials for clinical practice

### Data Science Projects

Data scientists can:
- Create machine learning features based on drug classifications
- Build recommendation systems for medication alternatives
- Perform natural language processing on drug names
- Develop predictive models for drug efficacy

---

## Limitations and Considerations

### What This Script Does Not Do

**It does not provide:**
- Drug dosing information
- Side effects or contraindications
- Drug-drug interaction data
- Pricing or availability information
- Brand name to generic mappings
- Clinical trial data
- Prescribing guidelines

**For clinical use:** This data should never be used as the sole source for clinical decision-making. Always consult current prescribing information and clinical guidelines.

### Data Completeness

- Some ATC categories may have few or no members in RxNorm
- Not all drugs available worldwide are in RxNorm (primarily focuses on US market)
- Some ingredients may appear in multiple categories
- The ATC system is maintained separately from RxNorm, leading to occasional inconsistencies

### API Dependency

This script depends on the continued availability of the free RxNav API. Changes to the API structure, rate limits, or availability may require script updates.

---

## Troubleshooting

### Common Issues

**"Failed to install httpx/tqdm"**
- Solution: Install manually with `pip install httpx tqdm`

**"HTTP request failed after retries"**
- Cause: Network connectivity issues or API unavailability
- Solution: Check internet connection, try again later, or increase --delay

**"Empty results for ATC code"**
- Cause: Some ATC codes have no ingredient-level members in RxNorm
- Solution: This is normal; the script continues processing other codes

**Script runs very slowly**
- Cause: Network latency or conservative rate limiting
- Solution: Consider reducing --delay slightly (minimum 0.5 seconds recommended)

**Output file has fewer rows than expected**
- Cause: API changes, network interruptions, or duplicate filtering
- Solution: Check log file for errors, try re-running specific roots with --debug

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional validation of extracted data
- Enhanced error recovery mechanisms
- Support for ATC Level 5 extraction
- Integration with other drug databases
- Performance optimizations
- Additional output formats (JSON, Parquet, SQL)

---

## License

This script is provided as-is for educational and research purposes. The RxNorm data accessed through this script is provided by the U.S. National Library of Medicine and is free to use. Please review the [RxNav API Terms of Service](https://rxnav.nlm.nih.gov/) for data usage policies.

---

## Acknowledgments

- **U.S. National Library of Medicine** for maintaining RxNorm and providing free API access
- **World Health Organization** for developing the ATC classification system
- **Open source community** for the excellent httpx and tqdm libraries

---

## Contact and Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Review the log file for detailed error messages
- Use --smoke-test to verify API connectivity
- Enable --debug mode for troubleshooting specific ATC codes

---

## Appendix: Complete ATC Level 1 Categories

| Code | Category | Description |
|------|----------|-------------|
| **A** | Alimentary tract and metabolism | Drugs for digestive system, diabetes, nutrition |
| **B** | Blood and blood forming organs | Anticoagulants, blood products, hemostatics |
| **C** | Cardiovascular system | Heart medications, blood pressure drugs |
| **D** | Dermatologicals | Topical medications for skin conditions |
| **G** | Genito-urinary system and sex hormones | Reproductive system drugs, hormones |
| **H** | Systemic hormonal preparations | Hormones affecting entire body systems |
| **J** | Antiinfectives for systemic use | Antibiotics, antivirals, antifungals |
| **L** | Antineoplastic and immunomodulating agents | Cancer drugs, immune system modifiers |
| **M** | Musculo-skeletal system | Drugs for muscles, bones, joints |
| **N** | Nervous system | Neurological and psychiatric medications |
| **P** | Antiparasitic products | Anti-parasite medications, insecticides |
| **R** | Respiratory system | Lung and airway medications |
| **S** | Sensory organs | Eye and ear medications |
| **V** | Various | Miscellaneous drugs not fitting other categories |

---

*Last updated: 2025*