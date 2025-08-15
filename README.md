# Automated Concept Inventories & MCQs Generation

## Overview
This project automates the generation of concept inventories and multiple-choice questions (MCQs) for introductory computer science (CS1) courses in Java. It integrates large language models (LLMs) with structured misconception data and an AST-based classifier to produce pedagogically valuable assessment items.

## Features
- **Text Parsing & Concept Extraction**: Extracts CS1 concepts from Java textbooks.
- **MCQ Generation**: Produces question stems, correct answers, Bloom's taxonomy levels, explanations, and distractors.
- **Misconception Classification**: Uses an AST-based model to identify likely structural misconceptions in code.
- **Distractor Database**: Incorporates a curated database of misconception-based distractors, with fallback options.

## File Structure
- 'generate_mcq4.py' – Main pipeline script.
- 'ast_inference.py' – Loads trained AST model and detects misconceptions.
- 'ast_classifier.py' – Model definition for AST-based classification.
- 'train_ast_classifier.py' – Training script for AST classifier.
- 'dataset.json' – Dataset for training the AST classifier.
- 'ast_model.pt' – Pre-trained AST classifier.
- 'PDFs/' – Java textbooks.
- 'cleaned_again2_classified_distractors' – Curated distractor database.
- 'fallback_distractors.csv' – Backup distractor list.
- 'concept_inventory2.json' & 'concept_inventory2.csv' – Extracted concept inventory.
- 'mcq_inventory2.csv' – Generated MCQs.

## Requirements
- Python 3.9+
- PyMuPDF ('fitz')
- PyTorch
- OpenAI Python client
- python-dotenv

## Setup
1. Clone the repository.
2. Place Java textbook PDFs into the 'PDFs/' directory.
3. Create a '.env' file with your OpenAI API key:
   '''
   OPENAI_API_KEY=your_api_key_here
   '''
4. Install dependencies:
   '''bash
   pip install -r requirements.txt
   '''

## Usage
Run the main pipeline:
'''bash
python generate_mcq4.py
'''

## Output
- Concept inventory in JSON and CSV formats.
- MCQs with question stems, answers, distractors, Bloom levels, and explanations.

## License
This project is for academic and research purposes only.
