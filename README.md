# Automated Concept Inventorues and MCQs Generation for CS Education

This project implements an AI-powered pipeline for **automatic generation of Concept Inventories (CIs)** and **Multiple-Choice Questions (MCQs)** in the domain of introductory computer programming (CS1). The system combines **Large Language Models (LLMs)**, a structured **misconception database**, and **AST-based structural analysis** to produce high-quality, pedagogically valid MCQs.

---

## Features

- **Concept Extraction**  
  Parses Java textbooks (PDF) and extracts core programming concepts grouped by topic.

- **MCQ Generation**  
  Automatically generates:
  - Question stem
  - Correct answer
  - Bloom's Taxonomy level
  - Explanation

- **Distractor Generation**  
  Selects plausible distractors from a curated misconception database or generates them via GPT, enhanced by AST-based misconception detection.

- **Structural Filtering**  
  Uses a trained **BiGRU-based AST classifier** to detect and annotate structural misunderstandings in Java code snippets.

- **Automated Output**  
  Saves generated concepts and MCQs in both **JSON** and **CSV** formats for further analysis.

---

##  Project Structure

'''
.
├── generate_mcq4.py                 # Main script for MCQ generation pipeline
├── ast_inference.py                  # Loads trained AST model and detects misconceptions
├── ast_classifier.py                 # Model definition for AST misconception classification
├── train_ast_classifier.py           # Training script for AST classifier. 
├── dataset.json                      # Dataset for training the AST classifier.
├── ast_model.pt                       # Pre-trained AST misconception classifier
├── PDFs/                              # Java textbooks for concept extraction
├── cleaned_again2_classified_distractors.csv    # Curated misconception-based distractor database
├── fallback_distractors.csv           # Backup distractor list
├── concept_inventory2.json            # Extracted concept inventory (JSON)
├── concept_inventory2.csv             # Extracted concept inventory (CSV)
├── mcq_inventory2.csv                 # Generated MCQs
└── README.md                          # Project documentation
```

---

##  How It Works

### 1. **Concept Extraction**
- Extracts text from Java textbooks using `PyMuPDF (fitz)`
- Cleans and filters irrelevant content
- Splits into manageable chunks
- Sends chunks to **GPT-4o** to extract structured `(topic, concept)` pairs

### 2. **Question & Correct Answer Generation**
- Prompts GPT-4o to generate a question stem, correct answer, Bloom level, and explanation for each concept

### 3. **Distractor Generation**
- Retrieves distractors from the misconception database
- If insufficient, uses GPT-4o to generate new distractors
- Incorporates **AST misconception detection** to guide distractor creation

### 4. **Output Formatting**
- Shuffles options and labels them A-D
- Exports results to CSV and JSON

---

##  Evaluation

- **Evaluation Dimensions**:
  1. Semantic clarity
  2. Distractor misleadingness
  3. Bloom taxonomy alignment
  4. Option diversity & consistency

- **Results**:
  - **82%** of generated MCQs met quality standards
  - Highest scores: semantic clarity (4.07/5), option consistency (4.00/5)
  - Main area for improvement: distractor misleadingness (3.53/5)

---

##  Requirements

- Python 3.9+
- Install dependencies:
  '''bash
  pip install -r requirements.txt
  '''
- '.env' file with your **OpenAI API key**:
  '''env
  OPENAI_API_KEY=your_api_key_here
  '''

---

## ▶️ Running the System

1. Place Java textbook PDFs in the 'PDFs/' folder.
2. Ensure 'fallback_distractors.csv' and 'distractor_output/' exist.
3. Run:
   '''bash
   python generate_mcq4.py
   '''

---

## 🔒 Security Notes

- **Never commit your '.env' file** containing the OpenAI API key.
- If a key is accidentally pushed to GitHub, **revoke it immediately** from your OpenAI account dashboard.

---

##  License

This project is for academic and research purposes only. Redistribution or commercial use is not permitted without explicit permission.

---

##  Contact

For questions or collaboration:
- **Author**: Xi Chen  
- **Email**: [c1781968765x@gmail.com]
