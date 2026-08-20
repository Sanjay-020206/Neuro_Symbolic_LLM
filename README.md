# Neuro-Symbolic LLM

This project explores a Neuro-Symbolic architecture combining the reasoning capabilities of Large Language Models (LLMs) with a deterministic symbolic engine. 

## Overview
The pipeline takes natural-language scenarios and uses **Qwen/Qwen2.5-1.5B-Instruct** to extract facts and logical rules into a structured JSON format. This structured output ensures that decision-making conditions are clearly defined (e.g., extracting conditions like `>= 700` and `true`/`false`). A symbolic engine then evaluates the facts against the rules to produce a definitive, traceable conclusion without hallucinations.

## Key Features
- **Strict Knowledge Extraction**: LLM processes natural text and converts it into a rigid JSON structure of rules and facts.
- **Symbolic Reasoning**: A deterministic engine processes the structured facts and rules to output the final decision.
- **Traceability**: Decisions are made by the symbolic engine, ensuring you know exactly *why* a decision was made.

## Requirements
- `torch`
- `transformers`
- `accelerate`
- `sentencepiece`

## How it Works
1. **Input text**: Provide a natural language text containing a scenario and rules (e.g., loan eligibility based on credit score).
2. **LLM Extraction**: `Qwen` identifies entities, attributes, values, and constructs rules with logical operators.
3. **Symbolic Execution**: The facts are evaluated against the extracted conditions to reach the conclusion.

## Model 
- **Qwen/Qwen2.5-1.5B-Instruct** 

## Usage
The primary implementation is available in the `neuro_sym.ipynb` Jupyter Notebook.
