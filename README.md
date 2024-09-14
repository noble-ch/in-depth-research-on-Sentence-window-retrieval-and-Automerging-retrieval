# Retrieval-Augmented Generation (RAG) System

## Overview

**Retrieval-Augmented Generation (RAG)** is a hybrid approach that combines retrieval mechanisms with generation models to enhance information processing and response generation. This system retrieves relevant documents or data from large corpora and then generates coherent and contextually accurate responses based on the retrieved information.

## Project Description

This project demonstrates a simple implementation of a sentence window retrieval system, focusing on the concept of RAG. The example uses a text document to split it into sentence-level windows and then retrieves the best matching window based on a given query.

## Key Components

1. **Sentence Window Retrieval**: The system splits the document into sentences and creates "windows" of consecutive sentences. It then retrieves the most relevant window based on the query.

2. **TF-IDF Vectorization**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) to convert text data into numerical vectors for similarity computation.

3. **Cosine Similarity**: Measures the cosine of the angle between vectors to determine the similarity between the query and each sentence window.

## Requirements

- `nltk`: For natural language processing tasks such as sentence tokenization.
- `scikit-learn`: For vectorizing text and calculating cosine similarity.

You can install the required packages using `pip`:

```bash
pip install nltk scikit-learn
```
