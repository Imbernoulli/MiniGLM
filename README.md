# MiniGLM Model Training Experiment Report

It is the second assignment during the course "Program Design Training (30240522)"(2023) at Tsinghua University's Department of Computer Science and Technology.

## Overview

This document serves as a comprehensive guide and report for the MiniGLM model training experiment. The experiment focuses on the development and training of models capable of understanding and generating text based on various Chinese literature and Q&A datasets. This README provides insights into the dataset preparation, model training, and evaluation processes, as well as highlights the novel approaches and findings of the experiment.

## Repository Structure

The repository contains several key directories and files essential for the training and evaluation of the MiniGLM model. Below is an overview of the repository structure:

- `config/`: Contains configuration files for model training and evaluation.
- `data/`: Includes scripts for data preparation (`prepare.py`, `prepare_jynews.py`, `prepare_zhihudata.py`, etc.) for a range of datasets, including news, Q&A, and literature related to Jin Yong's works.
- `lstm_model.py`, `lstm_train.py`: LSTM model definition and training script for exploratory purposes.
- `model.py`, `train.py`: Core model definition and training script for the MiniGLM model.
- `data_utils.py`, `evaluations.py`, `visualize.py`: Utilities for data handling, model evaluation, and result visualization.
- `sample.py`, `sample_for_test.py`, `sample_gradio.py`: Sample scripts for model testing and interaction.
- `report.pdf`: Detailed report of the experiment findings and methodology.

## Data Preparation

### Data Preparation

The experiment leverages diverse datasets to enrich the model's understanding of language. The data preparation phase involves several key steps:

- **Dataset Collection**: Data is collected from various sources, including historical news articles, encyclopedic entries related to Jin Yong and his works, and a vast amount of Q&A data from Zhihu. This diverse dataset ensures the model is exposed to a wide range of language uses and styles.

- **Data Preprocessing**: The collected data undergoes preprocessing to fit the model's input requirements. This includes converting data into a suitable format (e.g., JSONL), tokenizing text using a pre-defined vocabulary, and organizing the data into structured input-target pairs for training.

- **Fine-tuning Data Preparation**: For fine-tuning, the data is further processed to focus on specific domains or tasks, such as question-answering related to Jin Yong's novels. This involves selecting relevant Q&A pairs, formatting them for the model, and splitting the data into training and validation sets.

- **Data Batching and Masking**: During training, data is batched and masked appropriately to support efficient and effective learning. For fine-tuning tasks, specific masking strategies are applied to focus the model's learning on generating accurate answers to the provided questions.

## Model Architecture

The MiniGLM model is designed to be a compact yet powerful transformer-based neural network tailored for understanding and generating text. It draws inspiration from its larger predecessors like GPT and BERT but aims to provide a more accessible option for experiments and applications requiring fewer computational resources. The core components of the MiniGLM architecture include:

- **Embedding Layers**: MiniGLM utilizes token embedding (`wte`) and position embedding (`wpe`) layers to encode input tokens and their positions within a sequence, respectively. These embeddings capture the semantic meaning of words and their positional context, crucial for generating coherent text.

- **Transformer Blocks**: The model comprises multiple transformer blocks, each consisting of a layer normalization (`LayerNorm`), a causal self-attention mechanism (`CausalSelfAttention`), and a multilayer perceptron (`MLP`). These blocks are responsible for processing the input embeddings to model complex relationships between tokens.

  - **Causal Self-Attention**: This component allows each token to attend to previous tokens in the sequence, enabling the model to generate text one token at a time while considering the context provided by the preceding tokens.
  
  - **MLP**: Following the attention mechanism, the MLP further processes the information to capture deeper linguistic features and relationships.

- **Output Projection**: The final layer normalization is followed by a linear projection (`lm_head`) that maps the transformer output to the vocabulary space, predicting the likelihood of each token in the vocabulary as the next token in the sequence.

- **Configuration Flexibility**: The model is configurable, allowing adjustments to the number of layers (`n_layer`), attention heads (`n_head`), embedding dimensions (`n_embd`), and other hyperparameters to suit different datasets and computational constraints.

## Evaluations and Findings

- The models exhibit impressive generalization abilities, successfully generating contextually relevant and grammatically accurate responses without significant overfitting.
- Evaluations using perplexity and Rouge-L metrics on a locally prepared Q&A dataset indicate the models' proficiency in both text generation and question answering.
- Notably, the experiment identifies the models' limitations in understanding certain factual aspects and the influence of training dataset quality on model outputs.

## Exploratory LSTM Model

In addition to the Transformer-based MiniGLM models, an exploratory attempt was made using a simpler LSTM architecture. This exploration aimed to compare the effectiveness of LSTM models in text generation tasks against the more complex Transformer models.

## Conclusion

The MiniGLM model training experiment showcases the potential of leveraging extensive and diverse datasets for training language models capable of understanding and generating Chinese text. Through innovative approaches to data preparation, model training, and evaluation, this experiment contributes valuable insights and tools for further research in the field of natural language processing and text generation.