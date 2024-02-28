# RNA Sequencing Model

## Project Overview

### Problem Definition
The challenge is to predict RNA chemical reactivity, a crucial step for identifying RNA-based drug targets. This problem is significant for advancing medical research, especially in areas where traditional protein-level drug targeting is ineffective.

### Dataset
The project utilizes a dataset from Stanford, featuring 1,118,513 RNA sequences and their reactivities. The training data consists of 821,840 sequence profiles, focusing on RNA sequences and their chemical reactivity measures. The test set contains sequences of varying lengths to evaluate the model's generalization capabilities.

[Stanford Ribonanza RNA Folding Dataset](https://www.kaggle.com/datasets/ribonanza/rna-folding)

### Prior Work
- **Nucleic Transformer:** Utilizes 1D convolutions and self-attention for DNA sequence classification. [(He et al., 2023)](https://www.nature.com/articles/s41598-023-00000-1)
- **Vision Transformer (ViT):** Applies self-attention to process images, adaptable for RNA sequence prediction. [(Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)
- **GeneViT:** A vision transformer approach for classifying cancerous gene expressions. [(Gokhale et al., 2023)](https://www.sciencedirect.com/science/article/pii/S0957417421005565)
- **Compact Transformers:** Offers a streamlined version of ViT, reducing parameter count while maintaining performance. [(Hassani et al., 2022)](https://arxiv.org/abs/2104.05704)

### Proposed Models and Approach
We plan to compare three models, including a novel hybrid CNN-Transformer model, against a baseline provided by Stanford. The hybrid model is designed to capture both local and global sequence dependencies, potentially offering superior prediction accuracy.

1. **Baseline Model:** RNA Starter by Stanford.
2. **Proposed Model 1:** Vision Transformers (ViTs).
3. **Proposed Model 2:** CNN with 1D convolutions + Transformers.

![CNN-Transformer Model Diagram](https://example.com/cnn-transformer-diagram.png)

*Figure 1. CNN-Transformer Model Diagram*

#### High-Level Intuition
- The baseline model serves as a starting point with Stanford's initial setup.
- Proposed Model 1 (ViT) processes sequences in patches, focusing on long-range interactions.
- Proposed Model 2 combines CNNs for local pattern recognition and transformers for global context, aiming to deliver the most comprehensive analysis.

### Ethical Considerations
The dataset does not contain personal identifiers, minimizing ethical concerns. However, the project underscores the necessity of laboratory validation for any computational predictions, ensuring responsible use of the technology in medical research.

## Conclusion
This project aims to bridge a significant gap in RNA-based drug discovery, leveraging advanced machine learning techniques to predict RNA chemical reactivity. By comparing different models, including a novel hybrid approach, we hope to contribute valuable insights to the field of computational biology.
