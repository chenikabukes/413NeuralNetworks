# Ribonanza RNA Folding

## Project Overview

### Problem Definition
The challenge is to predict RNA chemical reactivity, a crucial step for identifying RNA-based drug targets. This problem is significant for advancing medical research, especially in areas where traditional protein-level drug targeting is ineffective.

### Dataset
The project utilizes a dataset from Stanford, featuring 1,118,513 RNA sequences and their reactivities. The training data consists of 821,840 sequence profiles, focusing on RNA sequences and their chemical reactivity measures. The test set contains sequences of varying lengths to evaluate the model's generalization capabilities.

[Stanford Ribonanza RNA Folding Dataset](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)

### Prior Work
- **Nucleic Transformer:** Utilizes 1D convolutions and self-attention for DNA sequence classification. [(He et al., 2023)](https://www.nature.com/articles/s41598-023-00000-1)
- **HyenaDNA:** substitutes attention operation with a convolutional layer, called Hyena operators, which allows it to sub-quadratically in sequence length, allowing it to train up to 160x faster than Transformers for DNA sequence generation tasks. [(Nguyen et al., 2023)](https://arxiv.org/abs/2306.15794)

### Proposed Models and Approach
We plan to compare three models, including a novel hybrid CNN-Transformer model, against a baseline provided by Stanford. The hybrid model is designed to capture both local and global sequence dependencies, potentially offering superior prediction accuracy.

1. **Baseline Model:** RNA Starter by Stanford.
3. **Proposed Model 1:** CNN with 1D convolutions + Transformers.
2. **Proposed Model 2:** CNN with Hyena operator.

#### High-Level Intuition
- The baseline model serves as a starting point with Stanford's initial setup.
- Proposed Model 1 combines CNNs for local pattern recognition and transformers for global context, aiming to deliver the most comprehensive analysis.
- Proposed Model 2 aims to leverage the computation efficiency of the Hyena operator to replace transformers while still utilizing long-range dependices of RNA sequences. 

### Ethical Considerations
The dataset does not contain personal identifiers, minimizing ethical concerns. However, the project underscores the necessity of laboratory validation for any computational predictions, ensuring responsible use of the technology in medical research.

## Conclusion
This project aims to bridge a significant gap in RNA-based drug discovery, leveraging advanced machine learning techniques to predict RNA chemical reactivity. By comparing different models, including a novel hybrid approach, we hope to contribute valuable insights to the field of computational biology.

## Getting Started
To get started, first download the data using the following sequence
```
(Insert how to download data)
```

To run the baseline model, execute the following command:
```
srun -p csc413 --gres gpu python3 -u fastai_script.py --model 1
```
Include the arg `--wandb` to log the run. 

Choose a model through the `--model` arg:
1. Basline Transformer
2. Single Layer CNN + Transformer
3. Multilayer CNN + Transformer
4. MLP + Hyena operator
5. CNN + Hyena operator

Additional flags can be seen in `fastai_script.py`.