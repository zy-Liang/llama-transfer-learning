# Fine-Tuned LLaMA Model for Medical Question Answering

This repository contains Python scripts for training and benchmarking a fine-tuned LLaMA (Long-Short Term Memory Attention) model on various medical question answering datasets.

## Getting Started

### Prerequisites

Before running the scripts, make sure you have the following dependencies installed:

- Python 3.10+
- PyTorch
- [transformers](https://huggingface.co/transformers/)
- [sentencepiece](https://github.com/google/sentencepiece)
- [accelerate](https://huggingface.co/docs/accelerate/)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/)
- [datasets](https://huggingface.co/docs/datasets/)
- [peft](https://huggingface.co/docs/peft/)
- [scipy](https://www.scipy.org/)
- [trl](https://huggingface.co/docs/trl/)
- ...

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
pip install -r requirements.txt
```
### Datasets

This project supports multiple medical question answering datasets:

- BigBio Medical QA
- [PubMed QA](https://huggingface.co/datasets/pubmed_qa)
- [MedMCQA](https://huggingface.co/datasets/medmcqa)
- [MedALPaCA Medical Meadow MedQA](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)

Ensure to download or reference the datasets appropriately.

## Fine-Tuning the Model

To fine-tune the LLaMA model on a specific medical QA dataset, use the `fine_tune_lora.py` script. Adjust the script parameters as needed, and run:

```bash
python fine_tune_lora.py
```

## Benchmarking the Fine-Tuned Model

To evaluate the accuracy of the fine-tuned model, utilize the `benchmark_llama2.py` script. Ensure the script is configured correctly, and run:

```bash
python benchmark_llama2.py
```

### Results
The results, including accuracy and other relevant metrics, will be displayed in the console.