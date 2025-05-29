# wolike666 PAN25 Generated Plagiarism Detection

This repository contains the code and Docker configuration for the PAN 2025 plagiarism detection task.

## Repository Structure

```text
pan25-generated-plagiarism-detection/
├── run.py                   # Entry-point for Docker and TIRA
├── scripts/
│   └── predict-fast.py      # Main inference script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker build instructions
└── README.md                # This file
```

## Model

A fine-tuned BERT model is hosted on Hugging Face and will be automatically downloaded at runtime:

- **Model:** [jrluo/PlagiarismDetection-bert-base-train10000](https://huggingface.co/jrluo/PlagiarismDetection-bert-base-train10000)

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/wolike666/wolike666-pan25-generated-plagiarism-detection.git
cd wolike666-pan25-generated-plagiarism-detection
pip install -r requirements.txt
```

## Local Usage
Run inference on local data:
```bash
python run.py --input ./example-data/input --output ./example-data/output
```

## Docker
### Build
```bash
docker build -t wolike666/predict-fast .
```

### Run
```bash
docker run --rm \
  -v $(pwd)/example-data/input:/input \
  -v $(pwd)/example-data/output:/output \
  wolike666/predict-fast
```
 
## TIRA Submission

### Image: ```wolike666/predict-fast```

### Command:
```bash
python run.py --input $inputDataset --output $outputDir
```