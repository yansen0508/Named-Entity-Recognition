# MultiNERD Fine-Tuning Project

## Description
This project involves fine-tuning a language model on the [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17) for Named Entity Recognition (NER) tasks. The focus is on two systems:
- System A: Fine-tuning on the English subset.
- System B: Fine-tuning on a subset of the dataset with specific 5 entity types: `PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)` as well as the `O` tag (all other entities will be set to zero).

## Requirements
To install the required packages, run:
```
pip install -r requirements.txt
```
### Framework Versions
Python: 3.10.12

SpanMarker: 1.5.0

Transformers: 4.35.2

PyTorch: 2.1.0+cu118

Datasets: 2.15.0

Tokenizers: 0.15.0

## Usage
The project has been done on Google Colab (with 1 V100 GPU).

For details, please open `Named Entity Recognition.ipynb` via Colab.

## Dataset
The dataset used is the MultiNERD dataset, which is a multi-lingual NER dataset. 
For System A, the dataset is the English subset.
For System B, the dataset is filtered to include only certain entity types.

## Models
The pre-trained model [tomaarsen/span-marker-mbert-base-multinerd](https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd) used for fine-tuning comes from Hugging Face's Transformers library.

Maximum Sequence Length: 256 tokens

Maximum Entity Length: 8 words

The checkpoints are provided at [models](models/readme.md).

## **System A**

<img src="img/label_A.png" alt="*Model Labels*" width="600"/>

**Training Hyperparameters**

- learning_rate: 5e-05

- train_batch_size: 16

- eval_batch_size: 16

- seed: 42

- gradient_accumulation_steps: 2

- total_train_batch_size: 32

- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08

- lr_scheduler_type: linear

- lr_scheduler_warmup_ratio: 0.1

- num_epochs: 1

<img src="img/training_A.png" alt="*Training Results*" width="800"/>

**Evaluation**

<img src="img/eva_A.png" alt="*Evaluation Results*" width="800"/>

**Framework Versions**

Python: 3.10.12

SpanMarker: 1.5.0

Transformers: 4.35.2

PyTorch: 2.1.0+cu118

Datasets: 2.15.0

Tokenizers: 0.15.0

## **System B**

<img src="img/label_B.png" alt="*Model Labels*" width="600"/>

**Training Hyperparameters**

- learning_rate: 5e-05

- train_batch_size: 16

- eval_batch_size: 16

- seed: 42

- gradient_accumulation_steps: 2

- total_train_batch_size: 32

- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08

- lr_scheduler_type: linear

- lr_scheduler_warmup_ratio: 0.1

- num_epochs: 1

<img src="img/training_B.png" alt="*Training Results*" width="800"/>

**Evaluation**

<img src="img/eva_B.png" alt="*Evaluation Results*" width="800"/>


**Framework Versions**

Python: 3.10.12

SpanMarker: 1.5.0

Transformers: 4.35.2

PyTorch: 2.1.0+cu118

Datasets: 2.15.0

# Conclusion

In this project (NER task on MultiNERD English subset), I observed distinct outcomes for System A and System B, each fine-tuned using the same pre-trained model. 
System A, which utilized the full scope of the English subset, demonstrated slightly performance than the pretrained model. 

Conversely, System B, focused on a subset of entities (PERSON, ORGANIZATION, LOCATION, DISEASES, ANIMAL), showed higher precision and recall in these specific categories. This specialization allowed for more targeted learning, leading to improved accuracy on these entities but at the expense of a narrower overall understanding.

A limitation of the selected pre-trained model was its generalization capability when adapted to specific subsets of data. While it performed well on common entity types like LOCATION and PERSON, its effectiveness varied on rarer categories like ANIMAL and DISEASES, indicating a potential need for more specialized or further pre-training on diverse datasets. This variation underscores the trade-off between specialized and generalized models, where the former excels in specific tasks but may lack broader applicability, a crucial factor in real-world applications.

Tokenizers: 0.15.0
