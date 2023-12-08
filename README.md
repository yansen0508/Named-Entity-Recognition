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

To review their results and training script of this pre-trained model, please refer to the official model page: [tomaarsen/span-marker-mbert-base-multinerd](https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd).


Maximum Sequence Length: 256 tokens

Maximum Entity Length: 8 words

### **The fine-tuned checkpoints of System A and System B are provided [here](models/readme.md).**

## **More about System A**

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

- num_epochs: 3

<img src="img/training_A.png" alt="*Training Results*" width="800"/>

**Testing** 

(also available at `models/test_results_A.json`)

<img src="img/eva_A.png" alt="*Evaluation Results*" width="800"/>


## **More about System B**

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

- num_epochs: 3

<img src="img/training_B.png" alt="*Training Results*" width="800"/>

**Testing** 

(also available at `models/test_results_B.json`)

<img src="img/eva_B.png" alt="*Evaluation Results*" width="800"/>


# Conclusion

In this project (NER task on MultiNERD English subset), I observed distinct outcomes for System A and System B, each fine-tuned using the same pre-trained model. 
System A, which utilized the full scope of the English subset, demonstrated slightly performance than the pretrained model. 

```
"distribution train": {
        "O": 4957198, "B-LOC": 117330, "B-PLANT": 14872, "I-PLANT": 4702,
        "B-PER": 125974, "I-PER": 132376, "B-DIS": 17404, "I-DIS": 11608,
        "B-BIO": 280, "B-MEDIA": 12162, "I-MEDIA": 20070, "I-LOC": 48800,
        "B-ORG": 55282, "I-ORG": 71998, "B-TIME": 5080, "B-ANIM": 25472,
        "I-ANIM": 10614, "B-EVE": 5050, "I-EVE": 8406, "I-TIME": 3942,
        "B-CEL": 5370, "I-CEL": 2972, "B-FOOD": 16558, "I-FOOD": 6060,
        "B-VEHI": 808, "I-VEHI": 956, "B-MYTH": 1138, "I-MYTH": 202,
        "B-INST": 758, "I-INST": 726, "I-BIO": 70
    },
"distribution validation": {
        "O": 664072, "B-EVE": 598, "I-EVE": 1076, "B-LOC": 15700,
        "B-MEDIA": 1838, "I-MEDIA": 2818, "B-DIS": 3390, "I-DIS": 2502,
        "B-PER": 15014, "I-PER": 15984, "B-TIME": 644, "I-TIME": 508,
        "I-LOC": 7234, "B-PLANT": 2360, "B-FOOD": 4240, "B-VEHI": 160,
        "I-VEHI": 174, "B-ORG": 5474, "I-ORG": 7824, "I-FOOD": 1656,
        "B-ANIM": 2268, "I-ANIM": 1154, "B-CEL": 188, "B-MYTH": 112,
        "I-MYTH": 18, "B-INST": 68, "I-PLANT": 830, "B-BIO": 28,
        "I-CEL": 38, "I-INST": 44, "I-BIO": 8
    },
"distribution test": {
        "O": 602884, "B-PER": 10530, "I-PER": 11460, "B-EVE": 704,
        "I-EVE": 1216, "B-LOC": 24048, "B-ANIM": 3208, "I-ANIM": 1852,
        "I-LOC": 11926, "B-ORG": 6618, "I-ORG": 9162, "B-FOOD": 1132,
        "B-PLANT": 1788, "I-PLANT": 796, "B-MYTH": 64, "B-DIS": 1518,
        "I-FOOD": 366, "B-MEDIA": 916, "I-MEDIA": 1542, "I-DIS": 1004,
        "B-TIME": 578, "I-TIME": 416, "B-CEL": 82, "B-VEHI": 64,
        "I-VEHI": 66, "B-BIO": 16, "B-INST": 24, "I-INST": 24,
        "I-MYTH": 14, "I-CEL": 32
    }
```

Conversely, System B, focused on a subset of entities (PERSON, ORGANIZATION, LOCATION, DISEASES, ANIMAL), showed higher precision and recall in these specific categories. This specialization allowed for more targeted learning, leading to improved accuracy on these entities but at the expense of a narrower overall understanding.

A limitation of the selected pre-trained model was its generalization capability when adapted to specific subsets of data. While it performed well on common entity types like LOCATION and PERSON, its effectiveness varied on rarer categories like ANIMAL and DISEASES, indicating a potential need for more specialized or further pre-training on diverse datasets. This variation underscores the trade-off between specialized and generalized models, where the former excels in specific tasks but may lack broader applicability, a crucial factor in real-world applications.

Tokenizers: 0.15.0
