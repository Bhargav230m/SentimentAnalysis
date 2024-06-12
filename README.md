# Sentiment Analysis Project

## Overview

This project is focused on sentiment analysis, where the goal is to classify text into two categories: positive and negative. The project uses an CNN model implemented with PyTorch. The repository includes the dataset, preprocessing code used to generate the dataset, the trained model files, training script and script to classify text.

## Repository Structure

```
.
├── data
│   └── train_preprocessed.parquet    # The preprocessed dataset file
├── models
│   ├── model.pt                # Trained model file
│   ├── vocabulary.vocab        # Vocabulary file
│   └── hyperparams.hparams     # Hyperparameters file
├── src
│   ├── preprocess.py           # Dataset preprocessing code.
│   ├── train.py                # Training code
│   ├── model.py                # Model definition script
│   └── classify.py             # Script used to classify
│                  
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
└── requirements.txt            # Required Python packages
```

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or higher installed. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset is located in the `data` directory. It consists of a Parquet file with text and corresponding sentiment labels (positive or negative).

### Model Training

The model is defined in `model.py`. The `Classifier` class is a simple neural network with three fully connected layers. You can train the model by running the `train.py` script:

```bash
python src/train.py
```

This script will load the preprocessed data, initialize the model and train it.

### Model Architecture

The model is a simple Feed Forward Neural Network with the following architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # [batch_size, hidden_size]
        x = F.relu(self.fc2(x)) # [hidden_size, hidden_size]
        x = self.fc3(x) # [hidden_size, output_size]

        return F.log_softmax(x, dim=1)
```

### Evaluation

After training, you can evaluate the model's performance on a test set.

### Using the Trained Model

You can use the trained model to predict the sentiment of new text data. Load the model then feed it into the model for prediction. Run the script classify.py

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Acknowledgements

Thank you to all the contributors and the open-source community for providing the tools and resources that made this project possible.
