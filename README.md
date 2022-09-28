# FastAI course and library

* Source code - [Github][10]
* Author - Gavin Noronha - <gavinln@hotmail.com>

[10]: https://github.com/gavinln/fast-ai-course

## About

Learning AI with the FastAI course and library

### FastAI course

#### Clone the course repository

1. Change to the ./course directory

2. Clone the repository

```
git clone https://github.com/fastai/fastbook
```

## Run FastAI on AWS

1. Setup an AWS GPU instance as in this document

```
d:/ws/aws-vm/doc/ec2-spot-gpu-setup.md
```

2. Connect to the instance

```
ssh -L 8888:localhost:8888 $INSTANCE_ID
```

3. Start tmux

```
tmux
```

4. Install poetry

```
pipx install poetry
```

5. List installed Python utilities

```
pipx list
```

6. Clone the `fast-ai` project

```
git clone https://github.com/gavinln/fast-ai-course
```

7.  Change to the project root directory

```
cd fast-ai-course
```

8. Clone the `fast-ai` course

```
git clone https://github.com/fastai/fastbook
```

9. Install Python libraries

```
poetry install
```

10. Run Jupyter lab

```
make jupyter
```

11. Access jupyter at http://127.0.0.1:8888/

### Install software manually

Not needed if software installed using poetry

1. Install pytorch

```
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Install fastai

```
pip install fastai
```

## Lessons - course 2020

After installing all the software you can run the notebooks in the fastai course

* Course page:  https://course20.fast.ai/
* Course videos: https://course20.fast.ai/videos/

### Lesson 1 - Your first models

* Universal approximation theorem - neural networks
* Update weights using Stochastic Gradient Descent - SGD
* Loss is typically used to optimize SGD
* Metric is typically defined for human consumption
* Epoch - process all data in dataset at least once

#### Transfer learning

Using a pretrained model for a task different to what it was originally trained for

#### Fine-tuning

1. Use one epoch to fit just those parts of the model necessary to get the new
   random head to work correctly with your dataset.

2. Use the number of epochs requested when calling the method to fit the entire
   model, updating the weights of the later layers (especially the head) faster
   than the earlier layers.

### Lesson 2 - Your first models - continued

* Classification - predict one or more discrete possibilities
* Regression - predict one or more numeric values

#### What is deep leaning good for?

* Vision: detection, classification
* Text: classification, translation
* Tabular: high cardinality
* Recommendation systems: predictions (not recommendations)
* Multi-modal: e.g. text & images, captioning

#### Pretrained models

https://modelzoo.co/

#### Gaining business value

Identify and manage constraints in each of these areas

1. Strategy: sources of value, levers
2. Data: availability, suitability
3. Analytics: predictions, insights
4. Implementations: IT, human capital
5. Maintenance: environmental changes

Determine costs of scenario

1. Relationship exists - act as if it exists
2. Relationship exists - act as if it does not exist
3. Relationship does not exist - act as if it exists
4. Relationship does not exist - act as if it does not exist

#### DataBlock api

```python
data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # (x-type, y-type)
    get_items=list_of_items_function,
    splitter=RandomSplitter(valid_pct=0.3, seed=42),  # validation %
    get_y=label_function,
    item_tfms=Resize(128))  # item transformation
```

#### Data issues

* Out of domain data
* Domain shift - data changes over time


### Lesson 4

* mini-batch - few inputs & labels randomly selected
* forward pass - applying the model to inputs and compute the prediction
* loss - value that represents the performance of the model
* gradient - derivative of the loss with respect to model parameters
* backward pass - computing gradients of the loss with respect to parameters
* gradient descent - taking a step in the directio opposite to the gradient
* learning rate - size of step when applying SGD

### Lesson 5

* discrimnative learning rates - different learning rates for different layers

### Lesson 6

#### Collaborative filtering

* Has users and items
* Latent factors exits that explains users preferences for items

### Lesson 7

#### Random forests - bagging

* Random forest cannot extrapolate outside the range of the training data
* Create random forest model to predict whether in training or validation set

Bagging models are less likely to overfit as they average predictions over
trees.

#### Gradient boosting

1. Train a small model on your dataset
2. Calculate predictions
3. Subtract predictions from the targets - called residuals
4. Go to step one predicting residuals

Predictions are made up from the sum of all trees. Boosting models are more
likely to overfit.

#### Embeddings

Embeddings created using neural networks can be used for random forests and gradient boosting methods.

1. Embeddings define a continuous notion of distance between categorical variables
2. Weights & activations are continuous and gradient descent works with continuous variables.
3. Continuous embeddings can be easily concatenated to define an input
4. Embeddings are more compact than one-hot encoding

#### Decision trees

Decision trees and neural networks have similar performance for tabular data.
But ensembles of decision trees are more convenient because:

1. Train faster
2. Easier to interpret
3. Do not require special GPU hardware
4. Less hyperparameter tuning
5. Better tooling and documentation as they have a longer history

Exception to using ensembles of decision trees is when the following features
exist in the data set.

1. High cardinality categorical features
2. Features which can be better interpreted by a neural network such as plain text
