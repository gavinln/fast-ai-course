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

## VSCode setup with Python extension

Use the following entries in `settings.json`

```
{
    "python.defaultInterpreterPath": "/home/gavin/.cache/pypoetry/virtualenvs/fast-ai-course-DsjtVAmj-py3.8/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["-l79"]
}
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

Bagging process

1. Randomly choose a subset of the rows of data
2. Train a model using this subset
3. Save the model and return to 1 training more models
4. Use all models to predict and take the average of the predictions

##### Out-of-bag error

As each round of the bagging process uses a random subset of rows for training,
the remaining rows can be used for validation and the prediction error on these
rows is called the out-of-bag (OOB) error. This reduces the need for a separate
validation set.

##### Feature importance calculation

The feature importance algorithm loops through each tree, and then recursively
explores each branch. At each branch, it looks to see what feature was used for
that split, and how much the model improves as a result of that split. The
improvement (weighted by the number of rows in that group) is added to the
importance score for that feature. This is summed across all branches of all
trees, and finally the scores are normalized such that they add to 1

##### Finding out-of-domain data

Out-of-domain data is data which is in the validation set but not in the training set. A model cannot provide good predictions on data different than what it has seen in the past (in the training set).

To find out-of-domain data use a random forest model to predict a target that
is `is_valid` which is 0 for training data and 1 for validation data. If there are features that can predict the `is_valid` well, it is likely that those features may have values that are not in both the training and validation set.

#### Gradient boosting

1. Train a small model on your dataset
2. Calculate predictions
3. Subtract predictions from the targets - called residuals
4. Go to step one but instead of target predict residuals
5. Continue until you reach some stopping criterion, such as maximum number of trees

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

##### Create a decision tree

1. Loop through each feature in the data set
2. For each feature loop through each value
3. Split data into two groups based on each value
4. Find the average value of target for each group
5. Compute how close each groups actual value compares to target
6. Pick the split point that gives the best split
7. Treat each of the two groups as separate data sets
8. Recursively continue process until a stopping criterion is reached. For example, stop splitting a group if it has only 20 items in it.

##### Processing dates with decision trees

A feature such as `saleDate` is converted into the following

* saleYear
* saleMonth
* saleWeek
* saleDay
* saleDayofweek
* saleDayofyear
* `saleIs_month_end`
* `saleIs_month_start`
* `saleIs_quarter_end`
* `saleIs_quarter_start`
* `saleIs_year_end`
* `saleIs_year_start`
* saleElapsed
