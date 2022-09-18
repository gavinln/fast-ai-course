# FastAI course and library

* Source code - [Github][10]
* Author - Gavin Noronha - <gavinln@hotmail.com>

[10]: https://github.com/gavinln/fast-ai-course

## About

Learning AI with the FastAI course and library

### FastAI course

https://course.fast.ai/

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

## Lessons

After installing all the software you can run the notebooks in the fastai course

### Lesson 1 - Your first models

* Universal approximation theorem - neural networks
* Update weights using Stochastic Gradient Descent - SGD
* loss is typically used to optimize SGD
* metric is typically defined for human consumption

#### Transfer learning

Using a pretrained model for a task different to what it was originally trained for

#### Fine-tuning

1. Use one epoch to fit just those parts of the model necessary to get the new
   random head to work correctly with your dataset.

2. Use the number of epochs requested when calling the method to fit the entire
   model, updating the weights of the later layers (especially the head) faster
   than the earlier layers.
