# Neural Networks From Scratch

Code written while following Andrej Karpathy's "Neural Networks: Zero to Hero" YouTube playlist.

Three things are built here, step by step, in plain Python and PyTorch:

- **micrograd** - a tiny engine that computes gradients by hand, the foundation of how neural networks learn.
- **makemore** - character-level models that generate new names, growing from a simple counting model up to a WaveNet-style network.
- **GPT** - a small transformer that generates text one character at a time.

Each file focuses on one idea, so any single file can be opened and understood on its own.


This is a learning project meant to understand how neural networks work internally, not production-ready code.

## What's actually learned here

### micrograd - how learning works underneath
- **Automatic differentiation**: every math operation (add, multiply, tanh, etc.) remembers how it was built, so gradients can be worked out automatically by walking backwards through the calculations. This is a tiny version of what PyTorch does internally.
  - Uses the **chain rule**: `dL/dx = dL/dy * dy/dx`. Each operation only needs to know its own local derivative (e.g. `d/dx(x*y) = y`, `d/dx tanh(x) = 1 - tanh(x)^2`) and multiplies it by the gradient flowing in from above.
- **Neurons, layers, networks**: built on top of that gradient engine - starting from a single neuron, then a layer of neurons, then a small multi-layer network trained with gradient descent.
  - A neuron computes `output = activation(w1*x1 + w2*x2 + ... + b)`. Training just repeats: compute the loss, get gradients via the chain rule, then nudge every weight a little in the direction that lowers the loss (`w = w - learning_rate * dL/dw`).

### makemore - building a name generator step by step
- **Counting vs. learning**: the first model just counts how often one letter follows another and turns those counts into probabilities. It's then rebuilt as a tiny neural network trained with gradient descent, to show a trained model can learn the same thing a simple count table already knows.
  - Probability of a next letter: `P(next | current) = count(current, next) / total count(current, *)`. Quality is measured with **negative log-likelihood**: `loss = -average(log(P(correct letter)))` - lower is better, and it's the same loss used almost everywhere in this project.
- **Multi-layer perceptron (MLP)**: instead of looking at only the previous letter, the model looks at several previous letters at once, using learned "embeddings" (vectors that represent each letter) to do it.
  - The network's raw outputs (**logits**) are turned into probabilities with **softmax**: `P(i) = e^(logit_i) / sum(e^(logit_j))`. Softmax + negative log-likelihood together are called **cross-entropy loss**.
- **Weight initialization problems**: shows what goes wrong when a network's starting weights are too large or careless - the loss starts off way too high, and neurons get "stuck" (saturated) and stop learning. Fixing how weights are initialized (Kaiming initialization) solves both problems.
  - Kaiming init: `std = gain / sqrt(fan_in)`, where `fan_in` is the number of inputs to a layer and `gain` depends on the activation function (`5/3` for tanh). This keeps values from shrinking or exploding as they pass through many layers.
- **Batch Normalization**: keeps the numbers flowing through the network in a healthy range, making training much more stable. It's explored in different forms here - written by hand directly inside the training loop, and as a reusable building block compared against a plain network without it.
  - Formula: `x_normalized = (x - mean) / sqrt(variance + tiny_number)`, then `output = gain * x_normalized + bias`. The mean and variance are computed per batch during training, and the `gain`/`bias` are learned so the network can undo the normalization if it needs to.
- **Manual backpropagation ("backprop ninja")**: instead of letting PyTorch calculate gradients automatically, the gradients for the loss function and for batch normalization are worked out by hand and used to actually train a model - to see exactly what autograd is doing under the hood.
  - Example: the whole softmax + cross-entropy gradient simplifies to just `probabilities - correct_answer` (as a one-hot vector), divided by batch size. What looks complicated in the forward pass often collapses into something surprisingly simple once you take the derivative.
- **WaveNet-style architecture**: rather than squashing all previous letters together in one big step, letters are combined two at a time in a tree-like structure, letting the model use a longer history of letters without one huge layer.
  - Each level halves the sequence length and doubles the feature size (`(B, T, C) -> (B, T/2, C*2)`), so information from many letters back gets merged in gradually instead of all at once.

### GPT - building a small text generator
- **Bigram baseline**: the simplest possible model, predicting the next character using only the current one, used as a starting point to compare against.
- **Self-attention**: lets each character "look back" at earlier characters in the text and decide which ones matter for predicting what comes next.
  - Formula: `Attention(Q, K, V) = softmax(Q x K^T / sqrt(head_size)) x V`. Every character produces a **query** (what it's looking for), a **key** (what it offers), and a **value** (the actual information); dividing by `sqrt(head_size)` just keeps the numbers from getting too extreme before the softmax.
  - A triangular **mask** blocks each character from seeing future characters, so the model can only use what came before when predicting what's next.
- **Transformer blocks**: multiple attention heads plus feed-forward layers are combined and stacked to build a small GPT-style model that generates text one character at a time.
  - Each block adds its output back onto its input (`x = x + sublayer(x)`, called a **residual connection**), which makes it much easier to train many stacked blocks without gradients fading away.
