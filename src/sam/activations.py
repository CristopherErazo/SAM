import math
import torch

# Define activation functions and their derivatives
def He1(x): return x
def He2(x): return (x**2 - 1) / math.sqrt(2)
def He3(x): return (x**3 - 3 * x) / math.sqrt(6)
def He4(x): return (x**4 - 6 * x**2 + 3) / math.sqrt(24)
def He5(x): return (x**5 - 10 * x**3 + 15 * x) / math.sqrt(120)
def relu(x): return torch.relu(x)
def tanh(x): return torch.tanh(x)
def sigmoid(x): return torch.sigmoid(x)


# Dictionary to map function names to their implementations
activations = {
    "tanh": tanh,
    "relu": relu,
    "He1": He1,
    "He2": He2,
    "He3": He3,
    "He4": He4,
    "He5": He5,
    "sigmoid": sigmoid
}

def _get_base_activation(name):
    """
    Retrieve the base activation function and its derivative by name.
    """
    if name not in activations:
        raise ValueError(f"Activation '{name}' not recognized. Available: {list(activations.keys())}")
    return activations[name]


def get_activation(spec):
    """
    Return f for spec which can be:
      - a string: one of the keys in `activations`
      - a dict: {name: coeff, ...} forming a linear combination
      - a list/tuple:
           * list of names -> each with coeff=1
           * list of (name, coeff) pairs

    Example usages:
      get_activation("He3")
      get_activation({"He2": 0.5, "He3": 0.5})
      get_activation([("He2",0.3), ("He3",0.7)])
      get_activation(["He2","He3"])  # equals He2 + He3
    """
    # string: simple lookup
    if isinstance(spec, str):
        return _get_base_activation(spec)

    # dict: name -> coeff
    if isinstance(spec, dict):
        items = spec.items()

    # list/tuple: either names or (name, coeff) pairs
    elif isinstance(spec, (list, tuple)):
        if all(isinstance(el, str) for el in spec):
            items = [(name, 1.0) for name in spec]
        else:
            items = []
            for el in spec:
                if not (isinstance(el, (list, tuple)) and len(el) == 2):
                    raise ValueError("List elements must be either names or (name, coeff) pairs.")
                name, coeff = el
                items.append((name, coeff))
    else:
        raise TypeError("spec must be a string, dict, or list/tuple.")

    # build combined function and derivative
    def combined_f(x):
        out = None
        for name, coeff in items:
            f = _get_base_activation(name)
            term = coeff * f(x)
            out = term if out is None else out + term
        return out
    return combined_f