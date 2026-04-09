from dataclasses import dataclass , field
from typing import Optional


# # Meta parameters
# dropout=0.0 # Dropout rate
# n_prints=60 # Number of prints during training
# steps=600 # Number of training epochs
# experiment_name='dual_task_new' # Name of the experiment for saving results
# n_prints_model=6 # Number of times to save model checkpoints during training.
# print_scale='linear' # Scale for printing steps: log or linear
# init='random' # initialization method: planted or random
# momentum=0.9 # Momentum for SGD optimizer
# weight_decay=0.0 # Weight decay for optimizers

# # Fix parameters
# vocab_size=64 # Vocabulary size
# seq_len=256  # Sequence length
# d_model=128 # Model dimension
# batch_size=64 # Batch size
# opt='adam' # Optimizer
# test_size=200 # Number of samples in the test set

# # Variable parameters
# K=20 # Number of trigger tokens
# lr=0.008 # Learning rate
# b_type='spiked' # P_b distribution type: dirichlet or spiked
# u_type='zipf' # P_u distribution type: dirichlet or zipf (only used if b_type is spiked)
# alpha=1 # Dirichlet concentration parameter or exponent for the Zipf's law
# beta=0.9 # Beta parameter for spiked bigram distribution (only used if b_type is spiked)
# fix_trig='True' # Whether to fix the trigger tokens across all experiments.
# trig_type='freq' # Whether the trigger tokens should be the most freq, rare or random according to P_u. Only used if fix_trig is True.

# # Configurations to loop over
# configurations=(
#     'dirichlet 15'
#     'spiked 20'
# )

@dataclass
class ModelArgs:
    vocab_size: int = 64  # Vocabulary size
    seq_len: int = 256 # Sequence length
    d_model: int = 128 # Model dimension
    dropout: float = 0.0 # Dropout rate
    lin_attn: bool = False # Whether to use linear attention or not
    # path: str = "full" # Path to follow (options are "full", "induction" and "bigram")

@dataclass
class DataArgs:
    b_type: str = 'dirichlet' # P_b distribution type: dirichlet or spiked
    alpha_d: float = 0.1 # Dirichlet concentration parameter for bigram distribution (only used if b_type is dirichlet or u_type is dirichlet)
    alpha_z: Optional[float] = 1.0 # Exponent for the Zipf distribution used to generate the unigram distribution P_u if b_type is 'spiked' and u_type is 'zipf'
    u_type: Optional[str] = 'dirichlet' # P_u distribution type: dirichlet or zipf (only used if b_type is spiked)
    beta: Optional[float] = 0.8 # Beta parameter for spiked bigram distribution (only used if b_type is spiked)
    fix_trig: bool = True # Whether to fix the trigger tokens or not
    trig_type: Optional[str] = 'freq' # Type of fixed trigger tokens if fix_trig is True (options are 'freq', 'rare' and 'rand')
    batch_size: int = 64 # Batch size for training
    test_size: int = 200 # Number of samples in the test set
    K : int = 15 # Number of trigger tokens

@dataclass
class OptimArgs:
    lr: float = 0.01
    opt: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0

@dataclass 
class ExtraArgs:
    total_steps: int = 1000 # Number of training steps
    n_prints: int = 100 # Number of times to print during training.
    n_prints_model: int = 10 # Number of times to save model checkpoints during training.
    print_scale: str = 'linear' # Scale for printing steps: log or linear
    experiment_name: str = 'tmp' # Name of the experiment for saving results
    file_name: str = 'results' # Name of the file for saving results
    path: str = "full" # Path to follow (options are "full", "induction" and "bigram")

@dataclass
class TrainerArgs:
    model_args: ModelArgs = field(default_factory=ModelArgs)
    optim_args: OptimArgs = field(default_factory=OptimArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    extra_args: ExtraArgs = field(default_factory=ExtraArgs)
