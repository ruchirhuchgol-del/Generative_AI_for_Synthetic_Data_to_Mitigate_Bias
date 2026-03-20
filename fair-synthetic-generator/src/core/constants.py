# Constants for the fair-synthetic-generator project
from enum import Enum

# Version
__version__ = "0.1.0"

# Seeds
DEFAULT_SEED = 42

# Data Modalities
TABULAR = "tabular"
TEXT = "text"
IMAGE = "image"
MULTIMODAL = "multimodal"

class ModalityType(Enum):
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

MODALITY_TYPES = ["tabular", "text", "image", "multimodal"]

# Fairness
class FairnessParadigm(Enum):
    GROUP = "group"
    INDIVIDUAL = "individual"
    COUNTERFACTUAL = "counterfactual"

FAIRNESS_PARADIGMS = ["group", "individual", "counterfactual"]

class GroupFairnessMetric(Enum):
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"

GROUP_FAIRNESS_METRICS = ["demographic_parity", "equalized_odds", "disparate_impact"]

class IndividualFairnessMetric(Enum):
    LIPSCHITZ = "lipschitz"
    CONSISTENCY = "consistency"

INDIVIDUAL_FAIRNESS_METRICS = ["lipschitz", "consistency"]

class CounterfactualMetric(Enum):
    FLIP_RATE = "flip_rate"

COUNTERFACTUAL_METRICS = ["flip_rate"]

# Components
class EncoderType(Enum):
    MLP = "mlp"
    CNN = "cnn"
    TRANSFORMER = "transformer"

class DecoderType(Enum):
    MLP = "mlp"
    CNN = "cnn"
    TRANSFORMER = "transformer"

class GeneratorType(Enum):
    VAE = "vae"
    GAN = "gan"
    DIFFUSION = "diffusion"

class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

class SensitiveAttributeType(Enum):
    BINARY = "binary"
    CATEGORICAL = "categorical"

# Loss & Metrics
class LossType(Enum):
    RECONSTRUCTION = "reconstruction"
    KLD = "kld"
    ADVERSARIAL = "adversarial"
    FAIRNESS = "fairness"

LOSS_TYPES = ["reconstruction", "kld", "adversarial", "fairness"]

class MetricType(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    FAIRNESS = "fairness"

METRIC_TYPES = ["accuracy", "f1", "fairness"]

# Dictionaries
ACTIVATION_FUNCTIONS = {"relu": "ReLU", "leaky_relu": "LeakyReLU", "tanh": "Tanh", "sigmoid": "Sigmoid"}
INITIALIZATION_METHODS = {"normal": "normal_", "xavier": "xavier_normal_", "kaiming": "kaiming_normal_"}
OPTIMIZER_TYPES = {"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop"}
LR_SCHEDULER_TYPES = {"step": "StepLR", "plateau": "ReduceLROnPlateau", "cosine": "CosineAnnealingLR"}

PRIVACY_DEFAULTS = {"epsilon": 1.0, "delta": 1e-5}
FAIRNESS_THRESHOLDS = {"demographic_parity": 0.1, "equalized_odds": 0.1}
DIFFUSION_DEFAULTS = {"steps": 1000, "beta_start": 1e-4, "beta_end": 0.02}

# File extensions
CHECKPOINT_EXTENSIONS = [".pt", ".pth", ".ckpt"]
DATA_EXTENSIONS = [".csv", ".npy", ".pt"]

# Numerical
EPS = 1e-8
INFINITY = float("inf")
NEG_INFINITY = float("-inf")

# Training
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3

# Registries (populated by respective modules)
GENERATOR_REGISTRY = {}
ENCODER_REGISTRY = {}
DECODER_REGISTRY = {}
DISCRIMINATOR_REGISTRY = {}
