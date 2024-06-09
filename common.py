
# Standard Library Imports
import os  # Operating system interface
import shutil  # High-level file operations
from copy import deepcopy  # Deep copy operations
import pickle  # Object serialization
import random  # Generate random numbers
from datetime import datetime  # Date and time manipulation

# Data Manipulation and Analysis
import pandas as pd  # Data analysis and manipulation
import numpy as np  # Numerical operations

# Machine Learning
from sklearn.model_selection import train_test_split  # Splitting data into training and test sets
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, recall_score, \
                            precision_score, f1_score, roc_curve, auc, confusion_matrix, \
                            balanced_accuracy_score, matthews_corrcoef, precision_recall_curve  # Evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight # Compute class weights

# Image Processing
from PIL import Image  # Image processing
import cv2  # Computer vision

# Deep Learning
import torch  # PyTorch library
from torch import nn # Neural network modules
from torch.nn import CrossEntropyLoss  # Cross-entropy loss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # Data handling in PyTorch
import torchvision.transforms as transforms  # Transformations for image data
from torchvision.utils import make_grid  # Utilities for image processing
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Reduce learning rate on plateau
from transformers import ViTFeatureExtractor, ViTForImageClassification  # Vision Transformer model

# Visualization
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical data visualization

# Experiment Tracking
import wandb  # Weights & Biases for experiment tracking
