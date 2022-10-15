import torch
import numpy as np
import pandas as pd 
from pathlib import Path

# Direction
output_path = '/Users/HPhuc/Practice/12. classification/vinbigdata/output/'

csv_path = Path(output_path, 'csv')

log_path = Path(output_path, 'logs')

config_path = Path(output_path, 'configs')

cpoints_path = Path(output_path, 'checkpoints')


size = 256

bz = 8

workers = 0