from .pymc3_models.epi_params import EpidemiologicalParameters
from .pymc3_models.models import DefaultModel
from .preprocessing.data_preprocessor import preprocess_data

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
print('Set Theano Environmental Variables for Parallelisation')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# very strange, but this fixes region holdout bug issue with region 'MA'
sys.setrecursionlimit(10000)