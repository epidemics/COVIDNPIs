from .pymc3_models.epi_params import EpidemiologicalParameters
from .pymc3_models.models import DefaultModel

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
print('Set Theano Environmental Variables for Parallelisation')
