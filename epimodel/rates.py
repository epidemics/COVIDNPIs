import datetime
import enum
import logging
import re
import weakref
from collections import OrderedDict
from pathlib import Path

import dateutil
import numpy as np
import pandas as pd
import unidecode

from .utils import normalize_name

log = logging.getLogger(__name__)

class Rates:
    def __init__(self, rds, code):
        self._rds = weakref.ref(rds)
        self._code = code

    def get_display_name(self):
        return self.Code

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._code}>"

    def __setattr__(self, name, val):
        """Forbid direct writes to anything but _variables."""
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise AttributeError(
                f"Setting attribute {name} on {self!r} not allowed (use ratesds.data directly)."
            )

class RatesDataset:
    """
    A set of rates
    """

    COLUMN_TYPES = OrderedDict(
        Hospitalization="f4",
        Critical="f4",
        CaseFatalityRate="f4"
    )

    def __init__(self):
        """
        Creates an empty region set. Use `RegionDataset.load` to create from CSV.
        """
        # Main DataFrame (empty)
        self.data = pd.DataFrame(
            index=pd.Index([], name="CodeM49", dtype=pd.StringDtype())
        )
        for name, dtype in self.COLUMN_TYPES.items():
            self.data[name] = pd.Series(dtype=dtype, name=name)
        # code: [Region, Region, ...]
        self._code_index = {}

    @classmethod
    def load(cls, *paths):
        """
        Create a RegionDataset and its Regions from the given CSV.

        Optionally also loads other CSVs with additional regions (e.g. GLEAM regions)
        """
        s = cls()
        cols = dict(cls.COLUMN_TYPES)
        for path in paths:
            log.debug("Loading regions from {path!r} ...")
            data = pd.read_csv(
                path,
                dtype=cols,
                index_col="CodeM49",
                na_values=[""],
                keep_default_na=False,
            )
            # Convert Level to enum
            s.data = s.data.append(data)
        s.data.sort_index()
        s._rebuild_index()
        return s

    @property
    def rates(self):
        """Iterator over all regions."""
        return self._code_index.values()

    def __getitem__(self, code):
        """
        Returns the Rates corresponding to code, or raise KeyError.
        """
        return self._code_index[int(code)]

    def get(self, code, default=None):
        """
        Returns the Rates corresponding to code, or `default`.
        """
        try:
            return self[code]
        except KeyError:
            return default
        except ValueError:
            return default

    def _rebuild_index(self):
        """Rebuilds the indexes and ALL Rates objects!"""
        self._name_index = {}
        self._code_index = {}
        self.data = self.data.sort_index()

        # Create Regions
        for ri in self.data.index:
            rat = Rates(self, ri)
            assert ri not in self._code_index
            self._code_index[ri] = rat
