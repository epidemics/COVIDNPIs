from uuid import UUID
from collections import namedtuple
import re
from typing import Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm
from epimodel import RegionDataset, Region, Level, algorithms
from .definition import GleamDefinition

try:
    import ergo
except ModuleNotFoundError:
    ergo = None
    # foretold functionality optional
    pass


class ConfigParser:
    """
    encapsulates credentials and logic for loading spreadsheet data and
    formatting it for use by the rest of the scenario classes
    """

    FIELDS = [
        "Region",
        "Value",
        "Parameter",
        "Start date",
        "End date",
        "Type",
        "Class",
    ]
    STR_PARAMS = [
        "name",
    ]

    def __init__(self, rds, foretold_token=None, progress_bar=True):
        self.rds = rds
        self.foretold = ergo.Foretold(foretold_token) if foretold_token else None
        self.progress_bar = progress_bar
        algorithms.estimate_missing_populations(rds)

    def get_config_from_csv(self, csv_file: str):
        return self.get_config(pd.read_csv(csv_file))

    def get_config(self, df: pd.DataFrame):
        df = df.replace({"": None})
        df = df[pd.notnull(df["Parameter"])][self.FIELDS].copy()
        df["Start date"] = pd.to_datetime(df["Start date"])
        df["End date"] = pd.to_datetime(df["End date"])
        df["Region"] = df["Region"].apply(self._get_region)

        is_str_value = df["Parameter"].isin(self.STR_PARAMS)
        df.loc[~is_str_value, "Value"] = self._values_to_float(
            df.loc[~is_str_value, "Value"]
        )
        return df

    def _values_to_float(self, values: pd.Series):
        values = values.copy()
        uuid_filter = values.apply(self._is_uuid)
        values[uuid_filter] = [
            self._get_foretold_mean(uuid)
            for uuid in self._progress_bar(
                values[uuid_filter], desc="fetching Foretold distributions"
            )
        ]
        return values.astype("float")

    @staticmethod
    def _is_uuid(value):
        try:
            UUID(value, version=4)
            return True
        except (ValueError, AttributeError):
            return False

    def _get_foretold_mean(self, uuid: str):
        question_distribution = self.foretold.get_question(uuid)
        # Sample the centers of 100 1%-wide quantiles
        qs = np.arange(0.005, 1.0, 0.01)
        ys = np.array([question_distribution.quantile(q) for q in qs])
        mean = np.sum(ys) / len(qs)
        return mean

    def _get_region(self, code_or_name: str):
        if pd.isnull(code_or_name):
            return None

        # Try code first
        if code_or_name in self.rds:
            region = self.rds[code_or_name]
        else:
            # Try code_or_name. Match Gleam regions first.
            matches = self.rds.find_all_by_name(code_or_name, levels=tuple(Level))
            if not matches:
                raise ValueError(f"No region found for {code_or_name!r}.")
            region = matches[0]

        if pd.isnull(region.GleamID):
            raise ValueError(f"Region {region!r} has no GleamID")
        return region

    def _progress_bar(self, enum, desc=None):
        if self.progress_bar:
            return tqdm(enum, desc=desc)
        return enum


class SimulationSet:
    """
    generates a matrix of different simulations from the config df based
    on the cartesian product of Countermeasure packages X Background
    conditions
    """

    def __init__(self, df: pd.DataFrame):
        self._prepare_ids(df)
        self._set_scenario_values(df)
        self._generate_scenario_definitions()

    def __getitem__(self, classes: tuple):
        return self.definitions[classes]

    def __contains__(self, classes: tuple):
        return classes in self.definitions.index

    def __iter__(self):
        return self.definitions.iteritems()

    def add_to_batch(self, batch):
        batch.set_simulations(
            [
                (def_gen.definition, "PLACEHOLDER", pacakge, background)
                for (package, background), def_gen in self
            ]
        )

    def _prepare_ids(self, df):
        """
        IDs are bit-shifted so every 2-class combo has a unique id when
        the two ids are added. This is then added to the base_id to make
        the resulting id sufficiently large and unique.
        """
        self.base_id = int(pd.Timestamp.utcnow().timestamp() * 1000)
        self.ids = {klass: 1 << i for i, klass in enumerate(df["Class"].unique())}

    def _id_for_class_pair(self, class1: str, class2: str):
        return self.base_id + self.ids[class1] + self.ids[class2]

    def _set_scenario_values(self, df):
        is_package = df.Type == "Countermeasure package"
        is_background = pd.isnull(df.Type) | (df.Type == "Background condition")
        if not df[~is_package & ~is_background].empty:
            bad_types = list(df[~is_package & ~is_background]["Type"].unique())
            raise ValueError("input contains invalid Type values: {bad_types!r}")

        self.package_df = df[is_package]
        self.background_df = df[is_background]

        self.package_classes = self.package_df["Class"].dropna().unique()
        self.background_classes = self.background_df["Class"].dropna().unique()

        # rows with no class are applied to all simulations
        self.package_classless_df = self.package_df[pd.isnull(self.package_df["Class"])]
        self.background_classless_df = self.background_df[
            pd.isnull(self.background_df["Class"])
        ]

    def _generate_scenario_definitions(self):
        index = pd.MultiIndex.from_product(
            [self.package_classes, self.background_classes]
        )
        self.definitions = pd.Series(
            [self._definition_for_class_pair(*pair) for pair in index], index=index
        )

    def _definition_for_class_pair(self, package_class: str, background_class: str):
        p_df = self.package_df[self.package_df["Class"] == package_class]
        b_df = self.background_df[self.background_df["Class"] == background_class]
        return DefinitionGenerator(
            # ensure that package exceptions come before background conditions
            pd.concat(
                [p_df, self.package_classless_df, b_df, self.background_classless_df]
            ),
            id=self._id_for_class_pair(package_class, background_class),
            classes=(package_class, background_class),
        )


class DefinitionGenerator:
    """
    Takes a simulation-specific config DataFrame and translates it into
    a GleamDefinition object.
    """

    GLOBAL_PARAMETERS = {
        "name": "set_name",
        "id": "set_id",
        "duration": "set_duration",
        "number of runs": "set_run_count",
        "airline traffic": "set_airline_traffic",
        "seasonality": "set_seasonality",
        "commuting time": "set_commuting_rate",
    }
    COMPARTMENT_VARIABLES = (
        "beta",
        "epsilon",
        "mu",
        "imu",
    )

    def __init__(self, df: pd.DataFrame, id, classes=None, default_xml=None):
        self.definition = GleamDefinition(default_xml)
        self.definition.set_id(id)

        self._parse_df(df)
        self._set_global_parameters()
        self._set_global_compartment_variables()
        self._set_exceptions()

        if classes is None and "name" not in df.Parameter:
            self.definition.set_default_name()
        else:
            self._set_name_from_classes(classes)

    @property
    def filename(self):
        nonplussed = self.definition.get_name().replace(" + ", "__")
        return "%s.xml" % re.sub(r"\W", "_", nonplussed).strip("_")

    def save_to_dir(self, dir):
        self.definition.save(dir / self.filename)

    def _set_name_from_classes(self, classes=None):
        name = self.definition.get_name() or self.definition.get_timestamp()
        if classes:
            name = f"{name} ({' + '.join(classes)})"
        self.definition.set_name(name)

    def _parse_df(self, df: pd.DataFrame):
        has_exception_fields = pd.notnull(df[["Region", "Start date", "End date"]])
        has_compartment_param = df["Parameter"].isin(self.COMPARTMENT_VARIABLES)

        is_exception = has_compartment_param & has_exception_fields.all(axis=1)
        is_compartment = has_compartment_param & ~is_exception
        is_multiplier = df["Parameter"].str.contains(" multiplier")
        is_parameter = ~has_compartment_param & ~is_multiplier

        has_any_exception_fields = has_exception_fields.any(axis=1)
        bad_parameter = (
            is_parameter & has_any_exception_fields & (df["Parameter"] != "run dates")
        )
        bad_compartment = is_compartment & has_any_exception_fields
        bad_multiplier = is_multiplier & has_any_exception_fields
        bad = bad_parameter | bad_compartment | bad_multiplier
        if bad.any():
            raise ValueError(
                "Region and/or dates included with parameters they do not "
                f"apply to: {df[bad]!r}"
            )

        multipliers = self._prepare_multipliers(df[is_multiplier])
        if multipliers:
            df = df.copy()
            for param, multiplier in multipliers.items():
                if param not in self.COMPARTMENT_VARIABLES:
                    raise ValueError("Cannot apply multiplier to {param!r}")
                df.loc[df["Parameter"] == param, "Value"] *= multiplier

        self.parameters = df[is_parameter]
        self.compartments = df[is_compartment][["Parameter", "Value"]]
        self.exceptions = self._prepare_exceptions(df[is_exception])

    def _set_global_parameters(self):
        self._assert_no_duplicate_values(self.parameters)

        for _, row in self.parameters.iterrows():
            self._set_parameter_from_df_row(row)

    def _set_global_compartment_variables(self):
        self._assert_no_duplicate_values(self.compartments)

        for _, row in self.compartments.iterrows():
            self.definition.set_compartment_variable(*row)

    def _set_exceptions(self):
        self.definition.clear_exceptions()
        for _, row in self.exceptions.iterrows():
            self.definition.add_exception(*row)

    def _prepare_exceptions(self, exceptions: pd.DataFrame) -> pd.DataFrame:
        """
        Group by time period and region set, creating a new df where
        each row corresponds to the Definition.add_exception interface,
        with regions as an array and variables as a dict.
        """
        output_columns = ["regions", "variables", "start", "end"]
        if exceptions.empty:
            return pd.DataFrame(columns=output_columns)
        return (
            exceptions.groupby(["Parameter", "Value", "Start date", "End date"])
            .apply(lambda group: tuple(sorted(set(group["Region"]))))
            .reset_index()
            .rename(columns={0: "regions"})
            .groupby(["regions", "Start date", "End date"])
            .apply(lambda group: dict(zip(group["Parameter"], group["Value"])))
            .reset_index()
            .rename(
                columns={0: "variables", "Start date": "start", "End date": "end",}
            )[output_columns]
        )

    def _prepare_multipliers(self, multipliers: pd.DataFrame) -> dict:
        """ returns a dict of param: multiplier pairs """
        self._assert_no_duplicate_values(multipliers)
        return dict(
            zip(
                multipliers["Parameter"].str.replace(" multiplier", ""),
                multipliers["Value"],
            )
        )

    def _set_parameter_from_df_row(self, row: namedtuple):
        param = row["Parameter"]
        if param == "run dates":
            if pd.notnull(row["Start date"]):
                self.definition.set_start_date(row["Start date"])
            if pd.notnull(row["End date"]):
                self.definition.set_end_date(row["End date"])
        else:
            value = row["Value"]
            getattr(self.definition, self.GLOBAL_PARAMETERS[param])(value)

    def _assert_no_duplicate_values(self, df: pd.DataFrame):
        counts = df.groupby("Parameter")["Value"].count()
        duplicates = list(counts[counts > 1].index)
        if duplicates:
            raise ValueError(
                "Duplicate values passed to a single scenario "
                f"for the following parameters: {duplicates!r}"
            )
