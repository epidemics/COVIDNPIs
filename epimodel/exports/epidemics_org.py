import datetime
import getpass
import json
import logging
import socket
from pathlib import Path

from tqdm import tqdm

from ..regions import Region

log = logging.getLogger(__name__)


class WebExport:
    """
    Document holding one data export to web. Contains a subset of Regions.
    """

    def __init__(self, comment=None):
        self.created = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.created_by = f"{getpass.getuser()}@{socket.gethostname()}"
        self.comment = comment
        self.export_regions = {}

    def to_json(self):
        return {
            "created": self.created.isoformat(),
            "created_by": self.created_by,
            "comment": self.comment,
            "regions": {k: a.to_json() for k, a in self.export_regions.items()},
        }

    def new_region(self, region):
        er = WebExportRegion(region)
        self.export_regions[region.Code] = er
        return er

    def write(self, path, name=None):
        if name is None:
            name = f"export-{self.created.isoformat()}"
            if self.comment:
                name += self.comment
        outdir = Path(path)
        assert (not outdir.exists()) or outdir.is_dir()
        exdir = Path(path) / name
        exdir.mkdir(exist_ok=False, parents=True)
        for rc, er in tqdm(list(self.export_regions.items()), desc="Writing regions"):
            fname = f"extdata-{rc}.json"
            er.data_url = f"{name}/{fname}"
            with open(exdir / fname, "wt") as f:
                json.dump(er.data_ext(), f)
        with open(exdir / "data-CHANNEL-v4.json", "wt") as f:
            json.dumps(self.to_json(), f, indent=2)
        log.info(f"Exported {len(self.export_regions)} regions to {exdir}")


class WebExportRegion:
    def __init__(self, region):
        assert isinstance(region, Region)
        self.region = region
        # Any per-region data. Large ones should go to data_ext.
        self.data = {}  # {name: anything}
        # Extended data to be written in a separate per-region file
        self.data_ext = {}  # {name: anything}
        # Relative URL of the extended data file, set on write
        self.data_url = None

    def to_json(self):
        d = {
            "data": self.data,
            "data_url": self.data_url,
            "name": self.region.DisplayName,
        }
        for n in [
            "Population",
            "Lat",
            "Lon",
            "OfficialName",
            "Level",
            "M49Code",
            "ContinentCode",
            "SubregionCode",
            "CountryCode",
            "CountryCodeISOa3",
            "SubdivisionCode",
        ]:
            d[n] = self.region[n]
            d[n.lower()] = self.region[n]
        return d
