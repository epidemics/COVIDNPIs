import logging
import requests
import re
import json

import pandas as pd
import numpy as np

from ..regions import RegionDataset
from ..utils import utc_date

log = logging.getLogger(__name__)


SKIP_NAMES = {
    "earth",
    "africa",
    "south america",
    "middle east",
    "european union",
    "london",
    "oxford",
    "san francisco",
    "dubai",
    "wuhan",
}

SUBST_NAMES = {"united kingdon": "United Kingdom", "ivory coast": "Cote d'Ivoire"}

QUANTILES = np.arange(0.01, 1.0, 0.01)
QUANTILE_COLS = [f"{x:.2f}" for x in QUANTILES]


def calculations(pa):
    pred_xs = np.array(pa["value"]["floatCdf"]["xs"])
    pred_ys = np.array(pa["value"]["floatCdf"]["ys"])
    pdf = np.concatenate((pred_ys[1:], [1.0])) - pred_ys
    mean = np.dot(pdf, pred_xs)
    var = np.dot(pdf, np.abs(pred_xs - mean) ** 2)
    # Quantiles, assumes both pred_ys, pred_xs are sorted
    qvals = [pred_xs[sum(pred_ys < qprob)] for qprob in QUANTILES]
    return [mean, var] + qvals


def import_foretold(rds: RegionDataset, foretold_channel: str):

    skipped = set()
    not_found = set()
    conflicts = set()
    data = []

    ft = fetch_foretold(foretold_channel)
    djs = json.loads(ft)

    for p in djs["data"]["measurables"]["edges"]:
        data_line = []
        pa = p["node"]["previousAggregate"]
        if not pa:
            continue
        name = re.sub("^@locations/n-", "", p["node"]["labelSubject"]).replace("-", " ")
        if name in SUBST_NAMES:
            name = SUBST_NAMES[name]
        if name in SKIP_NAMES:
            skipped.add(name)
            continue
        rs = rds.find_all_by_name(name)
        if len(rs) < 1:
            not_found.add(name)
            continue
        if len(rs) > 1:
            conflicts.add(name)
            continue

        data_line.append(rs[0].Code)
        data_line.append(utc_date(p["node"]["labelOnDate"]))
        data_line.extend(calculations(pa))
        data.append(data_line)

    if skipped:
        log.info(f"Skipped {len(skipped)} records: {skipped!r}")
    if not_found:
        log.info(f"No matches for {len(not_found)} records: {not_found!r}")
    if conflicts:
        log.info(f"Multiple matches for {len(conflicts)} records: {conflicts!r}")

    df = pd.DataFrame(
        data, columns=["Code", "Date", "Mean", "Variance", *QUANTILE_COLS]
    )
    df.set_index("Code", inplace=True)
    return df.sort_values(["Code", "Date"])


def fetch_foretold(channel_id: str) -> bytes:
    """Fetch the data from foretold.io.

    :param channel_id: Channel id (UUID)
    :param output_path: If set, writes into the path.
    :returns: None if written to a file; parsed JSON if output_path not set.
    """
    if not channel_id:
        raise ValueError("Please, set channel_id.")
    url = "https://www.foretold.io/graphql/"
    headers = {
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "keep-alive",
        "DNT": "1",
        "Origin": "https://www.foretold.io",
    }
    data = {
        "query": QUERY,
        "variables": {
            "channelId": channel_id,
            "first": 500,
            "order": [
                {"field": "stateOrder", "direction": "ASC"},
                {"field": "refreshedAt", "direction": "DESC"},
            ],
        },
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Error fetching data, status code: {response.status_code}")
    return response.content


QUERY = """query measurables(
  $measurableIds: [String!]
  $states: [measurableState!]
  $channelId: String
  $seriesId: String
  $creatorId: String
  $first: Int500
  $last: Int500
  $after: Cursor
  $before: Cursor
  $order: [OrderMeasurables]
) {
  measurables(
    measurableIds: $measurableIds
    states: $states
    channelId: $channelId
    seriesId: $seriesId
    creatorId: $creatorId
    first: $first
    last: $last
    after: $after
    before: $before
    order: $order
  ) {
    total
    pageInfo {
      hasPreviousPage
      hasNextPage
      startCursor
      endCursor
      __typename
    }
    edges {
      node {
        id
        labelCustom
        valueType
        measurementCount
        measurerCount
        labelSubject
        labelProperty
        state
        labelOnDate
        stateUpdatedAt
        expectedResolutionDate
        previousAggregate {
          id
          valueText
          value {
            floatCdf {
              xs
              ys
              __typename
            }
            floatPoint
            percentage
            binary
            unresolvableResolution
            comment
            __typename
          }
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}"""
