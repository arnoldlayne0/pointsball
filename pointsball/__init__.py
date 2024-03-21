from dagster import Definitions, EnvVar, load_assets_from_modules
from dagster_polars import PolarsParquetIOManager

from .assets import fpl_datasets, reddit_datasets
from .resources.reddit_resources import RedditResource

fpl_assets = load_assets_from_modules([fpl_datasets])
reddit_assets = load_assets_from_modules([reddit_datasets])
all_assets = [*fpl_assets, *reddit_assets]

defs = Definitions(
    assets=all_assets,
    resources={
        "polars_parquet_io_manager": PolarsParquetIOManager(base_dir="data/"),
        "reddit_credentials": RedditResource(
            client_id=EnvVar("REDDIT_CLIENT_ID"),
            secret=EnvVar("REDDIT_SECRET"),
            user_agent=EnvVar("REDDIT_USER_AGENT"),
            username=EnvVar("REDDIT_USERNAME"),
            password=EnvVar("REDDIT_PASSWORD"),
        ),
    },
)
