from dagster import Definitions, load_assets_from_modules
from dagster_polars import PolarsParquetIOManager

from .assets import fpl_datasets

fpl_assets = load_assets_from_modules([fpl_datasets])

defs = Definitions(
    assets=fpl_assets,
    resources={"polars_parquet_io_manager": PolarsParquetIOManager(base_dir="data/")},
)
