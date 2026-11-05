# Neware SQL

Read Neware BTS test data from a MySQL database or a local Parquet copy into Polars DataFrames.

## Requirements

- Python 3.11 or later
- Access to a Neware BTS MySQL database to fetch or synchronize data; an existing local copy can be read offline

Install from a local checkout:

```bash
python -m pip install .
```

Configure credentials either when connecting or with environment variables:

- `NEWARE_BTS_HOST`
- `NEWARE_BTS_PORT`
- `NEWARE_BTS_USERNAME`
- `NEWARE_BTS_PASSWORD`
- `NEWARE_BTS_DATABASE`

Explicit connection arguments take precedence over environment variables.

## Quick Start

Use a connector directly when making several requests. The context manager disposes of its database engine when the block exits.

```python
import newaresql

credentials = {
    "host": "db.example.com",
    "port": 3306,
    "username": "your-user",
    "password": "your-password",
    "database": "your-database",
}

with newaresql.connect(**credentials) as connector:
    tests = connector.list_tests()
    result = connector.get_data(
        tests[0],
        conditions={"cycle": (None, 5)},
    )

main = result.main
aux = result.aux
combined = result.data
```

Alternatively, use the package-level helpers. They create and close a connector for each call:

```python
import newaresql

tests = newaresql.list_tests()
result = newaresql.get_data(
    tests[0],
    conditions={"cycle": (None, 5)},
)
```

## Results and Filtering

`list_tests()` returns test metadata as a list of dictionaries. `get_tests()` returns the test overview as a Polars DataFrame. `get_data()` returns a `TestData` object with `main`, `aux`, and `data` properties; `aux` is `None` when no auxiliary data is available.

Conditions can use BTS column names, BDF codes, or exact BDF labels. Multiple conditions are combined with AND. Scalars specify equality, lists specify membership, and two-item tuples specify inclusive bounds. Use `None` for an open bound. Filter values use raw database units.

Recommended filtering columns include:

- `seq_id` (`record_index`, `Record Index / 1`)
- `cycle` (`cycle_count`, `Cycle Count / 1`)
- `step_id`
- `step_type`
- `test_atime`

`step_count` is added to the returned data from a separate query; it is not present in the raw columns used for filtering. Filter by `cycle`, `step_id`, or another raw column instead.

Choose the output column naming convention with `naming` when fetching data, or change it later on the returned `TestData` object. Changing `result.naming` affects the column names returned by subsequent accesses to `main`, `aux`, and `data`; it does not fetch the data again:

```python
result = newaresql.get_data(tests[0], naming="code")
main_code = result.main

result.naming = "bts"
main_bts = result.main
```

Supported values are `"bts"`, `"code"`, and `"label"`; the default is `"label"`.

## Local Storage

Use `newaresql.local` to synchronize test data into a local directory and read it later without a database connection. Set `NEWARE_LOCAL_ROOT` or pass `root` explicitly. Synchronization creates the root if needed and uses the same database credentials as the connector.

```python
from pathlib import Path
from newaresql import local

root = Path("./neware-data")
local.update(root=root)

tests = local.list_tests(root=root)
if tests:
    result = local.get_data(tests[0], root=root, naming="code")
    main = result.main.collect()
    combined = result.data.collect()
```

`local.get_tests()` returns a metadata DataFrame, while `local.list_tests()` returns dictionaries. Both return `None` when the root or metadata files are missing. Test dictionaries must contain `dev_uid`, `unit_id`, `chl_id`, and `test_id` to locate their data.

`local.get_data()` returns a `LocalData` object with `main`, `aux`, and `data` properties, using the same naming conventions as database results. It returns lazy Polars frames by default; pass `lazy=False` for eager frames or change `result.lazy` later. Accessing lazy auxiliary data still collects channel IDs to construct the pivot. `aux` is `None` when no auxiliary files exist.

`local.update()` appends data after the highest saved sequence ID for each test. Use `callback` to select tests, `chunk_size` to control the number of sequence IDs requested per chunk, and `workers` for parallel test updates (`None`, `0`, or `1` for sequential updates; `-1` for the thread pool default). An existing connector can be supplied with `connector`. Per-test data update failures are logged and do not stop other tests; check the logs for failures.

```python
local.update(
    root=root,
    callback=lambda test: test["dev_uid"] == 1,
    chunk_size=100_000,
    workers=4,
)
```

Each test is stored in a `<dev_uid>-<unit_id>-<chl_id>-<test_id>` directory containing metadata and chunked main and auxiliary Parquet files. To remove local data, `local.delete(test, root=root)` removes one test, `local.clean(root=root)` removes immediate subdirectories without `metadata.parquet`, and `local.purge(root=root)` removes all immediate subdirectories. These commands delete Parquet files before removing directories; non-Parquet files or nested directories can prevent removal. Use `clean` and `purge` only on a dedicated local root.

## Supported Versions

- BTS build `0760`, device type `24`
- BTS build `0800`, device types `24` and `26`

## Development

Version-specific transformations are in `src/newaresql/expressions/`, and database schemas are in `src/newaresql/schemas/`. Column-name mappings between BTS names, BDF codes, and BDF labels are defined in `src/newaresql/bdf.py`.


# Known issues
- Disconnected temperature sensors may a large, false temperature instead of null.  This will be adressed in a later release.