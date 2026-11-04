# Neware SQL

Read Neware BTS test data from a MySQL database into Polars DataFrames.

## Requirements

- Python 3.11 or later
- Access to a Neware BTS MySQL database

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

## Supported Versions

- BTS build `0760`, device type `24`
- BTS build `0800`, device types `24` and `26`

## Development

Version-specific transformations are in `src/newaresql/expressions/`, and database schemas are in `src/newaresql/schemas/`. Column-name mappings between BTS names, BDF codes, and BDF labels are defined in `src/newaresql/bdf.py`.