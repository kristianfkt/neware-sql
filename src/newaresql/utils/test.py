from newaresql.types import Test


def get_test_name(test: Test) -> str:
    return "-".join(
        str(test[key]) for key in ["dev_uid", "unit_id", "chl_id", "test_id"]
    )
