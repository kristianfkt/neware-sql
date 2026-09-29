import re
from enum import Enum


class TableType(Enum):
    MAIN_FIRST = "main_first"
    MAIN_SECOND = "main_second"
    AUX_FIRST = "aux_first"
    AUX_SECOND = "aux_second"
    META = "meta"
    OTHER = "other"


KEYS = {
    TableType.MAIN_FIRST: {"unit_id", "chl_id", "test_id", "seq_id"},
    TableType.MAIN_SECOND: {"seq_id"},
    TableType.AUX_FIRST: {"unit_id", "chl_id", "test_id", "auxchl_id", "seq_id"},
    TableType.AUX_SECOND: {"seq_id"},
    TableType.META: None,
    TableType.OTHER: None,
}


def is_main(table: str) -> bool:
    return table.startswith("data_")


def is_aux(table: str) -> bool:
    return table.startswith("aux_data_")


def is_first(table: str) -> bool:
    return bool(re.search(r"_\d{4}_\d{2}_\d{2}$", table)) and (
        is_main(table) or is_aux(table)
    )


def is_second(table: str) -> bool:
    return (not bool(re.search(r"_\d{4}_\d{2}_\d{2}$", table))) and (
        is_main(table) or is_aux(table)
    )


def is_meta(table: str) -> bool:
    known = {"test_note", "h_test", "test"}
    return table.startswith("h_test") or (table in known)


def get_type(table: str) -> TableType:
    if is_main(table) and is_first(table):
        _t = TableType.MAIN_FIRST
    elif is_main(table) and is_second(table):
        _t = TableType.MAIN_SECOND
    elif is_aux(table) and is_first(table):
        _t = TableType.AUX_FIRST
    elif is_aux(table) and is_second(table):
        _t = TableType.AUX_SECOND
    elif is_meta(table):
        _t = TableType.META
    else:
        _t = TableType.OTHER
    return _t


def get_keys(table: str) -> list[str] | None:
    keys = KEYS[get_type(table)]
    return list(keys) if keys is not None else None
