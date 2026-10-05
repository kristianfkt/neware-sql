import logging

from newaresql.schemas import schemas_0760, schemas_0800

logger = logging.getLogger(__name__)

_SCHEMAS: dict[str, dict[str, dict[str, type]]] = {
    "0760-24": {"main": schemas_0760.main_24, "aux": schemas_0760.aux_24},
    "0800-24": {"main": schemas_0800.main_24, "aux": schemas_0800.aux_24},
    "0800-26": {"main": schemas_0800.main_26, "aux": schemas_0800.aux_26},
}


def get_data_schema(version: str, dev_uid: int) -> dict[str, dict[str, type]]:
    """
    Fetches the data schema (name-type mapping) for the given BTS build version and device UID.

    Args:
        version (str): The BTS build version.
        dev_uid (int): The device UID.

    Returns:
        dict[str, dict[str, type]]: The data schema for the specified version and device.
    """
    logger.debug(f"Getting data schema for version {version} and device UID {dev_uid}")

    dev_type = str(dev_uid)[:2]
    key = f"{version}-{dev_type}"
    if key not in _SCHEMAS:
        raise ValueError(f"Unsupported version-device combination: {key}")
    return _SCHEMAS[key]
