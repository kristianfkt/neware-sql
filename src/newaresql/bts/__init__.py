from newaresql.bts.connector import BTSConnector


def connect(
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> BTSConnector:
    return BTSConnector(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
