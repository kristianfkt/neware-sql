import sqlalchemy as sa

from newaresql.clone.tables import get_keys
from newaresql.connectors.sql import SQLConnector
from newaresql.utils.config import get_config
from newaresql.utils.sql import wrap_table


class BTSSource(SQLConnector):
    """
    Connect to a BTS source server
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):

        host = get_config("NEWARE_BTS_HOST", value=host)
        port = get_config("NEWARE_BTS_PORT", value=port)
        database = get_config("NEWARE_BTS_DATABASE", value=database)
        username = get_config("NEWARE_BTS_USERNAME", value=username)
        password = get_config("NEWARE_BTS_PASSWORD", value=password)
        if not all([host, port, database, username, password]):
            raise ValueError("Missing required BTS connection parameters")

        url = sa.URL.create(
            "mysql+pymysql",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        super().__init__(url=url)
        return

    def get_combinations(self, table: str) -> list[dict] | None:
        """
        Get all combinations of key columns for the given table, excluding 'seq_id'.
        """
        keys = get_keys(table)
        if isinstance(keys, list) and ("seq_id") in keys:
            keys.remove("seq_id")
        if not keys:
            combinations = None
        else:
            combinations = self.get_distinct(table, columns=keys)
        return combinations

    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int | None:
        """
        Fetch the maximum 'seq_id' of a table, or a subset of a table
        """
        t = wrap_table(table, self.engine)
        stmt = sa.select(sa.func.max(t.c.seq_id).label("max_seq_id"))
        if combo:
            stmt = stmt.where(sa.and_(*(t.c[k] == v for k, v in combo.items())))
        result = self.get_query(stmt).select("max_seq_id").to_series().item()
        return result
