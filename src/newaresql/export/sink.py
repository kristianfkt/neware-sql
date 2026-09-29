from newaresql.connectors.blob import BlobConnector
from newaresql.connectors.bts import SQLConnector
from newaresql.connectors.file import FileConnector


class SQLSink(SQLConnector):
    pass


class FileSink(FileConnector):
    pass


class BlobSink(BlobConnector):
    pass
