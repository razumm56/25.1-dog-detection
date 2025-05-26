from threading import Thread
from queue import Queue

from typing import Any, List, Optional

from dataset.SQLALchemy_utils import PostgresConnection


class TableInsertScheme:
    def __init__(
            self,
            table_name: Optional[str] = None,
            table_class: Optional[Any] = None,
            columns: Optional[List[str]] = None,
    ) -> None:
        if table_name is not None and table_class is not None:
            raise ValueError("Table_names or table_classes must be provided, not both")
        elif table_name is None and table_class is None:
            raise ValueError("Either table_names or table_classes must be provided")

        if table_class is not None:
            table_name = table_class.__tablename__

        self.table_name = table_name
        self.columns = columns

        self.queue = Queue()


class DBWriter:
    def __init__(self, conn: PostgresConnection):
        self.conn = conn

        self.schemes: list[TableInsertScheme] = []
        self.__db_saving_thread = Thread(target=self.__db_saving_worker, args=(self.conn, self.schemes))

    def add_scheme(self, scheme: TableInsertScheme):
        self.schemes.append(scheme)

    @staticmethod
    def __db_saving_worker(conn, schemes):
        while True:
            with conn.session.begin():
                for scheme in schemes.copy():
                    values = []
                    while not scheme.queue.empty():
                        value = scheme.queue.get_nowait()

                        if value is None:
                            schemes.remove(scheme)
                            break

                        values.append(value)

                    if values != []:
                        conn.insert_data_to_table(
                            table_name=scheme.table_name,
                            columns=scheme.columns,
                            data=values,
                        )

            if schemes == []:
                break

    def start(self):
        self.__db_saving_thread.start()

    def stop(self):
        [scheme.queue.put(None) for scheme in self.schemes]
        self.__db_saving_thread.join()
