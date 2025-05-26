from typing import Any, List, Optional

from sqlalchemy import CursorResult, Inspector, MetaData, Table, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


class PostgresConnection:
    def __init__(
        self,
        host: str = "localhost",
        port: int | str = 5432,
        user: str | None = None,
        password: str | None = None,
    ):
        database_url = f"postgresql://{user}:{password}@{host}:{port}"

        self.engine = create_engine(database_url)

        self.__metadata: MetaData = MetaData()
        self.__metadata.reflect(self.engine)

        self.__inspector: Inspector = Inspector(self.engine)

        self.session: Session = Session(bind=self.engine)

    def update_metadata(self):
        self.__metadata.reflect(self.engine)

    def create_tables(self, base: DeclarativeBase):
        base.metadata.create_all(bind=self.engine)
        self.__metadata = base.metadata

    def get_data_from_table(
        self,
        table_class=None,
        table_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        filters: Optional[List[Any]] = None,
    ) -> List[Any]:
        """
        Get data from specific columns in a table.
        Args:
            table_class: ORM class representing the table
            table_name (str): Name of the table to insert data into
            columns (list[str]): List of column names to fetch. If not provided -> all table columns
            limit (int): Limit the numer of results
            filters (list[str]): SQLAlchemy type column filters
        Returns:
            list[Any]: List of rows with specified columns.
        """

        table_obj: Table

        # Input check and table object extraction
        if table_name is not None:
            if table_class is not None:
                raise ValueError(
                    "Table_names or table_classes must be provided, not both"
                )
            else:
                table_obj = self.__metadata.tables[table_name]
        else:
            if table_class is None:
                raise ValueError("Either table_names or table_classes must be provided")
            else:
                table_obj = table_class.__table__

        # Columns validation
        table_columns = [column.name for column in table_obj.columns]
        if columns is None:
            columns = table_columns
        else:
            invalid_columns = [col for col in columns if col not in table_columns]
            if invalid_columns:
                raise ValueError(
                    f"Invalid columns {invalid_columns} for table {table_obj.name} valid are {table_columns}"
                )

        # Query the database
        query = self.session.query(
            *[column for column in table_obj.columns if column.name in columns]
        )

        if filters:
            query = query.filter(*filters)

        if limit:
            query = query.limit(limit)

        results = query.all()

        return results

    def insert_data_to_table(
        self,
        data: List[List[str]],
        columns: Optional[List[str]] = None,
        table_class: Optional[Any] = None,
        table_name: Optional[str] = None,
    ) -> CursorResult:
        """
        Insert data into a specific table using ORM classes or a table name.
        Args:
            data (list(list(str))): A list of rows with data.
            columns: (list(str)) | None: A list of column names. If None then all table columns.
            table_class: ORM class representing the table.
            table_name (str): Name of the table to insert data into.
        """
        table_obj: Table

        # Input check and table object extraction
        if table_name is not None:
            if table_class is not None:
                raise ValueError(
                    "Table_names or table_classes must be provided, not both"
                )
            else:
                table_obj = self.__metadata.tables[table_name]
        else:
            if table_class is None:
                raise ValueError("Either table_names or table_classes must be provided")
            else:
                table_obj = table_class.__table__

        # Columns validation
        table_columns = [column.name for column in table_obj.columns]
        if columns is None:
            columns = table_columns
        else:
            invalid_columns = [col for col in columns if col not in table_columns]
            if invalid_columns:
                raise ValueError(
                    f"Invalid columns {invalid_columns} for table {table_obj.name}"
                )

        # data insertion
        insert_data = [dict(zip(columns, row)) for row in data]

        try:
            result = self.session.execute(table_obj.insert(), insert_data)
        except Exception as e:
            raise e
        finally:
            self.session.commit()

        return result

    def get_tables_list(self) -> list[str]:
        """
        List all tables in the database.
        Returns:
            list[str]: List of table names.
        """

        return self.__inspector.get_table_names()

    def get_columns_list(
        self, table_name: Optional[str] = None, table_class: Optional[Any] = None
    ) -> List[str]:
        """
        List all columns in a specific table.
        Args:
            table_name (str): Name of the table to inspect.
            table_class: ORM class representing the table.
        Returns:
            list[str]: List of column names.
        """

        # Input check
        if table_class is not None:
            if table_name is not None:
                raise ValueError("Table_name or table_class must be provided, not both")
            else:
                table_name = table_class.__tablename__

        if table_name is None:
            raise ValueError("Either table_name or table_class must be provided")

        # Check for existing tables
        if table_name not in self.__inspector.get_table_names():
            raise ValueError(f"Table '{table_name}' does not exist.")

        return [column["name"] for column in self.__inspector.get_columns(table_name)]

    def delete_tables(
        self,
        table_names: Optional[list[str]] = None,
        table_classes: Optional[list] = None,
    ) -> None:
        """
        Delete tables from the database. Table names and table classes will be united
        Args:
            table_classes list: List of ORM classes that represent table.
            table_names list(str): List of names of the tables to drop.
        """

        # Input check
        if table_classes is not None:
            if table_names is not None:
                raise ValueError(
                    "Table_names or table_classes must be provided, not both"
                )
            else:
                table_names = [
                    table_class.__tablename__ for table_class in table_classes
                ]

        if table_names is None:
            raise ValueError("Either table_names or table_classes must be provided")

        for table_name in table_names:
            if table_name not in self.__metadata.tables:
                print(f"WARNING: '{table_name}' table was not found")
                continue

            self.__metadata.tables[table_name].drop(self.engine)
