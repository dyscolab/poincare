from collections.abc import Mapping, Sequence

from tabulate import TableFormat, tabulate


class Table:
    def __init__(
        self,
        *,
        table: Sequence[Sequence[str]],
        headers: Sequence[str],
        title: str | None = None,
    ):
        self.table = table
        self.headers = headers
        self.title = title

    @classmethod
    def from_attributes(
        cls,
        iterable,
        *,
        attributes: Sequence[str] | Mapping[str, str],
    ):
        if isinstance(attributes, Mapping):
            headers = list(attributes.values())
        else:
            headers = attributes

        table = [[getattr(c, k) for k in attributes] for c in iterable]
        return cls(table=table, headers=headers)

    def as_table(self, *, tablefmt: TableFormat | str, **kwargs):
        return tabulate(self.table, headers=self.headers, tablefmt=tablefmt, **kwargs)

    def __repr__(self):
        table = self.as_table(tablefmt="simple")
        if self.title is None:
            return table
        _, header_separator, _ = table.split("\n", maxsplit=2)
        sep = "-" * len(header_separator)
        return f"{self.title}\n{sep}\n{table}"

    def _repr_html_(self):
        table = self.as_table(tablefmt="html")
        if self.title is None:
            return table

        n = len(self.headers)
        title_row = f'\n<colgroup span="{n}"></colgroup><tr><th colspan="{n}" scope="colgroup">{self.title}</th></tr>'
        a, b, c = table.partition("<thead>")
        return f"{a}{b}{title_row}{c}"
