from collections.abc import Callable, Mapping
from functools import partial
from inspect import cleandoc

from ..types import System
from .latex import ToLatex, latex_equations, parameter_table, variable_table

type Markdown = str


def _render(text: str) -> None:
    try:
        import marimo as mo

        if mo.running_in_notebook():
            mo.output.append(mo.md(text))
            return
    except (ImportError, AttributeError):
        pass
    from IPython.display import Markdown as IPythonMarkdown
    from IPython.display import display

    display(IPythonMarkdown(text))


def md_sections(blocks: Mapping[str, str], level: int = 2, show=True) -> Markdown:
    head = "#" * level
    parts = [f"{head} {title}\n\n{cleandoc(body)}" for title, body in blocks.items()]
    sep = "\n\n"
    markdown = sep.join(parts)
    if show:
        _render(markdown)
    return markdown


default_notebook_sections: Mapping[str, Callable[[System, ToLatex], Markdown]] = {
    "Equations": partial(latex_equations, format="markdown"),
    "Variables": partial(variable_table, format="markdown"),
    "Parameters": partial(parameter_table, format="markdown"),
}


def notebook_report(
    model: type[System],
    transform: dict | None = None,
    descriptions: dict | None = None,
    replace_algebraics: bool = False,
    sections: Mapping[
        str, Callable[[System, ToLatex], Markdown]
    ] = default_notebook_sections,
    level: int = 2,
    show: bool = True,
) -> Markdown | None:
    """Markdown counterpart of `model_report`, displayed in marimo or Jupyter.

    Each section writer takes the model and a shared `ToLatex` and returns
    Markdown. With `show=False` the Markdown is returned instead of displayed.
    """
    latex = ToLatex(
        system=model,
        transform=transform if transform is not None else {},
        descriptions=descriptions,
        replace_algebraics=replace_algebraics,
    )
    blocks = {title: writer(model, latex=latex) for title, writer in sections.items()}
    markdown = md_sections(blocks, level=level, show=show)
    if not show:
        return markdown
