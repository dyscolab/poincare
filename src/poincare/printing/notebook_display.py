from inspect import cleandoc

from ..types import System
from .latex import ToLatex, latex_equations, variable_table, parameter_table

def _render(text: str) -> None:
    try:
        import marimo as mo
        if mo.running_in_notebook():
            mo.output.append(mo.md(text))
            return
    except (ImportError, AttributeError):
        pass
    from IPython.display import display, Markdown
    display(Markdown(text))


def md_sections(blocks: dict[str, str], level: int = 2, show = True) -> str:
    head = "#" * level
    parts = [f"{head} {title}\n\n{cleandoc(body)}" for title, body in blocks.items()]
    sep = "\n\n"
    markdown = sep.join(parts)
    if show:
        _render(markdown)
    return markdown

def notebook_report(model: System, show = True): 
    latex = ToLatex(model)
    sections = {"Equations": latex_equations(model, latex=latex, format = "markdown"), 
                "Variables": variable_table(model, latex=latex, format = "markdown"), 
                "Parameters": parameter_table(model, latex=latex, format = "markdown")}
    md = md_sections(sections, show = show)
    if not show:
        return md
    