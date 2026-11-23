"""Install and check the ``dyscolab-modeling`` agent skill bundled with the libraries.

poincare, simbio and jablonski each install their part of the skill as data files in
``<environment>/share/agent-skills/dyscolab-modeling/``, where ``pixi skills`` finds it.
poincare ships ``SKILL.md`` and the poincare references and examples, simbio and
jablonski add their own files to the same folder. This module copies the merged skill
into an agent's skills folder, and checks whether a copy matches the installed libraries::

    python -m poincare.skill install --dest .claude/skills
    python -m poincare.skill check .claude/skills/dyscolab-modeling

``simbio.skill`` and ``jablonski.skill`` are aliases of this module.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

SKILL_NAME = "dyscolab-modeling"
PACKAGES = ("poincare", "simbio", "jablonski")
DEFAULT_DEST = Path(".agents") / "skills"

#: Location of the skill part inside the environment prefix (wheel ``data`` scheme).
DATA_PATH = f"share/agent-skills/{SKILL_NAME}"
#: Location of the skill part in a source checkout of each library.
SOURCE_PATH = Path("skill") / "share" / "agent-skills" / SKILL_NAME

#: Markers in SKILL.md around the "Versions and environment" section. The skills
#: repository (github.com/dyscolab/skills) replaces the whole section.
SECTION_START = "<!-- dyscolab-skill:versions-section:start -->"
SECTION_END = "<!-- dyscolab-skill:versions-section:end -->"
#: Markers around the list of documented versions, filled in by `install`.
VERSIONS_START = "<!-- dyscolab-skill:installed-versions:start -->"
VERSIONS_END = "<!-- dyscolab-skill:installed-versions:end -->"


@dataclass(frozen=True)
class Part:
    """The skill files installed by one package."""

    package: str
    version: str | None  # None: not installed
    files: dict[str, Path] = field(default_factory=dict)  # relative posix path -> file

    def describe(self) -> str:
        if self.version is None:
            return f"{self.package}: not installed"
        if not self.files:
            return f"{self.package} {self.version} (ships no skill files)"
        return f"{self.package} {self.version}"


def _editable_root(dist) -> Path | None:
    """Project directory of an editable (or local directory) install, if any."""
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    info = json.loads(text)
    if "dir_info" not in info or not info.get("url", "").startswith("file:"):
        return None
    return Path(url2pathname(urlparse(info["url"]).path))


def _is_skill_file(relative: str) -> bool:
    return "__pycache__" not in relative.split("/") and not relative.endswith(".pyc")


def find_part(package: str) -> Part:
    """Locate the skill files installed by ``package``.

    Editable installs take them from the source checkout (an editable install may not
    install data files, or may hold a copy from when it was built). Other installs list
    them in the distribution's RECORD.
    """
    try:
        dist = distribution(package)
    except PackageNotFoundError:
        return Part(package, None)

    files: dict[str, Path] = {}
    root = _editable_root(dist)
    if root is not None and (source := root / SOURCE_PATH).is_dir():
        for location in sorted(source.rglob("*")):
            relative = location.relative_to(source).as_posix()
            if location.is_file() and _is_skill_file(relative):
                files[relative] = location
        return Part(package, dist.version, files)

    prefix = DATA_PATH + "/"
    for entry in dist.files or ():
        path = str(entry).replace("\\", "/")
        start = path.find(prefix)
        if start < 0:
            continue
        relative = path[start + len(prefix) :]
        location = Path(dist.locate_file(entry))
        if _is_skill_file(relative) and location.is_file():
            files[relative] = location
    return Part(package, dist.version, files)


def find_parts() -> list[Part]:
    """The skill parts of poincare, simbio and jablonski (installed or not)."""
    return [find_part(package) for package in PACKAGES]


def replace_block(text: str, start: str, end: str, new: str) -> str:
    """Replace the text between the ``start`` and ``end`` markers (kept) with ``new``."""
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return f"{text[:i]}\n{new.strip()}\n{text[j:]}"


def versions_text(parts: Sequence[Part]) -> str:
    lines = ["Installed with `python -m poincare.skill install` from:", ""]
    lines += [f"- {part.describe()}" for part in parts]
    return "\n".join(lines)


def assemble(
    parts: Sequence[Part] | None = None, *, stamp: bool = True
) -> dict[str, bytes]:
    """Merge the parts into the skill's files, as ``{relative path: content}``.

    With ``stamp``, the versions of ``parts`` are written into SKILL.md.
    """
    if parts is None:
        parts = find_parts()
    files: dict[str, bytes] = {}
    owner: dict[str, str] = {}
    for part in parts:
        for relative, location in part.files.items():
            if relative in files:
                raise ValueError(
                    f"{relative} is shipped by both {owner[relative]} and {part.package}"
                )
            files[relative] = location.read_bytes()
            owner[relative] = part.package

    if owner.get("SKILL.md") != "poincare":
        raise RuntimeError(
            "poincare's SKILL.md was not found. Reinstall poincare, or install the skill "
            "from https://github.com/dyscolab/skills."
        )
    if stamp:
        text = files["SKILL.md"].decode("utf-8")
        text = replace_block(text, VERSIONS_START, VERSIONS_END, versions_text(parts))
        files["SKILL.md"] = text.encode("utf-8")
    return dict(sorted(files.items()))


def _is_our_skill(folder: Path) -> bool:
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        return False
    return f"\nname: {SKILL_NAME}\n" in skill_md.read_text(encoding="utf-8")


def install(dest: str | Path = DEFAULT_DEST) -> Path:
    """Copy the skill into ``dest/dyscolab-modeling``, replacing an older copy.

    ``dest`` is the agent's skills folder, for example ``.agents/skills``,
    ``.claude/skills`` or ``~/.codex/skills``.
    """
    files = assemble()
    target = Path(dest).expanduser() / SKILL_NAME
    if target.is_symlink():
        raise FileExistsError(
            f"{target} is a link (managed by `pixi skills`?). Remove the link first, or "
            "keep using it: it always matches its environment."
        )
    if target.exists():
        if not _is_our_skill(target):
            raise FileExistsError(f"{target} exists and is not the {SKILL_NAME} skill")
        shutil.rmtree(target)
    for relative, content in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return target


def check(skill_dir: str | Path) -> list[str]:
    """Differences between the skill in ``skill_dir`` and the installed libraries.

    Returns an empty list when ``skill_dir`` is up to date. An environment's own folder
    (``<prefix>/share/agent-skills/dyscolab-modeling``, for example linked by
    ``pixi skills``) has no version stamp, so it is compared without one.
    """
    folder = Path(skill_dir).expanduser()
    if not (folder / "SKILL.md").is_file():
        return [f"no SKILL.md in {folder}"]

    resolved = folder.resolve()
    in_environment = resolved.parts[-3:] == tuple(DATA_PATH.split("/"))
    expected = assemble(find_parts(), stamp=not in_environment)

    actual = {
        p.relative_to(folder).as_posix(): p
        for p in folder.rglob("*")
        if p.is_file() and _is_skill_file(p.relative_to(folder).as_posix())
    }
    problems = [f"missing: {r}" for r in expected if r not in actual]
    problems += [
        f"not in the installed libraries: {r}" for r in actual if r not in expected
    ]
    problems += [
        f"differs: {r}"
        for r, content in expected.items()
        if r in actual and actual[r].read_bytes() != content
    ]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m poincare.skill",
        description=f"Install or check the {SKILL_NAME} agent skill bundled with "
        "poincare, simbio and jablonski.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p_install = commands.add_parser(
        "install", help="copy the skill into a skills folder"
    )
    p_install.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"the agent's skills folder (default: {DEFAULT_DEST.as_posix()})",
    )
    p_check = commands.add_parser(
        "check", help="check a copy against the installed libraries"
    )
    p_check.add_argument(
        "skill_dir", type=Path, help=f"the {SKILL_NAME} folder to check"
    )
    args = parser.parse_args(argv)

    print("Installed: " + ", ".join(part.describe() for part in find_parts()))
    if args.command == "install":
        try:
            target = install(args.dest)
        except (FileExistsError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"Installed the {SKILL_NAME} skill in {target}")
        return 0

    problems = check(args.skill_dir)
    if not problems:
        print(f"{args.skill_dir} matches the installed libraries.")
        return 0
    print(f"{args.skill_dir} does not match the installed libraries:")
    for problem in problems:
        print(f"  {problem}")
    print(
        "Update it with: python -m poincare.skill install --dest "
        f"{Path(args.skill_dir).expanduser().resolve().parent}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
