import re
from pathlib import Path
from runpy import run_path

import pytest

from poincare import skill

SOURCE = Path(__file__).parents[3] / skill.SOURCE_PATH
EXAMPLES = sorted((SOURCE / "examples").glob("*.py"))
MARKERS = (
    skill.SECTION_START,
    skill.VERSIONS_START,
    skill.VERSIONS_END,
    skill.SECTION_END,
)


def frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines())


def test_frontmatter():
    fm = frontmatter((SOURCE / "SKILL.md").read_text(encoding="utf-8"))
    assert fm["name"] == skill.SKILL_NAME == SOURCE.name
    assert 0 < len(fm["description"]) <= 1024
    assert len(fm["compatibility"]) <= 500


def test_section_markers_in_order():
    text = (SOURCE / "SKILL.md").read_text(encoding="utf-8")
    assert all(text.count(marker) == 1 for marker in MARKERS)
    positions = [text.index(marker) for marker in MARKERS]
    assert positions == sorted(positions)


@pytest.mark.parametrize("md", sorted(SOURCE.rglob("*.md")), ids=lambda p: p.name)
def test_relative_links_exist(md: Path):
    for target in re.findall(
        r"\]\(([^)#:]+)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")
    ):
        if Path(target).name.startswith(("simbio", "jablonski")):
            continue  # shipped by simbio and jablonski
        assert (md.parent / target).exists(), f"{md.name} links to missing {target}"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example(example: Path):
    run_path(str(example), run_name="test")


def test_part_is_found():
    part = skill.find_part("poincare")
    assert {"SKILL.md", "references/poincare.md"} <= set(part.files)
    assert skill.find_part("not-a-dyscolab-package").version is None


def test_install_and_check(tmp_path: Path):
    target = skill.install(tmp_path)
    assert target == tmp_path / skill.SKILL_NAME
    text = (target / "SKILL.md").read_text(encoding="utf-8")
    poincare = skill.find_part("poincare")
    assert f"- poincare {poincare.version}" in text
    assert skill.check(target) == []
    assert skill.main(["check", str(target)]) == 0

    (target / "references" / "poincare.md").write_text("old", encoding="utf-8")
    (target / "references" / "extra.md").write_text("", encoding="utf-8")
    (target / "references" / "units.md").unlink()
    assert sorted(skill.check(target)) == [
        "differs: references/poincare.md",
        "missing: references/units.md",
        "not in the installed libraries: references/extra.md",
    ]
    assert skill.main(["check", str(target)]) == 1

    # Installing again replaces the outdated copy.
    skill.install(tmp_path)
    assert skill.check(target) == []


def test_check_environment_folder_without_stamp(tmp_path: Path):
    # The layout of <prefix>/share/agent-skills/, where wheels install the parts.
    folder = tmp_path / skill.DATA_PATH
    for relative, content in skill.assemble(stamp=False).items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_bytes(content)
    assert skill.check(folder) == []

    # Linked into an agent's skills folder, as `pixi skills` does.
    link = tmp_path / "agent" / skill.SKILL_NAME
    link.parent.mkdir()
    link.symlink_to(folder, target_is_directory=True)
    assert skill.check(link) == []

    # A stamped copy is not what wheels install there.
    skill.install(tmp_path / "share" / "agent-skills")
    assert skill.check(folder) == ["differs: SKILL.md"]


def test_install_refuses_other_folders(tmp_path: Path):
    other = tmp_path / skill.SKILL_NAME
    other.mkdir()
    (other / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        skill.install(tmp_path)
    assert (other / "notes.txt").exists()


def test_install_refuses_links(tmp_path: Path):
    (tmp_path / skill.SKILL_NAME).symlink_to(SOURCE, target_is_directory=True)
    with pytest.raises(FileExistsError, match="link"):
        skill.install(tmp_path)


def test_assemble_rejects_conflicting_files(tmp_path: Path):
    file = tmp_path / "a.md"
    file.write_text("", encoding="utf-8")
    parts = [
        skill.Part("poincare", "1", {"SKILL.md": SOURCE / "SKILL.md", "a.md": file}),
        skill.Part("simbio", "1", {"a.md": file}),
    ]
    with pytest.raises(ValueError, match="a.md"):
        skill.assemble(parts)


def test_replace_block():
    text = "a\n<s>\nold\n<e>\nb"
    assert skill.replace_block(text, "<s>", "<e>", "new") == "a\n<s>\nnew\n<e>\nb"
