"""Glosario del sitio (report/glosario.qmd) y sus enlaces desde las páginas.

Las páginas enlazan términos con el shortcode `{{< term id >}}`; la extensión
`report/_extensions/glosario` toma como definición el primer párrafo tras `### Nombre {#id}`.
Un id inexistente rompería el render, así que se comprueba aquí, sin necesidad de Quarto.
"""

import re
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "report"
GLOSSARY = REPORT / "glosario.qmd"
PAGES = sorted(p for p in REPORT.glob("*.qmd") if p != GLOSSARY)

TERM_USE = re.compile(r"\{\{<\s*term\s+([\w-]+)")
HEADING = re.compile(r"^###\s+(.+?)\s+\{#([\w-]+)\}\s*$", re.MULTILINE)
SECTION = re.compile(r"^##\s", re.MULTILINE)


def glossary_entries() -> list[tuple[str, str, str]]:
    """(id, nombre, definición corta) de cada término, en orden de aparición."""
    text = GLOSSARY.read_text(encoding="utf-8")
    headings = list(HEADING.finditer(text))
    entries = []
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        body = text[heading.end() : following.start() if following else len(text)]
        body = SECTION.split(body)[0]  # sin el encabezado de la sección siguiente
        definition = body.strip().split("\n\n")[0]
        entries.append((heading.group(2), heading.group(1), " ".join(definition.split())))
    return entries


def test_glossary_has_the_key_terms():
    ids = {entry_id for entry_id, _, _ in glossary_entries()}
    assert {"lrfmc", "k-means", "silhouette", "ari", "centroide", "pca", "tsne"} <= ids


def test_glossary_ids_are_unique():
    ids = [entry_id for entry_id, _, _ in glossary_entries()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("entry", glossary_entries(), ids=lambda e: e[0])
def test_every_entry_has_a_short_definition(entry):
    _, _, definition = entry
    # Es lo que se ve en el tooltip: un párrafo de texto, breve.
    assert definition and not definition.startswith(("#", "-", "|", ":::", "{{<"))
    assert len(definition) <= 300


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_linked_terms_exist_in_glossary(page):
    ids = {entry_id for entry_id, _, _ in glossary_entries()}
    used = set(TERM_USE.findall(page.read_text(encoding="utf-8")))
    assert used <= ids, f"Términos sin entrada en el glosario: {sorted(used - ids)}"


def test_pages_link_to_the_glossary():
    linked = [p.name for p in PAGES if TERM_USE.search(p.read_text(encoding="utf-8"))]
    assert linked == [p.name for p in PAGES]
