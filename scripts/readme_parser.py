import re
from pathlib import Path

# PROJECT_ROOT should be GenAIResultsComparator/
# Path(__file__) is GenAIResultsComparator/scripts/readme_parser.py
# .parent is GenAIResultsComparator/scripts/
# .parent.parent is GenAIResultsComparator/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"


def get_readme_content() -> str:
    if not README_PATH.exists():
        raise FileNotFoundError(f"README.md not found at {README_PATH}")
    return README_PATH.read_text(encoding="utf-8")


def extract_section(readme_content: str, section_key: str, strip: bool = True) -> str:
    """
    Extracts a section from the README content based on markers.
    Markers are expected to be <!-- SECTION_KEY_START --> and <!-- SECTION_KEY_END -->.
    """
    start_marker_tag = f"<!-- {section_key.upper()}_START -->"
    end_marker_tag = f"<!-- {section_key.upper()}_END -->"

    pattern = re.compile(
        f"{re.escape(start_marker_tag)}(.*?){re.escape(end_marker_tag)}", re.DOTALL
    )
    match = pattern.search(readme_content)

    if not match:
        raise ValueError(
            f"Section markers for '{section_key}' not found in README.md. "
            f"Expected {start_marker_tag} and {end_marker_tag}"
        )

    content = match.group(1)
    return content.strip() if strip else content
