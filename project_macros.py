# This file will be used by mkdocs-macros-plugin
# Configured in mkdocs.yml as module_name: project_macros

from scripts.readme_parser import extract_section, get_readme_content

# Global cache for README content to avoid multiple reads during one mkdocs build
_readme_content_cache = None


def get_cached_readme_content():
    global _readme_content_cache
    if _readme_content_cache is None:
        _readme_content_cache = get_readme_content()
    return _readme_content_cache


def define_env(env):
    """
    Hook function for mkdocs-macros plugin.
    """

    @env.macro
    def readme(section_key: str):
        """
        Macro to retrieve a section from README.md.
        Usage in Markdown: {{ readme("SECTION_KEY") }}
        """
        try:
            content = get_cached_readme_content()
            return extract_section(content, section_key)
        except Exception as e:
            return f"<strong style='color:red;'>Error processing README section '{section_key}': {e}</strong>"

    @env.macro
    def readme_verbatim(section_key: str):
        """
        Macro to retrieve a section from README.md without stripping whitespace.
        Usage in Markdown: {{ readme_verbatim("SECTION_KEY") }}
        """
        try:
            content = get_cached_readme_content()
            return extract_section(content, section_key, strip=False)
        except Exception as e:
            return f"<strong style='color:red;'>Error processing README section '{section_key}': {e}</strong>"
