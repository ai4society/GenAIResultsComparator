import tomllib


def get_version():
    with open("pyproject.toml", "rb") as f:  # Use "rb" for tomllib
        data = tomllib.load(f)  # Use toml.load(f) for toml package
    return data["project"]["version"]


if __name__ == "__main__":
    print(get_version())
