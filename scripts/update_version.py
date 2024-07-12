# update_version.py
import toml


def update_version():
    with open("VERSION") as version_file:
        version = version_file.read().strip()

    pyproject_path = "pyproject.toml"
    pyproject = toml.load(pyproject_path)

    pyproject["project"]["version"] = version

    with open(pyproject_path, "w") as toml_file:
        toml.dump(pyproject, toml_file)


if __name__ == "__main__":
    update_version()
