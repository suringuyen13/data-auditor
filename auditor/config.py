from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_rules(file_path, variable_name):
    """Load a named dictionary of user rules from a Python file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {file_path}")
    if path.suffix.lower() != ".py":
        raise ValueError("Rules files must be Python (.py) files.")

    spec = spec_from_file_location(f"_rules_{variable_name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load rules file: {file_path}")

    rules_module = module_from_spec(spec)
    spec.loader.exec_module(rules_module)

    rules = getattr(rules_module, variable_name, None)
    if not isinstance(rules, dict):
        raise ValueError(
            f"Rules file must define a dictionary named '{variable_name}'."
        )

    return rules
