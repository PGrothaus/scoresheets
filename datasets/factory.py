import sys
from datasets.build import build_move_section_only_dataset


def build(key):
    build_methods = {
        "move-section-only": build_move_section_only_dataset,
    }
    method = build_methods.get(key)
    if method:
        return method()
    else:
        print(key)
        print("No build method defined for this key.")
        return None


if "__main__" == __name__:
    key = sys.argv[1]
    build(key)
