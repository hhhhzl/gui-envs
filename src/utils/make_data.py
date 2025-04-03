import os
import json

path = '/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/GUI-VDILA/src/gui-envs/metadata/mapping'

for filename in os.listdir(path):
    if filename.endswith(".json"):
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)  # Load JSON content

            file.close()
        except json.JSONDecodeError as e:
            print(f"Error decoding {filename}: {e}")
        except Exception as e:
            print(f"Error opening {filename}: {e}")


def make_dataset():
    pass
