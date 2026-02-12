import os

folder = "images"

files = sorted(os.listdir(folder))

i = 1
for filename in files:
    old_path = os.path.join(folder, filename)

    if not os.path.isfile(old_path):
        continue

    name, ext = os.path.splitext(filename)
    new_name = f"{i}{ext}"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)
    i += 1