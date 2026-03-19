import os
import json

files_to_fix = [
    'train.py',
    'lpclip/feat_extractor.py'
]

for file in files_to_fix:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('import datasets.', 'import dataset.')
    content = content.replace('from datasets.', 'from dataset.')
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

with open('colab_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i in range(len(source)):
            source[i] = source[i].replace('import datasets.', 'import dataset.')
        
        # Remove the hack
        new_source = []
        skip_next = False
        for line in source:
            if "current_dir =" in line or "sys.path.insert(0," in line or "sys.path.remove(" in line:
                continue
            if line.strip() == "if current_dir not in sys.path:" or line.strip() == "elif sys.path[0] != current_dir:":
                continue
            if line.strip() == "# Prioritize local modules over site-packages (solves pip `datasets` name clash)":
                continue
            new_source.append(line)
            
        # Clean up empty lines from removal mapping
        final_source = []
        empty_count = 0
        for line in new_source:
            if line == "\n":
                empty_count += 1
                if empty_count > 1:
                    continue
            else:
                empty_count = 0
            final_source.append(line)

        cell['source'] = final_source

with open('colab_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
