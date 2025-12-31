#!/usr/bin/env python3
import json
from pathlib import Path

p = Path('src/column_headers.json')
with p.open('r', encoding='utf-8') as f:
    data = json.load(f)

name_index = {item['name']: i for i, item in enumerate(data)}
new_items = []
changed = False
for i, item in enumerate(list(data)):
    if 'ohe' in item and isinstance(item['ohe'], dict):
        source = item['name']
        # skip if ordinal already exists
        ordinal_name = f"{source}_ordinal"
        if ordinal_name in name_index:
            continue
        # build labels mapping: map non-NA keys in order to 0.. and NA to -1
        keys = list(item['ohe'].keys())
        labels = {}
        code = 0
        for k in keys:
            if str(k) == 'NA':
                labels[k] = '-1'
            else:
                labels[k] = str(code)
                code += 1
        # attach labels mapping to source (if not present)
        if 'labels' not in item:
            item['labels'] = labels
            changed = True
        else:
            # merge, but don't overwrite
            for k, v in labels.items():
                if k not in item['labels']:
                    item['labels'][k] = v
                    changed = True
        # create derived ordinal column
        ordinal_item = {
            'name': ordinal_name,
            'categorical': 'True',
            'X': 'True',
            'Y': 'False',
            'labels_from': source
        }
        new_items.append((i+1, ordinal_item))

# insert new items after their sources
offset = 0
for idx, itm in new_items:
    data.insert(idx + offset, itm)
    offset += 1

if changed or new_items:
    bk = p.with_suffix('.json.bak')
    p.rename(bk)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated schema written to {p}, backup at {bk}")
else:
    print('No changes needed')
