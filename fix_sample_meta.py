# fix_sample_meta.py
# Purpose: Fix sample_meta.json — remove Windows EDF paths completely
# App will auto-download EDF via MNE on Hugging Face

import json
import os

meta_path = "D:/cognitive/samples/sample_meta.json"

with open(meta_path, 'r') as f:
    meta = json.load(f)

for item in meta:
    subject = item['subject']
    # Fix image path to relative
    item['img_path'] = f"samples/subject_{subject}_eeg.png"
    # Remove Windows EDF path — app will auto-download
    item['edf_path'] = ""

with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print("Fixed sample_meta.json:")
for item in meta:
    print(f"  Subject {item['subject']}: img={item['img_path']} | edf=auto-download")

