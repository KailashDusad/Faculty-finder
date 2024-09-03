import json

with open('facultydetails.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('csvjson.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

info_map = {entry["Name"]: (entry["Office location"], entry["Email ID"]) for entry in data1}

for entry in data2:
    name = entry.get("Name")
    if name in info_map:
        entry["Office location"], entry["Email ID"] = info_map[name]

with open('csvjson.json', 'w', encoding='utf-8') as f2_updated:
    json.dump(data2, f2_updated, indent=4)

print("updated_json_file.json updated successfully.")
