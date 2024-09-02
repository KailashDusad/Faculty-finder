import csv
import json

# Paths to your files
csv_file_path = './faculty pictures.csv'  
json_file_path = './csvjson.json'  
output_json_path = './csvjson.json'  

# Load the JSON data with utf-8 encoding
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# Load the CSV data into a dictionary
csv_data = {}
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        name = row['Name']
        profile_link = row['Profile link']
        profile_picture = row['Profile Picture']
        csv_data[name] = {'Profile link': profile_link, 'Profile Picture': profile_picture}

# Update the JSON data with the information from the CSV
for entry in json_data:
    name = entry['Name']
    if name in csv_data:
        entry['Profile link'] = csv_data[name]['Profile link']
        entry['Profile Picture'] = csv_data[name]['Profile Picture']

# Save the updated JSON data with utf-8 encoding
with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
    json.dump(json_data, output_json_file, indent=4)

print(f"Updated JSON file saved to: {output_json_path}")
