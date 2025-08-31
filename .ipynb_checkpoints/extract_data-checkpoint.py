import json

file_path = "sky_conditions_20240701_20250830.json"

with open(file_path, 'r') as f:
	data = json.load(f)

print(type(data))

print(f'isinstance(data, list): {isinstance(data, list)}')
print(f'isinstance(data, dict): {isinstance(data, dict)}')
print(list(data.keys()))
print(data[message])