import json
import glob

files = glob.glob("../data/squad/*.json")
print(files)
for f in files:
    print("processing: {}".format(f))
    parsed = json.load(open(f, 'r'))
    with open(f + ".txt", 'w') as p:
        p.write(json.dumps(parsed, indent=2))
