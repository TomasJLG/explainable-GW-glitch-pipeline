import csv
with open(r"C:\Users\tlegu\Desktop\M1- Detector_de_anomalias\gravityspy_o3\H1_O3a.csv") as f:
    reader = csv.DictReader(f)
    print("Todas las columnas:")
    for i, col in enumerate(reader.fieldnames):
        print(f"  {i}: {col}")