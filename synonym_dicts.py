from utils import collect_circrna_names, get_circfunbase_disease
import pandas as pd

# CircR2Disease
CircR2Disease = pd.read_excel(
    "Datasets/raw/CircR2Disease_circRNA-disease associations.xlsx"
)
CircR2Disease = CircR2Disease[CircR2Disease.Species == "Human"]
CircR2Disease = collect_circrna_names(
    CircR2Disease, ["circRNA Name", "Region"], delimiter="/"
)
CircR2Disease["Source"] = "CircR2Disease"
CircR2Disease["Disease Name"] = CircR2Disease["Disease Name"].apply(
    lambda x: x.strip().lower()
)
CircR2Disease["disease"] = CircR2Disease["Disease Name"]

# CircBase
circbase_mappings = pd.read_csv(
    "Datasets/raw/circbase_circID_to_name.txt", delimiter="\t"
)
circbase_mappings = collect_circrna_names(
    circbase_mappings, ["circID", "name"], delimiter=","
)

# Circ2Disease
Circ2Disease = pd.read_excel("Datasets/raw/Circ2Disease_Association.xlsx", index_col=0)
Circ2Disease = collect_circrna_names(
    Circ2Disease, ["circRNA Name", "Synonyms", "Region"], delimiter="; "
)
Circ2Disease["Source"] = "Circ2Disease"
Circ2Disease["Disease Name"] = Circ2Disease["Disease Name"].apply(
    lambda x: x.strip().lower()
)

# circRNADisease
circRNADisease = pd.read_excel("Datasets/raw/circRNADisease.xlsx", index_col=0)
circRNADisease.columns = [
    "Title",
    "Journal",
    "Year",
    "PMID",
    "circRNA id",
    "circRNA name",
    "Synonyms",
    "Disease Name",
    "method of circRNA detection",
    "Species",
    "Expression Pattern",
    "associator",
    "Gene Symbol",
    "tissue/cell line",
    "Functional Description",
]
circRNADisease = collect_circrna_names(
    circRNADisease, ["circRNA id", "circRNA name", "Synonyms"], delimiter=","
)
circRNADisease["Source"] = "circRNADisease"
circRNADisease["Disease Name"] = circRNADisease["Disease Name"].apply(
    lambda x: x.strip().lower()
)

# CircFunBase
CircFunBase = pd.read_csv(
    "Datasets/raw/CircFunBase_Homo_sapiens_circ.txt", delimiter="\t"
)
columns = CircFunBase.columns.copy()
new_rows = []
for ind, row in CircFunBase.iterrows():
    text = row["Function"]
    if "[" not in text:
        continue
    diseases = [i.split(" [")[0].lower() for i in text.split("; ") if "[" in i]
    for d in diseases:
        new_rows.append(
            [
                row["circRNA"],
                row["Position"],
                row["Function"],
                row["Gene"],
                row["Gene Description"],
                row["PMID"],
                d.lower().strip(),
            ]
        )

CircFunBase = pd.DataFrame(new_rows, columns=columns)
CircFunBase = collect_circrna_names(CircFunBase, ["circRNA", "Position"], delimiter=",")
CircFunBase["Source"] = "CircFunBase"

# HMDD
HMDD = pd.read_excel("Datasets/raw/HMDD miRNA-disease association data.xlsx")

# CircBank
circbase = pd.read_csv("Datasets/raw/circbank_miRNA_all_v1.txt", delimiter="\t")
circbase["all"] = circbase["circbase_ID"]
