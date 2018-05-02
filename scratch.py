import numpy as np
import Stat_Fcns
import pandas as pd


DATA_PATH = "L:\Excel Sheets\MBS\MBS_PCA_Data.xlsx"
TAB_NAME = "Sheet3"
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)

pca = Stat_Fcns.PCAAnalysis(data)
pca.getPCA(n_components=4)
components = pca.getComponents()
print(components)