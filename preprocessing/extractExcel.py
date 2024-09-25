import pandas as pd


data = pd.read_excel("Data/LGA Offences.xlsx", sheet_name = "Table 03")
data = data.loc[data["Year"] == 2014]
# data.to_csv("crimeAll2014.csv", index = False)


# tally a total offence count by location for 2014
output = data.groupby("Suburb/Town Name", as_index = False)["Offence Count"].sum()
output.to_csv("crimeSums2014.csv", index = False)