import pandas as pd

data = pd.read_csv("mergedData.csv")

# split the final merged dataset into towns and suburbs for easier analysis
towns = data.loc[data["Community Name"].str.endswith(" (Town)")]
suburbs = data.loc[data["Community Name"].str.endswith(" (Suburb)")]

# remove the classifying tag (saying Town or Suburb) from the community name as it is implied from the file name (once saved to csv)
towns["Community Name"] = towns["Community Name"].apply(lambda name: name.replace(" (Town)", ""))
suburbs["Community Name"] = suburbs["Community Name"].apply(lambda name: name.replace(" (Suburb)", ""))

# save the towns and suburbs separately
towns.to_csv("townsData.csv", index = False)
suburbs.to_csv("suburbsData.csv", index = False)
