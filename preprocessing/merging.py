import pandas as pd

communitiesFile = pd.read_csv("Data/communities.csv")
communitiesFile = communitiesFile.drop_duplicates()

crimes = pd.read_csv("crimesToMerge.csv", header = None, index_col = 0)
crimes = crimes.squeeze()


# reduce communities back down to the final list (written in crimesToMerge.csv)
keeping = []
for index in communitiesFile.index:
    name = communitiesFile.loc[index, "Community Name"]
    if " (Catchment)" in name:
        continue
    name = name[:name.find(" (")]
    if name in crimes.index:
        keeping.append(index)
print("keeping " + str(len(keeping)))
communitiesFile = communitiesFile.loc[keeping]


# merge the datasets by adding the new crime column to the communities dataset
communitiesFile["Offence Count 2014"] = "filler"
# print(communitiesFile["Community Name"].tail())
for row in communitiesFile.index:
    name = communitiesFile.loc[row, "Community Name"]
    name = name[:name.find(" (")]
    numCrimes = crimes[name]
    communitiesFile.loc[row, "Offence Count 2014"] = numCrimes
communitiesFile.to_csv("mergedData.csv", index = False)
