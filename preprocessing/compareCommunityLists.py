import pandas as pd

communitiesFile = pd.read_csv("Data/communities.csv")
communitiesFile = communitiesFile.drop_duplicates()
print(len(communitiesFile))

crimesFile = pd.read_csv("crimeSums2014.csv", index_col = [0]).squeeze()
# fix rosebud west vs capel sound
crimesFile = crimesFile.rename(index = {"Capel Sound": "Rosebud West"})





towns = communitiesFile.loc[communitiesFile["Community Name"].str.endswith(" (Town)")]
suburbs = communitiesFile.loc[communitiesFile["Community Name"].str.endswith(" (Suburb)")]
# catchments = communitiesFile.loc[communitiesFile["Community Name"].str.endswith(" (Catchment)")]

towns = towns["Community Name"].str.replace(" (Town)", "")
suburbs = suburbs["Community Name"].str.replace(" (Suburb)", "")

# will need to work with a list of all towns and suburbs (not catchments)
allFromCommunities = pd.concat([towns, suburbs], ignore_index = True).sort_values()
# allFromCommunities.to_csv("allFromCommunities.csv", index = False)



# fix the places that are in Communities but not in Crimes



# add zeroes for the locations with any crime in any later years
# it is assumed that there were 0 crimes to record in 2014 for these places
# because other years may have had 1, a few, or no records of crime
zeroes = pd.Series([0, 0, 0], index = ["Mount Hotham", "Woodford", "Falls Creek"])
crimesFile = pd.concat([crimesFile, zeroes])



# the community data lists some communities with hyphenated names
# their crime data is recorded under two separate names
# add their crimes together under a new hyphenated name that matches the name given with the community data
compositeCommunities = allFromCommunities[allFromCommunities.str.contains(" - ")]
combinedToAdd = {}

for item in compositeCommunities:
    both = item.split(" - ")

    if both[0] in crimesFile.index and both[1] in crimesFile.index:
        # replace the places in the crime sums with their combined name and combined sums
        mergedOffenceCounts = crimesFile.loc[both[0]] + crimesFile.loc[both[1]]
        combinedToAdd[item] = mergedOffenceCounts


    else:
        print(item + " doesn't work :(")




seriesToAdd = pd.Series(combinedToAdd)
crimesFile = pd.concat([crimesFile, seriesToAdd])


# delete these four records of towns that have data about the community
# because they have no evidence of crime statistics being collected
dropping = ["Arcadia Downs", "Glenrowan North", "Hazeldene", "Wonga Park - South"]
allFromCommunities = pd.Series([item for item in allFromCommunities if not item in dropping])



# btw locations only listed in the crimes data were ignored



# code used for detecting communities with names that don't appear in the crime data
foundSuburbs = 0
foundTowns = 0

unfoundSuburbs = pd.Series()
unfoundTowns = pd.Series()

for item in towns:
    if item in crimesFile.index:
        foundTowns += 1
    else:
        unfoundSuburbs.loc[len(unfoundSuburbs)] = item
        print("Can't find: " + item)

for item in suburbs:
    if item in crimesFile.index:
        foundSuburbs += 1
    else:
        unfoundSuburbs.loc[len(unfoundSuburbs)] = item

print("Found " + str(foundTowns) + " towns but didn't find " + str(len(towns) - foundTowns))
print("Found " + str(foundSuburbs) + " suburbs but didn't find " + str(len(suburbs) - foundSuburbs))



foundPlaces = 0
for item in allFromCommunities:
    if item in crimesFile.index:
        foundPlaces += 1

print("Found " + str(foundPlaces) + " locations but didn't find " + str(len(allFromCommunities) - foundPlaces))


# save the names of all communities to which a column with 2014 offence count can be added
# not necessary (hence commented out), used for reference during development
# allFromCommunities.to_csv("locationsInBoth.csv", index = False, header = False)


# in a new file, save the crime data for only the usable locations
crimesFile = crimesFile.sort_index()
keeping = []
for item in crimesFile.index:
    if item in allFromCommunities.values:
        keeping.append(item)
crimesFile = crimesFile.loc[keeping]
crimesFile.to_csv("crimesToMerge.csv", index = True, header = False)


