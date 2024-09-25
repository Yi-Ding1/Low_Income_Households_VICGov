The code in the python files was used to create multiple .csv files

The code uses two of the datasets provided for the assignment. Copies of that data are included in the 'Data' folder

External python libraries are used and may need to be installed prior to running the code. They include:
* pandas
* openpyxl

To test the code, delete any csv files in the 'preprocessing' directory, leaving only the 4 python files, the 'Data' directory and this 'README.md' file. **Do not delete the files in the 'Data' directory.** Then, run the python files in the following order:

1. extractExcel.py
2. compareCommunityLists.py
3. merging.py
4. splitByLocationType.py

Note: On my system, the python file takes close to 30 seconds to complete and the rest are much faster to run

The code should produce multiple csv files. The 3 important output files for data processing are:
* mergedData.csv - similar to communities.csv from the 'Data' folder but with some rows removed and a new column added: "Offence Count 2014"
* suburbsData.csv - mergedData.csv but only for rows describing suburbs
* townsData.csv - mergedData.csv but only for rows describing towns

Note: mergedData.csv has a the location type tag included in the "Community Name" column, whereas suburbsData.csv and townsData.csv have the tag removed, storing purely the name in that column