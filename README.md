@Research question
"Which attributes describing the levels of remoteness, education,
disability and elderly population in a Victorian community reveal
the most about its household income levels?"

@Project overview
This project provides python code for:
- preprocess community data using data imputation, reshaping, remove duplicates...
- computing statistics for individual features
- compute Pearson correlation, normalized mutual information
- plotting histograms, and heatmaps for analysis of correlations and data distribution
- perform linear regression model training for the communities and plot corresponding
    scatter plot
- perform k-means clustering for the communities and plot corresponding diagram for
    elbow method as well as scatter plots showing the relationship between different
    features of communities
- perform PCA and k-means clustering for the communities
- perform KNN classification model training and plot corresponding confusion matrix
- compute the percision, recall, f1 score for the KNN model

@Project structure

                             |---| correlation_analysis |---| correlation_and_mu.py      # compute correlation and NMI
         |---| data_analysis |
         |                   |                               |---| histogram_plot.py     # plot histograms for features
         |                   |---| exploratory_data_analysis |
         |                                                   |---| statistics_table.py   # compute descriptive statistics
Root |---|                       
         |                      |---| Kmean_scatter_plot.py          # kmeans scatter plot visualization
         |                      |
         |                      |---| Kmean_with_PCA.py              # kmeans clustering with PCA dimnensionality reduction
         |---| machine_learning |
         |                      |---| KNN_Classification_Model.py    # KNN model training
         |                      |
         |                      |---| linear_regression.py           # linear regression model training
         |
         |---| communities_modified.csv      # the preprocessed communities data file
         |
         |---| communities.csv               # raw data for communities
         |
         |---| data_preprocess.py            # perform data preprocess (excluding normalization)
         |
         |---| run_all_tasks.py              # run all the .py files

@Usage
Require installation of Pandas, Sklearn, Numpy, Matplotlib, Seaborn.
- Change directory to the source folder.
- Refer to project structure to run certain part of the project.
- Run 'run_all_tasks.py' to obtain output from all parts of the project.
- Output diagrams will either appear in windows or as png files in the
    source folder.

@Contributors:
Rex Kelly 1449348
Yi Ding 1449054
Qinglin Huang 1616895
Huiming Zhang 1411831
