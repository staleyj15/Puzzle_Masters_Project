## Puzzle Masters Project
##### _Authors: Jessica Staley and Henry Staley_
<br>

This repository contains the following directories and files:

* _Puzzle_Masters_Project_Code.ipynb_
* _Puzzle_Masters_ISyE6740_Project.pdf_
* _requirements.txt_
* _Funcs4Testing.py_
* _ImgCluster.py_
* _ImgPrep.py_
* AdjacencyMatrices
* pickle_files
* plots
* puzzle_scans

The two main deliverables inside of this repository are <span style="color:#2b4e94">**_Puzzle_Masters_Project_Code.ipynb_**</span> and <span style="color:#2b4e94">**_Puzzle_Masters_ISyE6740_Project.pdf_**</span>. These two files contain the summary of our analysis and the code supporting it. 

To select a puzzle and run the code, set the `puzzle_folder` variable at the top of the _Puzzle_Masters_Project_Code.ipynb_ file. Valid values include "puzzle_1", "puzzle_2", or "puzzle_3". 

The `load_from_pickle` variable can be used to speed up script execution. If set to True, the script will skip data preprocessing and load preprocessed feature set from pickle file. 

The _requirements.txt_ file contains the list of package dependencies that are necessary to install inside of a virtual environment in order to run _Puzzle_Masters_Project_Code.ipynb_.

A description of the remaining files and directories are include in the table below:
<br>

|File/Directory|Description|
|--|--|
|_Funcs4Testing.py_|Python library containing functions to test the cluster models' performance and display images.|
|_ImgCluster.py_|Python library containing functions to build and run cluster models.|
|_ImgPrep.py_|Python library containing functions to preprocess data for clustering.|
|AdjacencyMatrices|Adjacency matrix for each puzzle stored in csv format. These adjacency matrices are used to measure the cluster models' performance.|
|pickle_files|Pickle files of feature sets. These files can be imported and fed directly into the clustering model in order to skip preprocessing step and speed up script execution.|
|plots|Images and graphs generated from _Puzzle_Masters_Project_Code.ipynb_ file.|
|puzzle_scans|Data source containing subdirectory of images for each puzzle.|

<br>
<br>

The project can be found [here](https://github.com/staleyj15/isye6740_project.git) on GitHub.