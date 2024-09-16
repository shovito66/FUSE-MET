# *Domain-Driven Label Fusion Surpasses LLMs in Free-Living  Activity Classification*

by
***Shovito Barua Soumma, Abdullah Mamun, and  Hassan Ghasemzadeh***
---

### _FUSE-MET is currently under review. The complete code will be made available at this repository after the acceptance of our submission. The reviewers have access to the source code in the supplementary materials of our submission._

```
pip3 freeze > requirements.txt
```


## Table of Contents
- [Abstract](#abstract)
- [Getting the Code](#getting-the-code)
- [Running the Code](#running-the-code)
  - [Requirements](#requirements)
  - [Setup Environment](#setup-environment)
  - [Reproducing the Results](#reproducing-the-results)


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://github.com/shovito66/FUSE-MET) repository:
  ```
    git clone git@github.com:shovito66/FUSE-MET.git
  ```
or [download a zip archive](https://github.com/shovito66/FUSE-MET/archive/master.zip).

# Running the code

-----
## Requirements
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `main.py`
is located) to create a separate environment and install all required
dependencies in it:
    
    conda env create

[//]: # (## Reproducing the results)
## Setup Environment
Before running any code you must activate the conda environment:
    
    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    activate ENVIRONMENT_NAME
**if you are on Windows/Linux:** To install the necessary dependencies, you can use the provided `requirements.txt` file. Run the following command:

    pip install -r requirements.txt
**if you are on Macos**: Use `requirements-macos.txt` file (inside `/code` directory)  to install the dependencies

    pip install -r requirements-macos.txt

----
## Reproducing the Results/ How To Run The Code
1. **Download the dataset**:
   Download the zip data file and unzip it into a folder named `data`. This `data` folder should be in the same location as the `code` directory.
   _Currently the dataset is not available for public use. Please contact the authors for access._

2. From the terminal provide the following command
    ```
    python main.py --task test_run --output_folder output --exp_name result_reproduce --override True --word_embedding bert
    ```
    * After running the above command, the output will be stored in the output folder with the name aaai-reproduce
3. Figure generation: To plot the results, run the following command
   ```
    python main.py --task plot_figures --output_folder output --exp_name result_reproduce --override True
   ```
4. Run the following command to generate baseline results (using all 36 distinct classes without any label fusion)
    ```
    python main.py --task baseline_no_cluster --output_folder output --exp_name result_reproduce --override True
    ```

# Contact
>For any questions or issues, please contact 
*  ***[Shovito Barua Soumma](https://www.shovitobarua.com)*** at [shovito@asu.edu](shovito@asu.edu) or 
*  ***[Abdullah Mamun](https://www.abdullah-mamun.com)*** at [a.mamun@asu.edu](a.mamun@asu.edu).


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
