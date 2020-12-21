
## Datasets
Vowel Database by James Hillenbrand (https://homepages.wmich.edu/~hillenbr/voweldata.html)
## File Structure
```bash
.
├── README.md 
├── main.ipynb                               # Contains all main code segments for loading, pre-processing, and using data for training classifiers
├── data_processing.py                       # Contains functions for pre-processing data and getting WPD coefficients
├── extra_functions.py                       # Contains functions for PCA and for simplifying k-fold cross validation
├── txt files                                # contain data and metadata for dataset. processed with functions from data_processing.py
├── .gitignore
└── 
```
## How to run the code
main.py consists of code segments that can be used to train and perform different types of classification. Wrapper and assisting functions are included in data_processing and extra_functions. Do not simply run main.py in console. Instead, use the simple code blocks within it to construct whatever program you may need. Ideally just run lines from console.
