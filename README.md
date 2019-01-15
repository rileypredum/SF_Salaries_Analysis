# SF_Salaries_Analysis
An analysis of the classic SF salaries dataset with a few ML algos examined. I looked at the data with linear and logistic regression, a decision tree classifier, random forest classifier, and finally a KMeans classifier. The best performer was the random forest classifier. Next steps would be to try an SVM classifier, but I might need AWS for that as that dataframe is a little large for my computer.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The import statements throughout the notebook:

```
import numpy as np
import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
```

### Installing

Have the .db file in your working directory and it should all work assuming you have pip installed the above packages.

## Author

* **Riley Predum** - [RileyPredum](https://github.com/rileypredum)
