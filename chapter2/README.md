# General

- 使用 independent variable 來預測 dependent variable
- 在 Data.csv 中，前面三個 column 就是 independent variable，要預測的就是最後一個 column，也就是 dependent variable

# Missing Data
- Two missing data in the Data.csv
- 最常見的做法之一是用*平均值*來填入 missing data

# Encoding Categorical Data
- 將 categorical data 轉換成數值型資料
- sklearn.preprocessing 中的 [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 可以將 categorical data 轉成數值資料
- 要注意，[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 會將資料轉換成 0 ~ n_classes-1 的資料，在某些時候，這會造成資料解釋上的問題。比如說，這會告訴 ML 說，2 > 1，或 1 > 0，但是在 Data.csv 資料中，這樣解釋並不合理（法國 > 希臘?)
- 為了解決這個問題，我們要用 *Dummy Variable* 表示法。
- 透過 [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)，將 Country 這個 feature 轉換成三個 features （因為我們共有三個城市），分別用 1 或 0 來代表這筆資料有沒有該城市

![image](https://github.com/kevingo/ml-az/raw/29675d098cae250f159f90df1768857c9c7963ad/screenshots/onehotencoder.png)

# Split Data into Train / Test Dataset

# Feature Scaling
- 由於 Salary 和 Age 這兩個 feature 的 scale 差很多，許多 ML 會使用 euclidean distance 當作距離公式，這會造成 Salary 會主導結果
- 我們需要做一個 nomalization 或 standardization 讓兩個 features 的結果在同一個 scale















