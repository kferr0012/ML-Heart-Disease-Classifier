import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


#Import Data
df = pd.read_csv("./data/heart.csv")

#Check out the dataset
df.head()

#Encode the categorical values in the dataframe
le_Sex = LabelEncoder()
le_Chest_Pain_Type = LabelEncoder()
le_Resting_ECG = LabelEncoder()
le_Exercise_Angina = LabelEncoder()
le_ST_Slope = LabelEncoder()

df["Sex"] = le_Sex.fit_transform(df["Sex"])
df["ChestPainType"] = le_Chest_Pain_Type.fit_transform(df["ChestPainType"])
df["RestingECG"] = le_Resting_ECG.fit_transform(df["RestingECG"])
df["ExerciseAngina"] = le_Exercise_Angina.fit_transform(df["ExerciseAngina"])
df["ST_Slope"] = le_ST_Slope.fit_transform(df["ST_Slope"])

#Make sure encoding worked
df.head()


#Split label data from feature data
X = df.iloc[:,:-1].values
y = df.iloc[:,11].values


#Split data into training and testing data with a 70-30 split; 70% for training , 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)


#Apply Feature Scaling to preprocess the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Train the Machine Learning Model , this will be a KNN model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)


#Let's test out our model
y_pred = classifier.predict(X_test)


#Let's print out a summary of how well it did
print(classification_report(y_test,y_pred))

