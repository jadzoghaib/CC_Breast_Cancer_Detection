
import pandas as pd
import boto3
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# 1. UPDATE THESE WITH YOUR ACTUAL BUCKET NAMES
SOURCE_BUCKET = 'breast-cancer-cleaneddata' 
MODEL_BUCKET = 'breast-cancer-prediction-models'
DATA_FILE = 'clean-data.csv'
LOCAL_DATA_PATH = 'data/clean-data.csv'

# 1. Load Data
if not os.path.exists(LOCAL_DATA_PATH):
    os.makedirs('data', exist_ok=True)
    print(f"Downloading {DATA_FILE} from S3...")
    try:
        s3 = boto3.client('s3')
        s3.download_file(SOURCE_BUCKET, DATA_FILE, LOCAL_DATA_PATH)
        print("Download successful.")
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        print("Make sure you updated SOURCE_BUCKET and that your EC2 role has S3 permissions.")
        raise

data = pd.read_csv(LOCAL_DATA_PATH)

# 2. Preprocessing (Cleaning)
if 'Unnamed: 0' in data.columns: data.drop('Unnamed: 0', axis=1, inplace=True)
if 'id' in data.columns: data.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in data.columns: data.drop('Unnamed: 32', axis=1, inplace=True)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode Target
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Define Pipeline (StandardScaler -> PCA -> SVM)
pipeline = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(n_components=2)),
    # Best params from Notebook (C=0.1, gamma=0.001, kernel='linear')
    ('clf', SVC(C=0.1, gamma=0.001, kernel='linear', probability=True)) 
])

# 4. Train
print("Training model...")
pipeline.fit(X, y)
print("Training complete.")

# 5. Save and Upload
LOCAL_MODEL_PATH = 'model_v1.pkl'
joblib.dump(pipeline, LOCAL_MODEL_PATH)
print(f"Model saved locally to {LOCAL_MODEL_PATH}")

print("Uploading to S3...")
s3 = boto3.client('s3')
s3.upload_file(LOCAL_MODEL_PATH, MODEL_BUCKET, 'latest_model.pkl')
print(f"Success! Model uploaded to s3://{MODEL_BUCKET}/latest_model.pkl")