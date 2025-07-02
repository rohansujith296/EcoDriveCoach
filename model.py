import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle





ev_data_df = pd.read_csv('/Users/rohansujith/Desktop/Python/EcoDriveCoach/docs/ev_driving_style_dataset.csv')
ice_data_df = pd.read_csv('/Users/rohansujith/Desktop/Python/EcoDriveCoach/docs/simulated_driving_style_dataset.csv')


def split_dataset(df, train_frac=0.6, val_frac=0.2, random_state=42):
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(df)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val
    return (
        df.iloc[:n_train],
        df.iloc[n_train:n_train + n_val],
        df.iloc[n_train + n_val:]
    )

train_ev_df, val_ev_df, test_ev_df = split_dataset(ev_data_df)
train_ice_df, val_ice_df, test_ice_df = split_dataset(ice_data_df)


target_col_ev = 'driving_style_label'
target_col_ice = 'driving_style'

# EV input sets
X_train_ev = train_ev_df.drop(columns=[target_col_ev])
X_val_ev = val_ev_df.drop(columns=[target_col_ev])
X_test_ev = test_ev_df.drop(columns=[target_col_ev])
#ev target sets
y_train_ev = train_ev_df[target_col_ev]
y_val_ev = val_ev_df[target_col_ev]
y_test_ev = test_ev_df[target_col_ev]

# ICE  input sets
X_train_ice = train_ice_df.drop(columns=[target_col_ice])
X_val_ice = val_ice_df.drop(columns=[target_col_ice])
X_test_ice = test_ice_df.drop(columns=[target_col_ice])
# ICE target sets
y_train_ice = train_ice_df[target_col_ice]
y_val_ice = val_ice_df[target_col_ice]
y_test_ice = test_ice_df[target_col_ice]


n_col_ev = X_train_ev.select_dtypes(include=[np.number]).columns.tolist()
n_col_ice = X_train_ice.select_dtypes(include=[np.number]).columns.tolist()

c_col_ev = X_train_ev.select_dtypes(exclude=[np.number]).columns.tolist()
c_col_ice = X_train_ice.select_dtypes(include='object').columns.tolist()

scaler = MinMaxScaler().fit(ev_data_df[n_col_ev])
X_train_ev[n_col_ev] = scaler.transform(X_train_ev[n_col_ev])
X_val_ev[n_col_ev] = scaler.transform(X_val_ev[n_col_ev])
X_test_ev[n_col_ev] = scaler.transform(X_test_ev[n_col_ev])
scaler = MinMaxScaler().fit(ice_data_df[n_col_ice])
X_train_ice[n_col_ice] = scaler.transform(X_train_ice[n_col_ice])   
X_val_ice[n_col_ice] = scaler.transform(X_val_ice[n_col_ice])
X_test_ice[n_col_ice] = scaler.transform(X_test_ice[n_col_ice]) 

encoder_ev = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_ev_df[c_col_ev])
encoded_col_ev = list(encoder_ev.get_feature_names_out(c_col_ev))
X_train_ev[encoded_col_ev] = encoder_ev.transform(X_train_ev[c_col_ev])
X_val_ev[encoded_col_ev] = encoder_ev.transform(X_val_ev[c_col_ev])
X_test_ev[encoded_col_ev] = encoder_ev.transform(X_test_ev[c_col_ev])
encoder_ice = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_ice_df[c_col_ice])
encoded_col_ice = list(encoder_ice.get_feature_names_out(c_col_ice))
X_train_ice[encoded_col_ice] = encoder_ice.transform(X_train_ice[c_col_ice])
X_val_ice[encoded_col_ice] = encoder_ice.transform(X_val_ice[c_col_ice])
X_test_ice[encoded_col_ice] = encoder_ice.transform(X_test_ice[c_col_ice])


X_train_ev = X_train_ev.drop(columns=c_col_ev)
X_val_ev = X_val_ev.drop(columns=c_col_ev)
X_test_ev = X_test_ev.drop(columns=c_col_ev)
X_train_ice = X_train_ice.drop(columns=c_col_ice)
X_val_ice = X_val_ice.drop(columns=c_col_ice)
X_test_ice = X_test_ice.drop(columns=c_col_ice) 

model_ev = RandomForestClassifier(random_state=30, max_depth=80,min_samples_split=8, n_estimators=50, max_features='sqrt', class_weight='balanced')
model_ice = RandomForestClassifier(random_state=42, max_depth=60,min_samples_split=8, n_estimators=50, max_features='sqrt', class_weight='balanced')

model_ev.fit(X_train_ev, y_train_ev)
print("EV model trained successfully")
print(model_ev.score(X_train_ev, y_train_ev))
print(model_ev.score(X_val_ev, y_val_ev))
model_ice.fit(X_train_ice, y_train_ice)
print("ICE model trained successfully") 
print(model_ice.score(X_train_ice, y_train_ice))
print(model_ice.score(X_val_ice, y_val_ice))

print("TEST ev",model_ev.score(X_test_ev, y_test_ev))
print("TEST ice",model_ice.score(X_test_ice, y_test_ice))

with open('/Users/rohansujith/Desktop/Python/EcoDriveCoach/docs/model.pkl', 'wb') as f:
    pickle.dump((model_ev, model_ice), f)







print("success")
