# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import seaborn as sns



st.title("Breast Cancer")
section = st.sidebar.radio("Go to", ["Introduction","DASHBOARD","EDA", "MODEL","RESULT"])
if section == "Introduction":
    st.write(" Breast cancer is a cancer that develops from breast tissue.[7] Signs of breast cancer may include a lump in the breast, a change in breast shape, dimpling of the skin, milk rejection, fluid coming from the nipple, a newly inverted nipple, or a red or scaly patch of skin.[1] In those with distant spread of the disease, there may be bone pain, swollen lymph nodes, shortness of breath, or yellow skin.[8] Risk factors for developing breast cancer include obesity, a lack of physical exercise, alcohol consumption, hormone replacement therapy during menopause, ionizing radiation, an early age at first menstruation, having children late in life (or not at all), older age, having a prior history of breast cancer, and a family history of breast cancer.[1][2][9] About five to ten percent of cases are the result of an inherited genetic predisposition,[1] including BRCA mutations among others.[1] Breast cancer most commonly develops in cells from the lining of milk ducts and the lobules that supply these ducts with milk.[1] Cancers developing from the ducts are known as ductal carcinomas, while those developing from lobules are known as lobular carcinomas.[1] There are more than 18 other sub-types of breast cancer.[2] Some, such as ductal carcinoma in situ, develop from pre-invasive lesions.[2] The diagnosis of breast cancer is confirmed by taking a biopsy of the concerning tissue.[1] Once the diagnosis is made, further tests are carried out to determine if the cancer has spread beyond the breast and which treatments are most likely to be effective.[1] Breast cancer screening can be instrumental, given that the size of a breast cancer and its spread are among the most critical factors in predicting the prognosis of the disease. Breast cancers found during screening are typically smaller and less likely to have spread outside the breast.[10] Training health workers to do clinical breast examination may have potential to detect breast cancer at an early stage.[11] A 2013 Cochrane review found that it was unclear whether mammographic screening does more harm than good, in that a large proportion of women who test positive turn out not to have the disease.[12] A 2009 review for the US Preventive Services Task Force found evidence of benefit in those 40 to 70 years of age,[13] and the organization recommends screening every two years in women 50 to 74 years of age.[14] The medications tamoxifen or raloxifene may be used in an effort to prevent breast cancer in those who are at high risk of developing it.[2] Surgical removal of both breasts is another preventive measure in some high risk women.[2] In those who have been diagnosed with cancer, a number of treatments may be used, including surgery, radiation therapy, chemotherapy, hormonal therapy, and targeted therapy.[1] Types of surgery vary from breast-conserving surgery to mastectomy.[15][16] Breast reconstruction may take place at the time of surgery or at a later date.[16] In those in whom the cancer has spread to other parts of the body, treatments are mostly aimed at improving quality of life and comfort.")



if section == "DASHBOARD":
    df = pd.read_csv("Breast_Cancer.csv")
    st.subheader("🎛️ Filter Options")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        col1, col2 = st.columns([1, 2])  # create layout
        with col1:
            filter_col = st.selectbox("Choose a column to filter:", cat_cols, key="filter_col")
            with col2:
                filter_vals = st.multiselect("Select values:", df[filter_col].unique(), key="filter_vals")
                if filter_vals:
                    df = df[df[filter_col].isin(filter_vals)]
                    st.subheader("📊 Dataset Preview")
                    st.dataframe(df.head())
                    st.subheader("📈 Key Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Columns", df.shape[1])
                    col3.metric("Missing Values", df.isnull().sum().sum())
                    st.subheader("📌 Summary Statistics")
                    st.write(df.describe())
                    st.subheader("📊 Feature Distribution")
                    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    if num_cols:
                        feature = st.selectbox("Select numeric feature:", num_cols, key="dist_feature")
                        fig, ax = plt.subplots()
                        df[feature].hist(ax=ax, bins=20, color="skyblue", edgecolor="black")
                        ax.set_title(f"Distribution of {feature}")
                        st.pyplot(fig)
                        st.subheader("📉 Correlation Heatmap")
                        if len(num_cols) > 1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                            st.pyplot(fig) 

                            

if section == "MODEL":
    df = pd.read_csv("Breast_Cancer.csv")


    st.subheader("Dataset Preview")
    st.write(df.head())

    # Identify target column
    possible_targets = ['diagnosis', 'target', 'class', 'label', 'y', 'A Stage']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        for c in df.columns:
            if df[c].nunique() == 2:
                target_col = c
                break
    if target_col is None:
        target_col = df.columns[-1]

    st.write(f"**Detected Target Column:** {target_col}")

    # Preprocess
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median())

    # Train/val/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def get_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    # Deep Learning (Keras)
    st.subheader("Training Deep Learning Model (Keras)")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    with st.spinner("Training deep learning model..."):
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )

    y_test_proba_dl = model.predict(X_test_scaled).ravel()
    y_test_pred_dl = (y_test_proba_dl >= 0.5).astype(int)
    dl_test_results = {"model": "DeepLearning_Keras (Test)", **get_metrics(y_test, y_test_pred_dl, y_test_proba_dl)}

    st.write("### Deep Learning Model Test Performance")
    st.write(f"**Accuracy:** {dl_test_results['accuracy']:.4f}")
    st.write(f"**Precision:** {dl_test_results['precision']:.4f}")
    st.write(f"**Recall:** {dl_test_results['recall']:.4f}")
    st.write(f"**F1-score:** {dl_test_results['f1']:.4f}")
    st.write(f"**ROC AUC:** {dl_test_results['roc_auc']:.4f}")
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(dl_test_results["confusion_matrix"],
                          index=["Actual Neg", "Actual Pos"],
                          columns=["Predicted Neg", "Predicted Pos"]))

    # Training curves
    st.subheader("Training and Validation Accuracy Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Training and Validation Loss Curve")
    fig1b, ax1b = plt.subplots()
    ax1b.plot(history.history['loss'], label='Train Loss')
    ax1b.plot(history.history['val_loss'], label='Val Loss')
    ax1b.set_xlabel('Epoch')
    ax1b.set_ylabel('Loss')
    ax1b.legend()
    ax1b.grid(True)
    st.pyplot(fig1b)

    # Traditional ML models
    st.subheader("Training Traditional Machine Learning Models")
    ml_models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    ml_test_results = []
    progress_bar = st.progress(0)
    for i, (name, clf) in enumerate(ml_models.items()):
        st.write(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        ml_test_results.append({"model": name + " (Test)", **get_metrics(y_test, y_test_pred, y_test_proba)})
        progress_bar.progress((i + 1) / len(ml_models))

    # Combine results
    all_results = [dl_test_results] + ml_test_results

    # Save to CSV
    csv_path = "breast_cancer_model_comparison_results.csv"
    if os.path.exists(csv_path):
        df_results = pd.read_csv(csv_path)
        df_results = pd.concat([df_results, pd.DataFrame(all_results)], ignore_index=True)
    else:
        df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_path, index=False)
    st.success(f"Results saved to {csv_path}")

    # Show results
    st.subheader("Model Performance Comparison Table (Test Set)")
    st.dataframe(df_results.style.format({
        "accuracy": "{:.4f}",
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1": "{:.4f}",
        "roc_auc": "{:.4f}"
    }))

    # Bar chart
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = pd.DataFrame(all_results)
    st.subheader("Model Metrics Comparison - Bar Chart (Test Set)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    width = 0.12
    x = np.arange(len(df_plot["model"]))
    for i, metric in enumerate(metrics_to_plot):
        ax2.bar(x + i * width, df_plot[metric], width=width, label=metric)

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(df_plot["model"], rotation=45, ha='right')
    ax2.set_ylabel("Score")
    ax2.set_title("Model Metrics Comparison (Test Set)")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)


if section == "RESULT":
    df = pd.read_csv("Breast_Cancer.csv")
    df.columns = df.columns.str.strip()
    if "Reginol Node Positive" in df.columns:
        df.rename(columns={"Reginol Node Positive": "Regional Node Positive"}, inplace=True)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    if df["Status"].dtype == "object":
        le_status = LabelEncoder()
        df["Status"] = le_status.fit_transform(df["Status"].astype(str))
                
    else:
        le_status = None
    X = df.drop("Status", axis=1)
    y = df["Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Deep Learning (Keras)": "keras_model"   # Placeholder
        }
    st.title("Breast Cancer Survival Prediction")
    st.write("Select a model and input patient features to predict **Alive / Dead** status.")
    model_choice = st.selectbox("Choose a model:", list(models.keys()))
    if model_choice == "Deep Learning (Keras)":
        st.subheader("Training Deep Learning Model (Keras)")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        with st.spinner("Training deep learning model..."):
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[es],
                verbose=0
        )

    else:
        model = models[model_choice]
        model.fit(X_train_scaled, y_train)
        
    user_input = {}
    for col in X.columns:
        if col in categorical_cols:
            options = label_encoders[col].classes_.tolist()
            choice = st.selectbox(f"Select {col}", options)
            user_input[col] = label_encoders[col].transform([choice])[0]
        else:
            val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            user_input[col] = val

    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)

# --- Prediction ---
    if model_choice == "Deep Learning (Keras)":
        proba = model.predict(user_scaled)[0][0]
        prediction = 1 if proba >= 0.5 else 0
    else:
        prediction = model.predict(user_scaled)[0]
        proba = model.predict_proba(user_scaled)[0][prediction]
        
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Status: Alive (Confidence: {proba*100:.2f}%) using {model_choice}")
    
    else:
        st.error(f"⚠️ Status: Dead (Confidence: {proba*100:.2f}%) using {model_choice}")

else:
    st.info("Please upload a CSV file to get started.")


