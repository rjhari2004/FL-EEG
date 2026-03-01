import streamlit as st
import pandas as pd
import json
import os
import glob
# Add this at the very top of your app.py script
if st.sidebar.button("🔄 Refresh Patient Data"):
    st.cache_data.clear()
    st.rerun()
st.set_page_config(page_title="Federated EEG Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data(ttl=5) # Refreshes every 5 seconds
def load_data():
    history = {}
    if os.path.exists("./runs/global_history.json"):
        with open("./runs/global_history.json", "r") as f:
            history = json.load(f)
            
    patient_files = glob.glob("./runs/patient_*.json")
    patients = {}
    for pf in patient_files:
        with open(pf, "r") as f:
            data = json.load(f)
            patients[data["pid"]] = data
    return history, patients

history, patients = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Leaf_icon_15.svg/1024px-Leaf_icon_15.svg.png", width=50)
st.sidebar.title("FL Navigation")
pages = ["Global Summary"] + sorted(list(patients.keys()))
selection = st.sidebar.radio("Go to", pages)

# --- PAGE 1: GLOBAL SUMMARY ---
if selection == "Global Summary":
    st.title("🌍 Federated Learning Global Network Summary")
    st.markdown("Live monitoring of the global model aggregation across all distributed hospital nodes.")
    
    if history and len(history["round"]) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Global Accuracy", f"{history['accuracy'][-1]*100:.2f}%")
        col2.metric("Final Global F1-Score", f"{history['f1'][-1]:.4f}")
        col3.metric("Total FL Rounds", history['round'][-1])
        
        st.subheader("Training Curves")
        df_hist = pd.DataFrame(history).set_index("round")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Global Accuracy & F1 over Rounds**")
            st.line_chart(df_hist[["accuracy", "f1"]])
        with c2:
            st.markdown("**Global Loss over Rounds (FedProx)**")
            st.line_chart(df_hist[["loss"]])
    else:
        st.info("Waiting for Federated Learning rounds to begin...")

    if patients:
        st.subheader("Average Edge Device Metrics (After Personalization)")
        avg_g_acc = sum(p["global_acc"] for p in patients.values()) / len(patients)
        avg_p_acc = sum(p["personal_acc"] for p in patients.values()) / len(patients)
        avg_boost = (avg_p_acc - avg_g_acc) * 100
        
        colA, colB, colC = st.columns(3)
        colA.metric("Avg Global Accuracy", f"{avg_g_acc*100:.2f}%")
        colB.metric("Avg Personalized Accuracy", f"{avg_p_acc*100:.2f}%", f"{avg_boost:+.2f}%")

# --- PAGE 2: INDIVIDUAL PATIENT REPORTS ---
else:
    pid = selection
    st.title(f"🏥 Patient Edge Device: {pid}")
    p_data = patients[pid]
    
    boost = (p_data['personal_acc'] - p_data['global_acc']) * 100
    st.metric(label="Personalization Boost", value=f"{p_data['personal_acc']*100:.2f}% Final Acc", delta=f"{boost:+.2f}% from Global")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Global Model (Before pFL)")
        st.markdown(f"**Accuracy:** {p_data['global_acc']*100:.2f}% | **F1:** {p_data['global_f1']:.4f}")
        df_global = pd.DataFrame(p_data["report_global"]).transpose()
        st.dataframe(df_global.style.background_gradient(cmap='Blues'))
        
    with col2:
        st.subheader("Personalized Model (After pFL)")
        st.markdown(f"**Accuracy:** {p_data['personal_acc']*100:.2f}% | **F1:** {p_data['personal_f1']:.4f}")
        df_personal = pd.DataFrame(p_data["report_personal"]).transpose()
        st.dataframe(df_personal.style.background_gradient(cmap='Greens'))