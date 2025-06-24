import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- Load or create dataset ---
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("data.csv")
        except:
            df = pd.DataFrame({
                "home_xg": [2.1, 1.8, 1.5, 2.2, 1.7],
                "away_xg": [1.3, 1.5, 1.1, 1.8, 1.2],
                "home_form": [10, 8, 7, 12, 9],
                "away_form": [7, 9, 6, 10, 8],
                "result": ["H", "D", "H", "A", "H"]
            })
            df.to_csv("data.csv", index=False)
    return df

# --- Train Model ---
def train_model(df):
    X = df[["home_xg", "away_xg", "home_form", "away_form"]]
    y = LabelEncoder().fit_transform(df["result"])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

# --- Load model ---
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        df = load_data()
        return train_model(df)

def prediksi_skor_poisson(xg_home, xg_away, max_goals=5):
    hasil = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
            hasil.append({"skor": f"{i}-{j}", "prob": prob})
    hasil = sorted(hasil, key=lambda x: x['prob'], reverse=True)
    return hasil[:5]

# --- Streamlit App ---
st.set_page_config(page_title="Prediksi Sepak Bola ML", layout="centered")
st.title("⚽ Prediksi Hasil Pertandingan Sepak Bola")

st.markdown("Masukkan statistik pertandingan untuk memprediksi hasil pertandingan (Home Win / Draw / Away Win)")

uploaded_file = st.file_uploader("Upload file dataset CSV kamu (opsional)", type="csv")
if uploaded_file:
    st.success("Dataset berhasil di-upload. Model akan dilatih ulang.")
    df = load_data(uploaded_file)
    model = train_model(df)
else:
    model = load_model()



# Tombol untuk auto-refresh data Understat (hanya lokal)
st.markdown("### 🔄 Update Data Understat (otomatis lokal)")
if st.button("🔁 Refresh Data Now"):
    import subprocess
    try:
        result = subprocess.run(["python", "understat_scraper.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Data berhasil di-refresh!")
        else:
            st.error("❌ Gagal menjalankan scraper. Cek pesan berikut:")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# --- Ambil data Understat jika ada ---
st.markdown("### ⚽ Pilih Tim Secara Otomatis (dari Understat)")
team_stats = None
if os.path.exists("tim_stats_understat.csv"):
    team_stats = pd.read_csv("tim_stats_understat.csv")
    all_teams = sorted(team_stats["team"].unique())
    selected_home_team = st.selectbox("Pilih Tim Kandang", ["(manual)"] + all_teams, index=0)
    selected_away_team = st.selectbox("Pilih Tim Tandang", ["(manual)"] + all_teams, index=0)

    if selected_home_team != "(manual)":
        row = team_stats[team_stats["team"] == selected_home_team].iloc[0]
        home_xg = row["xg"]
        home_form = int(row["form_points"])
    if selected_away_team != "(manual)":
        row = team_stats[team_stats["team"] == selected_away_team].iloc[0]
        away_xg = row["xg"]
        away_form = int(row["form_points"])
else:
    st.info("File tim_stats_understat.csv belum ditemukan. Silakan jalankan scraper terlebih dahulu.")


home_xg = st.slider("xG Tim Kandang", 0.0, 4.0, 1.8, step=0.1)
away_xg = st.slider("xG Tim Tandang", 0.0, 4.0, 1.3, step=0.1)
home_form = st.slider("Form Tim Kandang (poin 5 laga terakhir)", 0, 15, 10)
away_form = st.slider("Form Tim Tandang (poin 5 laga terakhir)", 0, 15, 7)

st.markdown("### 🎯 Masukkan Odds Taruhan (Optional untuk Value Betting)")
odd_home = st.number_input("Odds Menang Tim Kandang", min_value=1.0, max_value=20.0, value=1.80, step=0.01)
odd_draw = st.number_input("Odds Seri", min_value=1.0, max_value=20.0, value=3.50, step=0.01)
odd_away = st.number_input("Odds Menang Tim Tandang", min_value=1.0, max_value=20.0, value=4.20, step=0.01)

if st.button("Prediksi!"):
    X_input = np.array([[home_xg, away_xg, home_form, away_form]])
    pred_proba = model.predict_proba(X_input)[0]
    pred_class = model.predict(X_input)[0]

    label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    st.subheader("📊 Hasil Prediksi")
    st.write(f"**Prediksi Hasil:** {label_map[pred_class]}")
    st.write("**Probabilitas:**")
    st.progress(pred_proba[2], text=f"Home Win: {pred_proba[2]:.2%}")
    st.progress(pred_proba[1], text=f"Draw: {pred_proba[1]:.2%}")
    st.progress(pred_proba[0], text=f"Away Win: {pred_proba[0]:.2%}")

    # Visualisasi Pie Chart Probabilitas
    st.markdown("### 📊 Diagram Probabilitas 1X2")
    fig, ax = plt.subplots()
    labels = ['Away Win', 'Draw', 'Home Win']
    colors = ['#FF6961', '#FFD700', '#77DD77']
    ax.pie(pred_proba, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)


    st.markdown("### 💰 Value Bet Analysis")
    value_home = (pred_proba[2] * odd_home) - 1
    value_draw = (pred_proba[1] * odd_draw) - 1
    value_away = (pred_proba[0] * odd_away) - 1

    def highlight_value(val):
        return f":green[{val:.2%}]" if val > 0 else f":red[{val:.2%}]"

    st.write(f"Home Win Value: {highlight_value(value_home)}")
    st.write(f"Draw Value: {highlight_value(value_draw)}")
    st.write(f"Away Win Value: {highlight_value(value_away)}")

    st.markdown("**🔢 Prediksi Skor Paling Mungkin (Model Poisson):**")
    skor_prediksi = prediksi_skor_poisson(home_xg, away_xg)
    for row in skor_prediksi:
        st.write(f"Skor {row['skor']} : {row['prob']:.2%}")

    new_log = pd.DataFrame([{
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_form": home_form,
        "away_form": away_form,
        "predicted": label_map[pred_class],
        "home_win_proba": pred_proba[2],
        "draw_proba": pred_proba[1],
        "away_win_proba": pred_proba[0]
    }])

    if os.path.exists("prediksi_log.csv"):
        log_df = pd.read_csv("prediksi_log.csv")
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        log_df = new_log
    log_df.to_csv("prediksi_log.csv", index=False)

    st.markdown("_Model: Random Forest (dilatih dari data historis)_")
    st.markdown("_Riwayat prediksi disimpan di 'prediksi_log.csv'_")

# Riwayat prediksi
st.markdown("### 🧾 Riwayat Prediksi Sebelumnya")
if os.path.exists("prediksi_log.csv"):
    log_df = pd.read_csv("prediksi_log.csv")
    st.dataframe(log_df)
    st.download_button("⬇️ Download Riwayat Prediksi", data=log_df.to_csv(index=False), file_name="riwayat_prediksi.csv", mime="text/csv")
else:
    st.info("Belum ada riwayat prediksi tersimpan.")

st.markdown("---")
st.header("📤 Prediksi Massal (Batch)")
batch_file = st.file_uploader("Upload file CSV untuk prediksi banyak pertandingan", type="csv", key="batch")
if batch_file:
    batch_df = pd.read_csv(batch_file)
    batch_features = batch_df[["home_xg", "away_xg", "home_form", "away_form"]]
    batch_preds = model.predict(batch_features)
    batch_probas = model.predict_proba(batch_features)
    label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    batch_df["Prediksi"] = [label_map[p] for p in batch_preds]
    batch_df["Prob_HomeWin"] = batch_probas[:, 2]
    batch_df["Prob_Draw"] = batch_probas[:, 1]
    batch_df["Prob_AwayWin"] = batch_probas[:, 0]
    
    if "home_team" in batch_df.columns and "away_team" in batch_df.columns:
        result_cols = ["home_team", "away_team", "home_xg", "away_xg", "home_form", "away_form", "Prediksi",
                       "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"]
    else:
        result_cols = ["home_xg", "away_xg", "home_form", "away_form", "Prediksi",
                       "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"]

    st.dataframe(batch_df[result_cols])
    batch_df.to_csv("batch_prediksi_output.csv", index=False)
    st.success("✅ Hasil disimpan ke batch_prediksi_output.csv")

    st.markdown("### 💰 Value Bet Ranking")
    # Input odds untuk batch (sama semua)
    odd_home_b = st.number_input("Odds Menang Kandang (Batch)", min_value=1.0, value=1.80, step=0.01, key="odd_home_b")
    odd_draw_b = st.number_input("Odds Seri (Batch)", min_value=1.0, value=3.50, step=0.01, key="odd_draw_b")
    odd_away_b = st.number_input("Odds Menang Tandang (Batch)", min_value=1.0, value=4.20, step=0.01, key="odd_away_b")

    batch_df["Value_Home"] = (batch_df["Prob_HomeWin"] * odd_home_b) - 1
    batch_df["Value_Draw"] = (batch_df["Prob_Draw"] * odd_draw_b) - 1
    batch_df["Value_Away"] = (batch_df["Prob_AwayWin"] * odd_away_b) - 1

    def highlight_value(val):
        color = 'background-color: lightgreen' if val > 0 else ''
        return color

    st.markdown("#### 📋 Daftar Pertandingan dengan Value Bet")
    st.dataframe(batch_df.style.applymap(highlight_value, subset=["Value_Home", "Value_Draw", "Value_Away"]))

    # Ranking
    batch_df["Max_Value"] = batch_df[["Value_Home", "Value_Draw", "Value_Away"]].max(axis=1)
    top_value_df = batch_df.sort_values(by="Max_Value", ascending=False).head(10)
    st.markdown("#### 🏆 Top 10 Value Bet Tertinggi")
    st.dataframe(top_value_df[["home_xg", "away_xg", "home_form", "away_form", "Prediksi", "Max_Value"]])

