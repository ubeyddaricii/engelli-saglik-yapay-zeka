# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:45:25 2026

@author: ubeyd
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- MODEL ARKA PLAN EĞİTİMLERİ ---
# Modül 1: Bası Yarası
df_basi = pd.DataFrame({
    'Yatirma_Suresi': [4, 12, 6, 15, 5, 14, 8, 16],
    'Cilt_Hassasiyeti': [1, 3, 1, 3, 2, 3, 1, 3],
    'Mobilite': [1, 0, 1, 0, 1, 0, 1, 0],
    'Bası_Yarasi_Riski': [0, 1, 0, 1, 0, 1, 0, 1]
})
model_basi = RandomForestClassifier(random_state=42)
model_basi.fit(df_basi[['Yatirma_Suresi', 'Cilt_Hassasiyeti', 'Mobilite']], df_basi['Bası_Yarasi_Riski'])

# Modül 2: Solunum & Spazm
df_spazm = pd.DataFrame({
    'Spazm_Sikligi': [1, 4, 2, 5, 1, 4, 3, 5],
    'Yutma_Guclugu': [0, 1, 0, 1, 0, 1, 1, 1],
    'Oksijen_Spo2': [98, 91, 97, 89, 98, 90, 95, 88],
    'Solunum_Riski': [0, 1, 0, 1, 0, 1, 0, 1]
})
model_spazm = RandomForestClassifier(random_state=42)
model_spazm.fit(df_spazm[['Spazm_Sikligi', 'Yutma_Guclugu', 'Oksijen_Spo2']], df_spazm['Solunum_Riski'])

# Modül 3: Dolaşım & Diyabet
df_dolasim = pd.DataFrame({
    'Kan_Sekeri': [95, 190, 100, 220, 110, 185, 90, 210],
    'Tansiyon_Sistolik': [120, 155, 115, 165, 125, 150, 110, 160],
    'Odem_Durumu': [0, 1, 0, 1, 0, 1, 0, 1],
    'Dolasim_Riski': [0, 1, 0, 1, 0, 1, 0, 1]
})
model_dolasim = RandomForestClassifier(random_state=42)
model_dolasim.fit(df_dolasim[['Kan_Sekeri', 'Tansiyon_Sistolik', 'Odem_Durumu']], df_dolasim['Dolasim_Riski'])

# --- STREAMLIT ARAYÜZ TASARIMI ---
st.title("Engelli Sağlık ve Komplikasyon Risk Takip Sistemi")
st.write("Bu panel, farklı engel gruplarındaki bireyler için yapay zeka destekli erken uyarı sağlar.")

modul_secim = st.selectbox(
    "Lütfen İlgili Engel / Modül Grubunu Seçiniz:",
    ("1. Bedensel Engel (Bası Yarası Riski)", 
     "2. Serebral Palsi / Nöromüsküler (Solunum & Spazm Riski)", 
     "3. Hareketsiz Birey (Dolaşım & Diyabet Riski)")
)

st.markdown("---")

if modul_secim == "1. Bedensel Engel (Bası Yarası Riski)":
    st.subheader("Modül 1: Bası Yarası (Ülser) Analizi")
    yatirma = st.slider("Günlük Yatma / Oturma Süresi (Saat)", 1, 24, 10)
    cilt = st.selectbox("Cilt Hassasiyeti Seviyesi", [1, 2, 3], format_func=lambda x: "Normal" if x==1 else ("Orta" if x==2 else "Yüksek Hassasiyet"))
    mobilite = st.selectbox("Mobilite Durumu", [0, 1], format_func=lambda x: "Tam Bağımlı" if x == 0 else "Yarı Bağımlı")
    
    if st.button("Riski Hesapla"):
        tahmin = model_basi.predict([[yatirma, cilt, mobilite]])
        if tahmin[0] == 1:
            st.error("SONUÇ: Yüksek Bası Yarası Riski! Acil pozisyon değiştirilmeli.")
        else:
            st.success("SONUÇ: Normal Durum (Risk Düşük)")

elif modul_secim == "2. Serebral Palsi / Nöromüsküler (Solunum & Spazm Riski)":
    st.subheader("Modül 2: Solunum & Spazm Analizi")
    spazm = st.slider("Günlük Kas Spazmı Sıklığı (1-5 Puan)", 1, 5, 2)
    yutma = st.selectbox("Yutma Güçlüğü", [0, 1], format_func=lambda x: "Yok" if x == 0 else "Var")
    spo2 = st.slider("Anlık Oksijen Satürasyonu (SpO2 %)", 70, 100, 95)
    
    if st.button("Riski Hesapla"):
        tahmin = model_spazm.predict([[spazm, yutma, spo2]])
        if tahmin[0] == 1:
            st.error("SONUÇ: Yüksek Solunum / Aspirasyon Riski!")
        else:
            st.success("SONUÇ: Stabil Durum (Risk Düşük)")

else:
    st.subheader("Modül 3: Dolaşım & Diyabet Komplikasyon Analizi")
    seker = st.number_input("Açlık Kan Şekeri (mg/dL)", 50, 300, 110)
    tansiyon = st.number_input("Büyük Tansiyon (Sistolik mmHg)", 80, 220, 120)
    odem = st.selectbox("Alt Ekstremitede Ödem / Şişlik", [0, 1], format_func=lambda x: "Yok" if x == 0 else "Var")
    
    if st.button("Riski Hesapla"):
        tahmin = model_dolasim.predict([[seker, tansiyon, odem]])
        if tahmin[0] == 1:
            st.error("SONUÇ: Yüksek Dolaşım Komplikasyonu Riski!")
        else:
            st.success("SONUÇ: Normal Durum (Risk Düşük)")