import streamlit as st
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load('model.pkl')
    model_loaded = True
except:
    model_loaded = False

# Streamlit Page Setup
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.markdown(
    """
    Aplikasi ini memprediksi jenis bunga **Iris** berdasarkan 4 fitur utama menggunakan **Machine Learning**.
    """
)

# Tambah gambar iris
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Flower Example", use_column_width=True)

# Sidebar for Inputs
with st.form("input_form"):
    st.subheader("Masukkan Nilai Karakteristik Bunga:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

    submit = st.form_submit_button("🔎 Prediksi Jenis Iris")

if submit:
    if model_loaded:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        predicted_class = {0: 'Setosa 🌼', 1: 'Versicolor 🌷', 2: 'Virginica 🌺'}[prediction[0]]
        st.success(f"Hasil Prediksi: **{predicted_class}**")

        with st.expander("Deskripsi Spesies"):
            if prediction[0] == 0:
                st.info("**Setosa**: Memiliki kelopak kecil, dominan warna putih keunguan.")
            elif prediction[0] == 1:
                st.info("**Versicolor**: Ukuran sedang, kelopak warna ungu ke biru.")
            else:
                st.info("**Virginica**: Paling besar, kelopak dominan warna ungu tua.")
    else:
        st.error("❌ Model tidak berhasil dimuat. Pastikan file model.pkl ada di folder yang sama!")

# Footer
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit | @2025")
