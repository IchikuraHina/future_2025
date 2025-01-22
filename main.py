import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ページ設定
st.set_page_config(page_title="Cloud Type Classifier", page_icon="☁️")

# タイトルと説明
st.title("天気診断 🌥️ vs 🌧️")
st.write("Upload a cloud image to check if it's a rain cloud or not!")

# モデルの読み込み
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cloud_classifier.h5')

model = load_model()

def predict_cloud(img):
    # 画像の前処理
    img = img.resize((183, 183))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 予測
    prediction = model.predict(img_array)
    confidence_value = float(prediction[0][0])
    
    # 結果の解釈
    if confidence_value > 0.5:
        return "雨雲 🌧️", confidence_value
    else:
        return "晴雲 ☁️", confidence_value

# ファイルアップローダーの作成
uploaded_file = st.file_uploader("Choose a cloud image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cloud Image', use_column_width=True)
    
    # 予測ボタン
    if st.button('Classify Cloud'):
        with st.spinner('Analyzing the cloud...'):
            # 予測の実行
            result, confidence = predict_cloud(image)
            
            # 結果の表示
            st.write("### 結果:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**予測:**")
                st.write(f"### {result}")
            
            with col2:
                st.write("**Confidence:**")
                st.write(f"### {confidence:.2%}")
            
            # 追加の説明
            st.write("---")
            if result.startswith("Rain"):
                st.write("この雲は雨雲のようです。傘を持って行った方がいいかもしれません！ ☔")
            else:
                st.write("晴天の雲のようです。天気は良さそうです！ 🌤️")

# フッター
st.markdown("---")