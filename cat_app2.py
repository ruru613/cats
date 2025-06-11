import streamlit as st
import sys

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

from fastai.vision.all import *
import pathlib
from fastai.learner import load_learner

@st.cache_resource
def load_model():
    """加载并缓存模型，兼容跨平台路径"""
    # 自动适配路径，无需强制替换路径类
    model_path = pathlib.Path(__file__).parent / "cats.pkl"  
    try:
        return load_learner(model_path)
    except FileNotFoundError:
        st.error(f"模型文件未找到：{model_path}")
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
    return None



st.title("Desert cat, Mainecoon, Siberian Cats分类器")
st.write("上传一张图片，看看它是 Desert cat, Mainecoon还是 Siberian Cats!")

model = load_model()
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    pred, pred_idx, probs = model.predict(image)
    st.write(f"预测结果：{pred}")
    st.write(f"概率：{probs[pred_idx]:.04f}")