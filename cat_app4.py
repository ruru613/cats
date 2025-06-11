import streamlit as st
import pathlib
from fastai.learner import load_learner
from typing import Optional

import streamlit as st
import pathlib
from fastai.learner import load_learner

@st.cache_resource(show_spinner="正在加载AI模型...")
def load_model():
    """安全加载fastai模型，处理跨平台路径问题"""
    # 保存原始路径类型，避免跨平台冲突
    posix_backup = pathlib.PosixPath
    try:
        # 临时修改路径类型（仅在Windows环境下需要）
        if sys.platform == "win32":
            pathlib.PosixPath = pathlib.WindowsPath
            
        # 动态构建模型路径（确保相对路径正确）
        model_path = pathlib.Path(__file__).parent / "cats.pkl"
        
        # 检查文件是否存在
        if not model_path.exists():
            st.error(f"模型文件不存在: {model_path}")
            return None
            
        # 加载模型
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None
    finally:
        # 恢复原始路径类型
        pathlib.PosixPath = posix_backup

# 在主程序中使用
def main():
    st.title("猫咪品种识别")
    
    # 加载模型
    model = load_model()
    if not model:
        st.stop()
        
    # 后续应用逻辑...

# 调用示例（在主流程中）
model = safe_load_model()
if model:
    st.success("模型加载成功！")
    # 后续推理逻辑...



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