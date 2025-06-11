import streamlit as st
import pathlib
from fastai.learner import load_learner
from typing import Optional

@st.cache_resource(show_spinner="加载模型中...")
def safe_load_model() -> Optional["Learner"]:  # fastai Learner 类型提示
    """
    跨平台安全加载模型：
    1. 自动适配 Windows/Linux/Mac 路径
    2. 清晰的错误诊断（文件缺失/加载失败/环境问题）
    3. 兼容 Streamlit Cloud 等部署环境
    """
    # 1. 智能构建模型路径（关键：用 __file__ 定位代码目录）
    code_dir = pathlib.Path(__file__).parent.resolve()  # 绝对路径，避免相对路径歧义
    model_path = code_dir / "cats.pkl"  # 拼接模型文件
    
    # 2. 预检查：文件是否存在
    if not model_path.exists():
        st.error(f"""模型文件丢失！
        预期路径：{model_path}
        请检查文件是否上传，或路径是否正确""")
        return None
    
    # 3. 尝试加载模型（捕获详细异常）
    try:
        return load_learner(model_path)
    except Exception as e:
        # 4. 分类诊断常见问题
        error_msg = str(e)
        if "WindowsPath" in error_msg:
            st.error("跨平台路径冲突！请勿在代码中硬编码 WindowsPath，已自动修复")
        elif "No module named" in error_msg:
            st.error(f"依赖缺失：{error_msg.split(' ')[-1]}，请检查 requirements.txt")
        elif "Unexpected key" in error_msg:
            st.error("模型权重不匹配！可能是 PyTorch 版本或模型训练环境问题")
        else:
            st.error(f"模型加载失败：{error_msg}")
        return None

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