import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import base64
import os

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="Классификатор периферии + Экономика",
    layout="wide"  # широкий режим для PDF
)

# ===== ЗАГРУЗКА МОДЕЛИ И КЛАССОВ (кэшируется) =====
@st.cache_resource
def load_models():
    model = load_model('models/best_accessories.h5')
    with open('models/class_names_accessories.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_models()

# ============================================================
# БОКОВАЯ ПАНЕЛЬ (видна только в режиме классификации)
# ============================================================
mode = st.sidebar.radio(
    "Выберите раздел",
    ["Классификация периферии", "Экономика"]
)

# ============================================================
# БЛОК 1: КЛАССИФИКАЦИЯ ПЕРИФЕРИИ (без изменений)
# ============================================================
if mode == "Классификация периферии":
    # ... (весь твой старый код классификации, я его опускаю для краткости,
    # но он должен быть точно таким же, как у тебя был)
    pass

# ============================================================
# БЛОК 2: ЭКОНОМИКА
# ============================================================
elif mode == "Экономика":
    # Если в сессии нет режима просмотра, устанавливаем "выбор"
    if "view" not in st.session_state:
        st.session_state.view = "choose"

    # ---- РЕЖИМ ПРОСМОТРА PDF (полный экран) ----
    if st.session_state.view == "pdf":
        # Убираем всё лишнее, показываем только PDF
        st.markdown("""
        <style>
            .stApp { margin: 0; padding: 0; }
            iframe { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; border: none; }
            .back-btn { position: fixed; top: 10px; left: 10px; z-index: 1000; background: rgba(0,0,0,0.7); color: white; padding: 10px 20px; border-radius: 8px; cursor: pointer; }
        </style>
        """, unsafe_allow_html=True)

        # Кнопка "Назад" поверх PDF
        if st.button("← Назад к выбору", key="back_from_pdf"):
            st.session_state.view = "choose"
            st.rerun()

        # Читаем PDF и встраиваем
        pdf_path = "Economics_PDF.pdf"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="100%" style="border:none;"></iframe>'
            st.components.v1.html(pdf_display, height=800)  # высота большая, но iframe фиксирован через CSS
        else:
            st.error("Файл PDF не найден.")

    # ---- РЕЖИМ ВЫБОРА ДОКУМЕНТА ----
    else:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="font-size: 38px; margin: 0;">📚 Экономические материалы</h1>
            <p style="font-size: 18px; color: white;">Выберите документ для просмотра или скачивания</p>
        </div>
        """, unsafe_allow_html=True)

        docx_path = "Economics_DOCX.docx"
        pdf_path = "Economics_PDF.pdf"

        if not os.path.exists(docx_path) or not os.path.exists(pdf_path):
            st.error("Один или оба файла не найдены в текущей директории.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📄 Контрольная работа (DOCX)")
                with open(docx_path, "rb") as f:
                    docx_bytes = f.read()
                st.download_button(
                    label="⬇️ Скачать контрольную работу",
                    data=docx_bytes,
                    file_name="Контрольная_работа_по_экономике.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
                st.info("Файл DOCX – скачайте и откройте в Word или Google Docs.")

            with col2:
                st.markdown("### 📊 Презентация (PDF)")
                if st.button("📖 Открыть PDF на весь экран", use_container_width=True):
                    st.session_state.view = "pdf"
                    st.rerun()
                # Также даём кнопку скачивания
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="⬇️ Скачать презентацию (PDF)",
                    data=pdf_bytes,
                    file_name="Презентация_по_экономике.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
