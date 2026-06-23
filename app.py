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
    layout="centered"
)

# ===== ЗАГРУЗКА МОДЕЛИ И КЛАССОВ (кэшируется) =====
@st.cache_resource
def load_models():
    model = load_model('models/best_accessories.h5')
    with open('models/class_names_accessories.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_models()

# ===== БОКОВАЯ ПАНЕЛЬ С ВЫБОРОМ РАЗДЕЛА =====
mode = st.sidebar.radio(
    "Выберите раздел",
    ["Классификация периферии", "Экономика"]
)

# ============================================================
# БЛОК 1: КЛАССИФИКАЦИЯ ПЕРИФЕРИИ (старый код, без изменений)
# ============================================================
if mode == "Классификация периферии":
    # Боковая панель со списком классов
    with st.sidebar:
        st.markdown("### Доступные классы")
        for cls in class_names:
            st.markdown(cls)

    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 42px; margin: 0;">Классификатор периферии</h1>
        <p style="font-size: 18px; color: white;">Загрузите изображение — нейросеть определит устройство</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Выберите изображение...",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Загруженное изображение", use_container_width=True)

        with st.spinner("Анализируем изображение..."):
            time.sleep(0.3)
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

        predicted_class = class_names[predicted_index]

        st.markdown(f"""
        <div style="background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #636e72; font-size: 14px;">Распознано как</div>
                    <div style="font-size: 28px; font-weight: bold; color: #2d3436;">{predicted_class}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #636e72; font-size: 14px;">Уверенность</div>
                    <div style="font-size: 22px; font-weight: 600; color: {'#00b894' if confidence > 0.7 else '#ff6b6b'};">{confidence:.1%}</div>
                </div>
            </div>
            <div style="background: #dfe6e9; border-radius: 10px; height: 10px; margin-top: 10px; overflow: hidden;">
                <div style="height: 100%; width: {confidence * 100}%; background: linear-gradient(90deg, #00b894, #00cec9); border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Вероятности по классам")
        sorted_indices = np.argsort(predictions[0])[::-1]
        for idx in sorted_indices:
            prob = predictions[0][idx] * 100
            class_name = class_names[idx]
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; border-bottom: 1px solid #f1f2f6;">
                <span style="font-weight: 500; color: #2d3436;">{class_name}</span>
                <span style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 150px; background: #f1f2f6; border-radius: 4px; height: 8px; overflow: hidden;">
                        <span style="display: block; height: 100%; width: {prob}%; background: #0984e3; border-radius: 4px;"></span>
                    </span>
                    <span style="font-weight: 600; color: #0984e3;">{prob:.1f}%</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 0; color: white !important;">
            <p style="font-size: 20px; color: white !important;">Загрузите изображение для классификации</p>
            <p style="font-size: 16px; color: white !important;">
                Поддерживаются: мышь, клавиатура, монитор, наушники, смартфон, смарт-часы, колонки, зарядка, игровой контроллер
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# БЛОК 2: ЭКОНОМИКА (новый раздел)
# ============================================================
elif mode == "Экономика":
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 38px; margin: 0;">📚 Экономические материалы</h1>
        <p style="font-size: 18px; color: white;">Выберите документ для просмотра или скачивания</p>
    </div>
    """, unsafe_allow_html=True)

    # Проверяем, существуют ли файлы
    docx_path = "Economics_DOCX.docx"
    pdf_path = "Economics_PDF.pdf"

    if not os.path.exists(docx_path) or not os.path.exists(pdf_path):
        st.error("Один или оба файла не найдены в текущей директории. Убедитесь, что они загружены в репозиторий.")
    else:
        # Две колонки для выбора документа
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Контрольная работа (DOCX)", use_container_width=True):
                st.session_state.selected_doc = "docx"
            # Если выбран DOCX, показываем его
            if st.session_state.get("selected_doc") == "docx":
                st.markdown("#### Контрольная работа (40 страниц)")
                # Читаем файл
                with open(docx_path, "rb") as f:
                    docx_bytes = f.read()
                # Кнопка скачивания
                st.download_button(
                    label="⬇️ Скачать контрольную работу (DOCX)",
                    data=docx_bytes,
                    file_name="Контрольная_работа_по_экономике.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
                # Пояснение: встроенный просмотр DOCX невозможен в браузере
                st.info("Файл в формате DOCX. Нажмите кнопку выше, чтобы скачать и открыть в Word или Google Docs.")

        with col2:
            if st.button("📊 Презентация (PDF)", use_container_width=True):
                st.session_state.selected_doc = "pdf"
            if st.session_state.get("selected_doc") == "pdf":
                st.markdown("#### Презентация (10 слайдов)")
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                # Кнопка скачивания
                st.download_button(
                    label="⬇️ Скачать презентацию (PDF)",
                    data=pdf_bytes,
                    file_name="Презентация_по_экономике.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                # Встроенный просмотр PDF через iframe
                b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                st.components.v1.html(pdf_display, height=620)

        # Если ничего не выбрано, показываем подсказку
        if "selected_doc" not in st.session_state:
            st.markdown("""
            <div style="text-align: center; padding: 30px 0; color: #b2bec3;">
                <p style="font-size: 18px;">Нажмите на одну из кнопок выше, чтобы открыть материал.</p>
            </div>
            """, unsafe_allow_html=True)
