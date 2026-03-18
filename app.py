import streamlit as st
from retriever import search
from back_end import build_index
from PIL import Image
import time

st.title("Task 1")

query_type = st.radio("select query type", ["Text", "Image"])

query_text = None
query_image_path = None

if query_type == "Text":
    query_text = st.text_input("enter your query")

else:
    uploaded_file = st.file_uploader("upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        query_image_path = f"temp_{uploaded_file.name}"
        with open(query_image_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(query_image_path, caption="query Image")

if st.button("Search"):
    results = search(query_text=query_text, image_path=query_image_path)

    st.subheader("RESULTS")

    for res in results:
        st.write(f"source: {res['source']}")

        if res["type"] == "text":
            st.write(res["content"][:500])

        else:
            img = Image.open(res["content"])
            st.image(img)