# filepath: d:\PG_article\test_torch.py
import streamlit as st
import torch

st.title("PyTorch Test")
st.write("PyTorch version:", torch.__version__)

tensor = torch.rand(5, 3)
st.write("Random tensor:", tensor)