import streamlit as st
import pandas as pd
import numpy as np


def run(st,data):
    st.text("Data Preparation - ")
    encoding_options = st.radio("Encoding",["Automatic","Manual"])
    if encoding_options:
        pass