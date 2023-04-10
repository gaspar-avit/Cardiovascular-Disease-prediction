# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:45:36 2023.

@author: Gaspar Avit Ferrero
"""

import os
import streamlit as st
import pandas as pd

from streamlit import session_state as session
from htbuilder import HtmlElement, div, hr, a, p, styles
from htbuilder.units import percent, px
from catboost import CatBoostClassifier
from datetime import datetime


###############################
## ------- FUNCTIONS ------- ##
###############################

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        height=px(10),
        color="black",
        text_align="center",
        # height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(0)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made by ",
        link("https://www.linkedin.com/in/gaspar-avit/", "Gaspar Avit"),
    ]  # with ❤️
    layout(*myargs)


def update_prediction(input_data):
    """Callback to automatically update prediction if button has already been
    clicked"""
    if IS_CLICKED:
        launch_prediction(input_data)


def get_input_data():
    """
    Generate input layout and get input values.

    -return: DataFrame with input data.
    """
    session.input_data = pd.DataFrame()

    input_expander = st.expander('Input parameters', True)
    with input_expander:
        # Row 1
        col_age, col_sex = st.columns(2)
        with col_age:
            session.input_data.loc[0, 'age'] = st.slider('Age', 18, 75)
            # on_change=update_prediction(session.input_data)
        with col_sex:
            session.input_data.loc[0, 'gender'] = st.radio(
                'Sex', ['Female', 'Male'])
            session.input_data["gender"] = session.input_data["gender"].astype(
                'category')

        # Row 2
        col_height, col_weight = st.columns(2)
        with col_height:
            session.input_data.loc[0, 'height'] = st.slider('Height', 140, 200)
            session.input_data["height"] = session.input_data["height"].astype(
                int)
        with col_weight:
            session.input_data.loc[0, 'weight'] = st.slider('Weight', 40, 140)
            session.input_data["weight"] = session.input_data["weight"].astype(
                int)

        # Row 3
        col_ap_hi, col_ap_lo = st.columns(2)
        with col_ap_hi:
            session.input_data.loc[0, 'ap_hi'] = st.slider(
                'Systolic blood pressure', 90, 200)
            session.input_data["ap_hi"] = session.input_data["ap_hi"].astype(
                int)
        with col_ap_lo:
            session.input_data.loc[0, 'ap_lo'] = st.slider(
                'Diastolic blood pressure', 50, 120)
            session.input_data["ap_lo"] = session.input_data["ap_lo"].astype(
                int)

        # Row 4
        col_chole, col_gluc = st.columns(2)
        with col_chole:
            cholest = st.radio(
                'Cholesterol', ['Normal', 'Above normal', 'Well above normal'])
            session.input_data.loc[0, 'cholesterol'] = [
                1 if 'Normal' in cholest else 2 if 'Above normal' in cholest
                else 3][0]
            session.input_data["cholesterol"] = (session
                                                 .input_data["cholesterol"]
                                                 .astype(int)
                                                 .astype('category')
                                                 )
        with col_gluc:
            gluc = st.radio(
                'Glucose', ['Normal', 'Above normal', 'Well above normal'])
            session.input_data.loc[0, 'gluc'] = [
                1 if 'Normal' in gluc else 2 if 'Above normal' in gluc
                else 3][0]
            session.input_data["gluc"] = (session
                                          .input_data["gluc"]
                                          .astype(int)
                                          .astype('category')
                                          )

        # Row 5
        col_alco, col_smk = st.columns(2)
        with col_alco:
            alco = st.radio('Alcohol intake', ['Yes', 'No'], 1)
            session.input_data.loc[0, 'alco'] = [1 if 'Yes' in alco else 0][0]
            session.input_data["alco"] = session.input_data["alco"].astype(
                bool)
        with col_smk:
            smoke = st.radio('Smoking', ['Yes', 'No'], 1)
            session.input_data.loc[0, 'smoke'] = [1 if 'Yes' in smoke
                                                  else 0][0]
            session.input_data["smoke"] = session.input_data["smoke"].astype(
                bool)

        # Row 6
        active = st.radio('Physical activity', ['Yes', 'No'])
        session.input_data.loc[0, 'active'] = [1 if 'Yes' in active else 0][0]
        session.input_data["active"] = session.input_data["active"].astype(
            bool)

        st.write("")

    # Compute extra features
    session.input_data["bmi"] = session.input_data["weight"] \
        / (session.input_data["height"]/100)**2
    session.input_data["bad_habits"] = session.input_data["smoke"] \
        & session.input_data["alco"]

    return session.input_data


def generate_prediction(input_data):
    """
    Generate prediction of cardiovascular disease probability based on input
    data.

    -param input_data: DataFrame with input data

    -return: predicted probability of having a cardiovascular disease
    """
    # Compute probability
    probs = MODEL.predict_proba(input_data)    
    positive_proba = round(probs[0,1]*100)

    # Show results
    st.markdown("<h2 style='text-align: center; color: black;'>\
                Predicted cardiovascular disease probability:</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: grey;'>\
                {positive_proba} %</h1>",
                unsafe_allow_html=True)
    st.progress(positive_proba)
    st.text("")
    st.text("")
    
    return probs


###############################
## --------- MAIN ---------- ##
###############################


if __name__ == "__main__":

    ## --- Page config ------------ ##
    # Set page title
    st.title("""
    Cardiovascular Disease predictor
    #### This app aims to give a scoring of how probable is that an individual \
    would suffer from a cardiovascular disease given its physical \
         characteristics
    #### Just enter your info and get a prediction.
    """)
    st.text("")

    # Set page footer
    # footer()

    # Initialize clicking flag
    IS_CLICKED = False

    ## --------------------------- ##

    # Load classification model
    MODEL = CatBoostClassifier()
    MODEL.load_model('./model.cbm')

    # Get inputs
    session.input_data = get_input_data()

    # Create button to trigger poster generation
    st.text("")
    st.text("")
    buffer1, col1, buffer2 = st.columns([1.3, 1, 1])
    IS_CLICKED = col1.button(label="Generate predictions")

    st.text("")
    st.text("")

    # Generate prediction
    if IS_CLICKED:
        predicted_probs = generate_prediction(session.input_data)

    st.text("")
    st.text("")
