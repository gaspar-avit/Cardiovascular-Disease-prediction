# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:45:36 2023

@author: Gaspar Avit Ferrero
"""

import streamlit as st

from htbuilder import HtmlElement, div, hr, a, p, styles
from htbuilder.units import percent, px
from catboost import CatBoostClassifier


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
        color="black",
        text_align="center",
        height="auto",
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
        hr(
            style=style_hr
        ),
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
        "Made with ❤️ by ",
        link("https://www.linkedin.com/in/gaspar-avit/", "Gaspar Avit"),
    ]
    layout(*myargs)


def update_prediction():
    """Callback to automatically update prediction if button has already been
    clicked"""
    if is_clicked:
        launch_prediction()


def input_layout():

    input_expander = st.expander('Input parameters', True)
    with input_expander:
        # Row 1
        col_age, col_sex = st.columns(2)
        col_age = st.slider('Age', 18, 75, on_change=update_prediction())
        col_sex = st.radio('Sex', ['Female', 'Male'],
                           on_change=update_prediction())
        st.write(‘div.row-widget.stRadio > div{flex-direction: row
                                               justify-content: center}’,
                 unsafe_allow_html=True)

        # Row 2
        col_height, col_weight = st.columns(2)
        col_height = st.slider(
            'Height', 140, 200, on_change=update_prediction())
        col_weight = st.slider(
            'Weight', 40, 140, on_change=update_prediction())

        # Row 3
        col_ap_hi, col_ap_lo = st.columns(2)
        col_ap_hi = st.slider(
            'AP Hi', 90, 200, on_change=update_prediction())
        col_ap_lo = st.slider(
            'AP Lo', 50, 120, on_change=update_prediction())
        

###############################
## --------- MAIN ---------- ##
###############################


if __name__ == "__main__":

    # Initialize image variable
    poster = None

    ## --- Page config ------------ ##
    # Set page title
    st.title("""
    Cardiovascular Disease predictor
    #### This app aims to give a scoring of how probable is that an individual \
    would suffer from a cardiovascular disease given its physical \
         characteristics
    #### Just enter your info and get a prediction.
    """)

    # Set page footer
    footer()

    # Load classification model
    model = CatBoostClassifier()      # parameters not required.
    model.load_model('./train/model.cbm')

    # Define inputs
    input_layout()