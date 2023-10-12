import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices, dmatrix


df = pd.read_csv('wo_men.csv')
df = df.reindex(columns=['sex', 'shoe_size', 'height'])

st.title("LETS GUESS YOUR HEIGHT")
text = """designed by 
Andrea
Felix
Tyler
Solomon
"""
st.text(text)


st.subheader("Predictive Power of MLR using 2 Predictors")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Shoe Size")
    st.image("images/shoe.png", width=200)

with col3:
    st.subheader("Gender")
    st.image("images/gender.webp", width=200)

with col2:
    st.image("images/plus.png", width=200)

st.subheader("Important Libraries")
st.code("""import matplotlib.pyplot 
import pandas 
import numpy
import statsmodels.api
""")
st.subheader("Sample Data")

st.dataframe(df.head(10))

st.subheader("Model")

######## Check VIF ######
y, X = dmatrices('height ~ shoe_size + sex', data=df, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
#######

st.code("""model = smf.ols('height ~ shoe_size + sex', df).fit()
model.summary()
""")


model = smf.ols('height ~ shoe_size + sex', df).fit()
summary = model.summary()

st.write(summary)


p = model.fittedvalues
res = model.resid
data = pd.DataFrame({'Fitted Values': p, 'Residuals': res})


st.subheader("Residuals")
fig = px.scatter(data, x='Fitted Values', y='Residuals',
                 title="Fitted Values vs. Residuals")
fig.update_traces(marker=dict(size=8, color='Blue'))
st.plotly_chart(fig)

infl = model.get_influence()
n = len(df)
p = model.df_model + 1
inflsum = infl.summary_frame()
reg_cook = inflsum.cooks_d

atyp_cook = np.abs(reg_cook) >= 4/n

st.subheader("Index of Residuals (Drop them like its hot)")
st.write(df.index[:-1][atyp_cook])

df_trim = df.drop(df.index[:-1][atyp_cook])
st.code("df_trim = df.drop(df.index[:-1][atyp_cook])")

st.subheader("Updated Model")
model_t = smf.ols('height ~ shoe_size + sex', df_trim).fit()
model_t.summary()

st.write(model_t.summary())

p = model_t.fittedvalues
res = model_t.resid
data = pd.DataFrame({'Fitted Values': p, 'Residuals': res})

st.subheader("Updated Chart of Residuals")
fig = px.scatter(data, x='Fitted Values', y='Residuals',
                 title="Fitted Values vs. Residuals")
fig.update_traces(marker=dict(size=8, color='Red'))
st.plotly_chart(fig)

st.header("LETS GUESS SOME HEIGHTS")

sex = ["Man", "Woman"]
gender = st.radio("Gender", sex)

if gender == "Man":
    sex = ['man']
else:
    sex = ['woman']


shoe_size = st.text_input(
    " ", placeholder='Shoe Size in EU Size', label_visibility="hidden")

try:
    shoe_size = float(shoe_size)
except ValueError:
    st.error("Please put in a Valid Shoe Size")

try:
    df_p = pd.DataFrame({'sex': sex, 'shoe_size': [shoe_size]})
    predictions = model_t.get_prediction(df_p)
    prediction = predictions.summary_frame(alpha=0.05)
    st.write(prediction)
except AttributeError:
    print(" ")
