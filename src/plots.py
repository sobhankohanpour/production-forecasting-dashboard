import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def scatterplot(dataset, x, y, hue=None, title="Scatter Plot"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=dataset, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def lineplot(dataset, x, y, hue=None, title="Line Plot"):
    fig, ax = plt.subplots()
    sns.lineplot(data=dataset, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def distplot(dataset, x, hue=None, title="Distribution Plot"):
    g = sns.displot(data=dataset, x=x, hue=hue)
    plt.suptitle(title)
    st.pyplot(g.fig)


def histplot(dataset, x, hue=None, bins=30, title="Histogram"):
    fig, ax = plt.subplots()
    sns.histplot(data=dataset, x=x, hue=hue, bins=bins, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def kdeplot(dataset, x, hue=None, fill=True, title="KDE Plot"):
    fig, ax = plt.subplots()
    sns.kdeplot(data=dataset, x=x, hue=hue, fill=fill, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def ecdfplot(dataset, x, hue=None, title="ECDF Plot"):
    fig, ax = plt.subplots()
    sns.ecdfplot(data=dataset, x=x, hue=hue, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def rugplot(dataset, x, hue=None, title="Rug Plot"):
    fig, ax = plt.subplots()
    sns.rugplot(data=dataset, x=x, hue=hue, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)



