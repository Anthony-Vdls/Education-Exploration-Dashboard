
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="College Graduate Satisfaction",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# left side bar
with st.sidebar:
    st.title('College Graduate Satisfaction Dashboard')
    st.markdown('## Indoduction:\nThis is a dashboard that showcases collage graduates of all levels and their satisfaction with there jobs after graduation (in the same field as major), as well as the pay diffrences among the same demographic')
    st.markdown('\nThe data is from: https://ncses.nsf.gov/explore-data/microdata/national-survey-college-graduates')

    #  color choices for map and heat map
    color_scales = ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Blues", "Greens", "Reds"]
    color_scale = st.selectbox("Color scale", color_scales, index=0)

    selected_regions_placeholder = st.empty()
    focus_region_placeholder = st.empty()

def make_map(q2_region_very_sat_filtered, color_scale):
    # map regions to states 
    region_to_states = {
        "New England": ["CT","ME","MA","NH","RI","VT"],
        "Middle Atlantic": ["NJ","NY","PA"],
        "East North Central": ["IL","IN","MI","OH","WI"],
        "West North Central": ["IA","KS","MN","MO","NE","ND","SD"],
        "South Atlantic": ["DE","DC","FL","GA","MD","NC","SC","VA","WV"],
        "East South Central": ["AL","KY","MS","TN"],
        "West South Central": ["AR","LA","OK","TX"],
        "Mountain": ["AZ","CO","ID","MT","NV","NM","UT","WY"],
        "Pacific & U.S. Terr.": ["AK","CA","HI","OR","WA"]
    }

    # regional percent its states
    rows = []
    for _, r in q2_region_very_sat_filtered.iterrows():
        reg = r["region"]
        pct = float(r["pct_very_satisfied_overall"]) * 100.0  # show better on map 
        for s in region_to_states.get(reg, []):
            rows.append({"state": s, "region": reg, "pct": pct})
    map_df = pd.DataFrame(rows)

    # standardize legend of states/region
    vmin = map_df["pct"].min()
    vmax = map_df["pct"].max()

    fig = px.choropleth(
        map_df,
        locations="state",
        locationmode="USA-states",
        color="pct",
        scope="usa", # 
        color_continuous_scale=color_scale,
        range_color=(vmin, vmax),
        labels={"pct": "satisfied %"},
        title="Overall satisfaction percentage of College Graduates per Region/State"
    )
    fig.update_coloraxes(colorbar_title="% very satisfied")
    return fig

def make_heatmap(q1_filtered, color_scale):
    if q1_filtered.empty:
        return px.imshow([[0]], labels={"color":"% very satisfied"}, title="No data for selected regions")
    pivot = q1_filtered.pivot(index="region", columns="major_group", values="pct_very_satisfied")
    heat_vals = pivot.values * 100.0  # %
    fig = px.imshow(
        heat_vals,
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale=color_scale,
        labels={"x":"Major group", "y":"Region", "color":"% very satisfied"},
        title="In-field: % Very Satisfied by Region × Major"
    )
    fig.update_coloraxes(colorbar_title="% very satisfied")
    fig.update_layout(margin=dict(l=0,r=0,t=50,b=0))
    return fig

def make_chart(gender_median_filtered, race_median_filtered):
    fig_gender = px.bar(
        gender_median_filtered,
        x="gender", y="median_salary",
        text="median_salary",
        labels={"median_salary":"Median salary", "gender":"Gender"},
        title="Median Salary by Gender of selected regions"
    )
    fig_gender.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_gender.update_yaxes(tickformat="$,")
    fig_gender.update_layout(uniformtext_minsize=10, uniformtext_mode="hide", margin=dict(l=0,r=0,t=50,b=0))

    fig_race = px.bar(
        race_median_filtered,
        x="race_eth", y="median_salary",
        labels={"median_salary":"Median salary", "race_eth":"Race/Ethnicity"},
        title="Median Salary by Race/Ethnicity (in the same field) of selected regions"
    )
    fig_race.update_yaxes(tickformat="$,")
    fig_race.update_layout(xaxis_tickangle=30, margin=dict(l=0,r=0,t=50,b=0))

    return fig_gender, fig_race

# load data and reshape it 

use_cols = [
    "WRKG","RESPLCUS","RESPLOC","NDGMEMG","OCEDRLP",
    "JOBSATIS","SALARY","SEX_2023","RACETHM","AGE"
]
df = pd.read_csv('epcg23.csv', usecols=use_cols)

for col in ["JOBSATIS", "OCEDRLP", "NDGMEMG", "SALARY", "RESPLOC"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[(df["RESPLCUS"] == "Y") & (df["WRKG"] == "Y")].copy()
df.loc[df["SALARY"] >= 9_999_998, "SALARY"] = np.nan

df["is_same_field"] = (df["OCEDRLP"] == 1) # works related field
df["very_sat"] = (df["JOBSATIS"] == 1)     # very satisfied people

major_map = {
    1: "Computer & Math",
    2: "Bio/Ag/Env Life Sci",
    3: "Physical Sciences",
    4: "Social Sciences",
    5: "Engineering",
    6: "S&E-Related",
    7: "Non-S&E"
}
df["major_group"] = df["NDGMEMG"].map(major_map)

region_map = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific & U.S. Terr."
}
df["region"] = df["RESPLOC"].map(region_map)

# Q1 #####################################################
same_field = df[df["is_same_field"]].copy()
q1 = (same_field
      .groupby(["region","major_group"])["very_sat"]
      .mean()
      .reset_index(name="pct_very_satisfied"))
q1["rank_in_region"] = q1.groupby("region")["pct_very_satisfied"].rank(ascending=False, method="dense")
q1_top3_by_region = q1[q1["rank_in_region"] <= 3].sort_values(
    ["region","rank_in_region","pct_very_satisfied"], ascending=[True, True, False]
)

# Q2 #####################################################
q2_region_very_sat = (df
    .groupby("region")["very_sat"]
    .mean()
    .reset_index(name="pct_very_satisfied_overall"))

# Q3 ######################################################
gender_median = (
    same_field.groupby("SEX_2023")["SALARY"]
    .median()
    .reset_index()
    .rename(columns={"SEX_2023":"gender", "SALARY":"median_salary"})
)
race_map = {
    1: "Asian (NH)",
    2: "Am. Indian/AK Native (NH)",
    3: "Black (NH)",
    4: "Hispanic (Any)",
    5: "White (NH)",
    6: "NH/Other Pacific Islander (NH)",
    7: "Multiple Race (NH)"
}
same_field["race_eth"] = same_field["RACETHM"].map(race_map)
race_median = (
    same_field.groupby("race_eth")["SALARY"]
    .median()
    .reset_index()
    .rename(columns={"SALARY":"median_salary"})
    .sort_values("median_salary", ascending=False)
)

#print(q1_top3_by_region.head(15))
#print(q2_region_very_sat.head())
#print(gender_median)
#print(race_median)

# sidebar widgets effect choropleth map and bar charts 
regions_available = sorted([r for r in df["region"].dropna().unique()])
selected_regions = selected_regions_placeholder.multiselect(
    "Filter regions (Doesnt effect US Map)",
    regions_available,
    default=regions_available
)
focus_region = focus_region_placeholder.selectbox(
    "Focus region",
    ["All US"] + regions_available,
    index=0
)

# filters for interactive plots 
df_sel = df[df["region"].isin(selected_regions)]
same_field_sel = df_sel[df_sel["is_same_field"]]

q1_sel = (same_field_sel
          .groupby(["region","major_group"])["very_sat"]
          .mean()
          .reset_index(name="pct_very_satisfied"))

gender_median_sel = (
    same_field_sel.groupby("SEX_2023")["SALARY"]
    .median().reset_index().rename(columns={"SEX_2023":"gender","SALARY":"median_salary"})
)
same_field_sel = same_field_sel.assign(race_eth=same_field_sel["RACETHM"].map(race_map))
race_median_sel = (
    same_field_sel.groupby("race_eth")["SALARY"]
    .median().reset_index().rename(columns={"SALARY":"median_salary"})
    .sort_values("median_salary", ascending=False)
)

# focus region vs national avg
nat_avg = float(q2_region_very_sat["pct_very_satisfied_overall"].mean()) * 100.0
if focus_region == "All US":
    focus_val = nat_avg
else:
    m = q2_region_very_sat["region"] == focus_region
    focus_val = float(q2_region_very_sat.loc[m, "pct_very_satisfied_overall"].iloc[0] * 100.0) if m.any() else np.nan
delta = focus_val - nat_avg if pd.notna(focus_val) else 0.0

#  Dashboard Columns
col = st.columns((3,1.5), gap='medium')

with col[0]:

    st.markdown("""
    ### US Map and Bar Charts 
    - The number above the map shows how your chosen region stacks up to the U.S. average.
    - Darker regions on the map = more people say they are **very satisfied** at workoverall. 
    - The pay bars show **typical pay** by gender and by race/ethnicity for people working in the field they studied.
"""
                )
    # KPI
    st.metric(label="Overall % Very Satisfied", value=f"{focus_val:.1f}%", delta=f"{delta:+.1f} pts vs US")

    fig_map = make_map(q2_region_very_sat, color_scale)
    st.plotly_chart(fig_map, use_container_width=True)

    fig_gender, fig_race = make_chart(gender_median_sel, race_median_sel)
    st.plotly_chart(fig_gender, use_container_width=True)
    st.plotly_chart(fig_race, use_container_width=True)

with col[1]:

    st.markdown("""
    ### What the grid shows
    - In each region, the **darkest boxes** mark the fields where people feel happiest in their jobs.
    - Select the displayed Regios on the left hand panel. 
    """
                )

    fig_heat = make_heatmap(q1_sel, color_scale)
    st.plotly_chart(fig_heat, use_container_width=True)

