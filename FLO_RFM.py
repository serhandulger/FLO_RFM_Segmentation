########################################
# Business Problem
########################################

"""FLO wants to set a roadmap for sales and marketing activities. It is necessary to segment the customers based on the values of them just before making any decision and investment plan for marketing & customer experience strategies.
Our project will help FLO to segment their customers to be able to detect their customer segments based on customers’ values"""
########################################
# Dataset Story
########################################

"""
The dataset consists of information obtained from the past shopping behaviors of customers who
made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.
"""

# master_id: Unique client number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : The date of the customer's first purchase
# last_order_date : The date of the last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : The total price paid by the customer for offline purchases
# customer_value_total_ever_online : The total price paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

df=pd.read_csv('/Users/serhandulger/flo_data_20k.csv')

df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### NA SUM #####################")
    print(dataframe.isnull().sum().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Nunique #####################")
    print(dataframe.nunique())

check_df(df)

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

missing_values_analysis(df)

def data_visualizations(data):
    import missingno as msno
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("VISUALIZING", "\n\n")
    msno.bar(data)
    plt.show()
    #msno.heatmap(data)
    #plt.show()
    msno.matrix(data)
    plt.show()
    print("CORRELATION GRAPH", "\n\n")
    plt.figure(figsize=(14, 12))
    #sns.heatmap(data.corr(), annot=True, cmap="BuPu")
    #plt.show()
    #sns.pairplot(data)
    #plt.show()

data_visualizations(df)

import datetime as dt
df["first_order_date"] = pd.to_datetime(df["first_order_date"]).dt.normalize()
df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.normalize()
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"]).dt.normalize()
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"]).dt.normalize()

def create_date_features(df):
    df['first_order_month'] = df["first_order_date"].dt.month
    df['first_order_day_of_month'] = df["first_order_date"].dt.day
    df['first_order_day_of_week'] = df["first_order_date"].dt.dayofweek
    df['first_order_year'] = df["first_order_date"].dt.year
    df["first_order_is_wknd"] = df["first_order_date"].dt.weekday // 4
    return df

def grab_col_names(dataframe, cat_th=13, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    import seaborn as sns
    import matplotlib.pyplot as plt
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

create_date_features(df)

# We checked the dataset to see if there are any order channels in other years.
sns.catplot(x="order_channel",y="first_order_year",data=df)

cat_cols , num_cols, cat_but_car = grab_col_names(df)

for i in cat_cols:
    print(cat_summary(df,i,plot=True))

df["Total_Order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Total_Payment"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()

df.master_id.nunique()

####################################
# CALCULATION OF RFM METRICS
####################################

df["last_order_date"].max()

import datetime as dt
today_date = dt.datetime(2021,6,2)
today_date

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                     "Total_Order": lambda x: x.sum(),
                                     "Total_Payment": lambda x: x.sum()})
rfm.head(10)

rfm.columns = ["Recency","Frequency","Monetary"]

rfm["recency_score"] = pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"],5,labels=[1,2,3,4,5])

rfm["RFM_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# RFM segment tags
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm["segment"] = rfm["RFM_score"].replace(seg_map,regex=True)
rfm.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=2)
sns.countplot(y=rfm["segment"],data=rfm)
plt.show()

rfm[["segment","Recency","Frequency","Monetary"]].groupby("segment").agg(["count","mean","std","min","max"])

rfm.describe()

rfm

rfm['Monetary'].describe([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

plt.figure(figsize=(12, 6))
plt.title('Distribution of Monetary < 95%')
sns.distplot(rfm[rfm['Monetary']<3606].Monetary);

plt.figure(figsize=(12, 6))
sns.boxplot(x='Recency', data=rfm)
plt.title('Boxplot of Recency');

plt.figure(figsize=(12, 6))
sns.boxplot(x='Frequency', data=rfm)
plt.title('Boxplot of Frequency');

segments_counts = rfm['segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='red')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['champions']:
            bar.set_color('Blue')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left')
plt.show()


###############################
# FUNCTIONALIZE ALL STEPS
###############################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_csv('/Users/serhandulger/flo_data_20k.csv')

def data_preprocessing(dataframe):
    dataframe.dropna(inplace=True)

    dataframe["first_order_date"] = pd.to_datetime(dataframe["first_order_date"]).dt.normalize()
    dataframe["last_order_date"] = pd.to_datetime(dataframe["last_order_date"]).dt.normalize()
    dataframe["last_order_date_online"] = pd.to_datetime(dataframe["last_order_date_online"]).dt.normalize()
    dataframe["last_order_date_offline"] = pd.to_datetime(dataframe["last_order_date_offline"]).dt.normalize()

    dataframe["Total_Order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["Total_Payment"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    return dataframe


def rfm_segmentation(dataframe):
    import datetime as dt
    today_date = dt.datetime(2021, 6, 2)

    rfm = dataframe.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                              "Total_Order": lambda x: x.sum(),
                                              "Total_Payment": lambda x: x.sum()})

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # RFM isimlendirmesi
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'}

    rfm["segment"] = rfm["RFM_score"].replace(seg_map, regex=True)
    return rfm

def visualize_segments(rfm):
    import matplotlib.pyplot as plt
    segments_counts = rfm['segment'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots()

    bars = ax.barh(range(len(segments_counts)),
                  segments_counts,
                  color='red')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                   bottom=False,
                   labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)

    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['champions']:
                bar.set_color('Blue')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                       int(value*100/segments_counts.sum())),
                    va='center',
                    ha='left')
    plt.show()

def rfm_pipeline(dataframe):
    dataframe = data_preprocessing(dataframe)
    rfm = rfm_segmentation(dataframe)
    visualized = visualize_segments(rfm)
    return dataframe , rfm , visualized

dataframe , rfm, visualized = rfm_pipeline(df)

colors  = ("darkorange", "darkseagreen", "orange", "cyan", "cadetblue", "hotpink", "lightsteelblue", "coral",  "mediumaquamarine","palegoldenrod")
explodes = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

rfm["segment"].value_counts(sort=False).plot.pie(colors=colors,
                                                 textprops={'fontsize': 12},
                                                 autopct = '%4.1f',
                                                 startangle= 90,
                                                 radius =2,
                                                 rotatelabels=True,
                                                 shadow = True,
                                                 explode = explodes)
plt.ylabel("");

############################
# BUSINESS SCENARIOS
############################

df = dataframe

df.head(2)

# Top 10 customers with the most profits

df.groupby(["master_id"])["Total_Payment"].sum().sort_values(ascending=False).to_frame("total_profit").reset_index().head(10)

# Top 10 customers with the most orders

df.groupby(["master_id"])["Total_Order"].sum().sort_values(ascending=False).to_frame("total_order").reset_index().head(10)

############################
# BUSINESS CASE 1
############################

"""FLO includes a new women's shoe brand. The product prices of the brand it includes are above the general customer preferences. For this reason, it is desired to contact the customers in the profile that will be interested in the promotion of the brand and product sales. Those who shop from their loyal customers (champions, loyal_customers)
 and women category are the customers to be contacted specifically. Save the id numbers of these customers to the csv file."""

segment1 = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")].reset_index()

segment2 = df[df["interested_in_categories_12"].str.contains("KADIN")]

target_group = pd.merge(segment1,segment2[["interested_in_categories_12","master_id"]], on="master_id")

target_group.head(2)

import researchpy as rp
rp.summary_cat(target_group["segment"])

rp.summary_cat(target_group["interested_in_categories_12"])

target_group["master_id"].to_csv("flo_customer_segment_case_1_id.csv")

############################
# BUSINESS CASE 2
############################

"""Nearly 40% discount is planned for Men's and Children's products. It is aimed to specifically target customers who are good customers in the past, but who have not shopped for a long time, who are interested in the categories
 related to this discount, who should not be lost, those who are asleep and new customers. Save the ids of the customers in the appropriate profile to the csv file."""


case2_segment1 = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")]

case2_segment2 = df[df["interested_in_categories_12"].str.contains("ÇOCUK | ERKEK")]

target_group2 = pd.merge(case2_segment1,case2_segment2[["interested_in_categories_12","master_id"]], on="master_id")

target_group2.head(2)

target_group["master_id"].to_csv("flo_customer_segment_case_1_id.csv")