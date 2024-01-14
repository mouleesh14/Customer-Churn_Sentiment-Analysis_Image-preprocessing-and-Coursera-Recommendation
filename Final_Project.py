import streamlit as st
import re
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import googletrans
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from PIL import Image,ImageFilter,ImageEnhance,ImageOps
import easyocr
import cv2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

st.set_page_config(layout='wide',page_title="Final_Project")

# adding background image for the streamlit
def back_img(image):
    with open(image, "rb") as image_file:
        encode_str = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encode_str.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

back_img("img4.jpg") 

# Recommendation function and recommend the courses
def recommd(course_data,name):
    item_matrix=pd.read_csv("recommend.csv")
    a=[]
    for j in name:
        a.append(course_data[course_data["Course Name"]==j]["course_name_label"].values)
    university=[]
    for k in a:
        university.append(list(item_matrix.loc[int(k)].sort_values(ascending=False).iloc[:20].index))
    for d,n in zip(university,name):
        st.write(f"""<p>Course Name: <b>{n}</b>""",unsafe_allow_html=True)
        name=[]
        un=[]
        rate=[]
        skill=[]
        link=[]
        for f in d:
            name.append(course_data[course_data["course_name_label"]==int(f)]["Course Name"].values[0])
            un.append(course_data[course_data["course_name_label"]==int(f)]["University"].values[0])
            rate.append(course_data[course_data["course_name_label"]==int(f)]["Course Rating"].values[0])
            skill.append(course_data[course_data["course_name_label"]==int(f)]["Skills"].values[0])
            link.append(course_data[course_data["course_name_label"]==int(f)]["Course URL"].values[0])
        r=pd.DataFrame({"Course Name":name,"University":un,"Course Rating":rate,"Skills":skill,"Course URL":link})
        st.write("""<b>Recommended Courses:</b>""",unsafe_allow_html=True)
        st.dataframe(r.sort_values("Course Rating",ascending=False),hide_index=True)
    
# Drop the unwanted columns and preform the sparcity and fill null values
def zero_preprocessing(cls_data):
    cls_data1=cls_data.copy()
    zero=[]
    for z in cls_data1.columns:
        value=((cls_data1[z]==0).mean()*100).round(2)
        zero.append(value)
    zero_df=pd.DataFrame({"Column_name":cls_data1.columns,"Zero_Percentage":zero}).sort_values("Zero_Percentage",ascending=False)
    col_to_rem=["youtube","days_since_last_visit","bounces","totals_newVisits","latest_isTrueDirect",
         "earliest_isTrueDirect","time_latest_visit","time_earliest_visit","device_isMobile","device_browser","device_operatingSystem","last_visitId","latest_visit_id",
        "visitId_threshold","earliest_visit_id","earliest_visit_number","latest_visit_number","days_since_first_visit",
      "earliest_source","latest_source","earliest_medium","latest_medium","earliest_keyword","latest_keyword",
        "device_deviceCategory","channelGrouping","geoNetwork_region","target_date","bounce_rate","historic_session_page","avg_session_time_page","products_array"]
    cls_data1.drop(col_to_rem,axis=1,inplace=True)
    for spar in cls_data1.columns:
        me=cls_data1[spar].mean()
        if spar=="has_converted" or spar=="transactionRevenue":
            continue
        values=[]
        for spar_val in cls_data1[spar].values:
            if spar_val<=0:
                values.append(me)
            else:
                values.append(spar_val)
        cls_data1[spar]=values
    zero1=[]
    for z in cls_data1.columns:
        value=((cls_data1[z]==0).mean()*100).round(2)
        zero1.append(value)
    zero_df_pre=pd.DataFrame({"Column_name":cls_data1.columns,"Zero_Percentage":zero1}).sort_values("Zero_Percentage",ascending=False)
    return zero_df,zero_df_pre,cls_data1   

# It will treat the outlier in all the columns by using log1p
def outlier(cls_data):
    cls_data2=cls_data.copy()
    for out in cls_data2.columns:
        if out == "transactionRevenue":
            cls_data2[out]=np.log1p(cls_data2[out])
        elif out == "has_converted":
            continue
        else:
            cls_data2[out],_=stats.boxcox(cls_data2[out])
    return cls_data2

# Predict that user will convert or not
def predict(l,t):
    l,_=stats.boxcox(l)
    t=np.log1p(t)
    li=[]
    for i in l:
        li.append(i)
    li.append(t)
    with open("log_reg.pkl","rb") as lg:
        lg=pickle.load(lg)
    predicted=lg.predict([li])
    return predicted

# Calculate the metrics for svm,knn and logistic regression
def model_predict(data,tar,model):
    train_data, test_data, train_lab, test_lab = train_test_split(data,tar,test_size=0.2,random_state=42)
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)
    train_acc = (accuracy_score(train_lab, train_pred)*100).round(2)
    train_prec = (precision_score(train_lab, train_pred)*100).round(2)
    train_recall = (recall_score(train_lab, train_pred)*100).round(2)
    train_f1 = (f1_score(train_lab, train_pred)*100).round(2)
    test_acc = (accuracy_score(test_lab,test_pred)*100).round(2)
    test_prec = (precision_score(test_lab,test_pred)*100).round(2)
    test_recall = (recall_score(test_lab,test_pred)*100).round(2)
    test_f1score = (f1_score(test_lab,test_pred)*100).round(2)
    return train_acc,train_prec,train_recall,train_f1,test_acc,test_prec,test_recall,test_f1score

# Loading the dataset from the local system and returning the requried values
def getting_data():
    cls_data=pd.read_csv("classification_data.csv")
    sh=cls_data.shape
    is_null=(cls_data.isnull().mean()*100).round(2)
    is_null_df=pd.DataFrame({"Column_Name":is_null.index,"Null_Percentage":is_null.values}).sort_values("Null_Percentage",ascending=False)
    cls_data.drop_duplicates(inplace=True)
    sh1=cls_data.shape
    des=cls_data.describe()
    zero_df,zerodf_pre,cls_data1=zero_preprocessing(cls_data)
    return sh,sh1,des,is_null_df,zero_df,zerodf_pre,cls_data1,cls_data

# Return image details like format, size and array
def image_details(img):
    f=img.format
    h=img.size[0]
    w=img.size[1]
    arr=np.array(img)
    arr_size=arr.shape
    m=img.mode
    return f,h,w,arr,arr_size,m

with st.sidebar:
    st.sidebar.image("profile5.png",use_column_width=True,width=500)     
    opt = option_menu("Final Project",["Prediction","Image Preprocessing","Text Preprocessing","Recommendation"],menu_icon="cast",styles={"container": {"padding":"4!important"},"nav-link": {"text-align":"left"},"nav-link-selected": {"background-color": "#C2452D"}})
    
if opt=="Prediction":
    st.markdown("<h2><FONT COLOR='#000000'>Predicton</h3>",unsafe_allow_html=True)
    with st.expander("ABOUT"):
        st.write("""<p>Classification is a type of machine learning task where the goal is to predict the category or class of a given input based on its features. Data preprocessing follows, involving tasks such as handling missing values, encoding categorical variables, and scaling features to ensure the dataset's quality and consistency.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Task Definition:</b> Clearly define the classification task, including the classes or categories you want to predict.""",unsafe_allow_html=True)
        st.write("""<p><b>Data Collection:</b> Gather a dataset that includes examples of input features along with their corresponding class labels.""",unsafe_allow_html=True)
        st.write("""<p><b>Data Preprocessing:</b> Clean and preprocess the data. This may involve handling missing values, encoding categorical variables, and scaling numerical features.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Feature Extraction:</b> Identify and select relevant features from the input data that will be used to train the model.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Split the Data:</b> Split the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Choose a Classification Algorithm:</b> Select a classification algorithm suitable for your data and problem. Common algorithms include decision trees, support vector machines, logistic regression, k-nearest neighbors, and neural networks.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Train the Model:</b> Use the training data to train the classification model. The model learns the patterns and relationships in the data that allow it to make predictions.""",unsafe_allow_html=True)
        st.write("""<p><b>Hyperparameter Tuning:</b> Fine-tune the hyperparameters of your model to optimize its performance. This may involve using techniques like cross-validation.""",unsafe_allow_html=True)
    shap,shape,des,isnull,zerodf,zerodf_pre,cls_data,map_data= getting_data()
    tab1,tab2,=st.tabs(["EDA","Prediction"])
    with tab1:
        st.markdown("<h3><FONT COLOR='#000000'>Exploratory Data Analysis (EDA)</h2>",unsafe_allow_html=True)
        st.write("<p>EDA, or Exploratory Data Analysis, is a critical phase in the data analysis process that involves visually and statistically exploring and summarizing key characteristics, patterns, and trends within a dataset. The primary objectives of EDA are to uncover insights, identify relationships, and gain an understanding of the structure and distribution of the data. This process is crucial for informing subsequent steps in the data analysis pipeline, such as feature engineering, modeling, and hypothesis testing.</p>",unsafe_allow_html=True)
        map_data["geoNetwork_region"].replace('Decentralized Administration of Peloponnese, Western Greece and the Ionian', 'Western Greece', inplace=True)
        st.markdown("<h4><FONT COLOR='#000000'>User's Geolocations:</h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            fig = px.scatter_mapbox(map_data,lat="geoNetwork_latitude",lon= "geoNetwork_longitude",color="has_converted",title="User's Geolocation",width=600, height=400)
            fig.update_layout(mapbox_style = "carto-positron")
            st.plotly_chart(fig)
        with c2:
            st.write("")
        with c3:
            st.write("""<p>This map represents the user's location where they interact with the website and colored dots represents the user has converted or not.</p>""",unsafe_allow_html=True)
            region=map_data["geoNetwork_region"].value_counts()
            region=pd.DataFrame(region)
            st.dataframe(region)
        st.markdown("<h4><FONT COLOR='#000000'>Devices used:</h2>",unsafe_allow_html=True)
        st.write("You can select any types of feature to visualize the value counts and the distributions.")
        sel=st.multiselect("Select the features: ",options=["device_deviceCategory","device_operatingSystem","device_browser"],key="select1")
        if len(sel)==0:
            st.write("Please select the features.")
        else:
            c1,c2,c3=st.columns(3)
            with c1:
                st.markdown("""<h4>User's Device details</h4> """,unsafe_allow_html=True)
                fig=px.sunburst(map_data, path=sel,title="Device Counts",width=600, height=400)
                st.plotly_chart(fig)
            with c2:
                st.write("")
            with c3:
                st.write("""<p>This pie chart and table represents the selected column classes and their values counts for each features</p>""",unsafe_allow_html=True)
                ab=map_data.groupby(sel)[sel].value_counts()
                ab=pd.DataFrame(ab,columns=["Count"]).sort_values("Count",ascending=False)
                st.dataframe(ab)
        st.markdown("<h4><FONT COLOR='#000000'>Source used:</h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            fig = px.pie(map_data, names='latest_source',hole=.5,width=600, height=400,title="Source Count")
            fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)
        with c2:
            st.write("")
        with c3:
            st.write("""<p>This pie chart and tables represents that the user viewed this website and their counts for each source</p> """,unsafe_allow_html=True)
            source=map_data["latest_source"].value_counts()
            source=pd.DataFrame(source)
            st.dataframe(source)
        st.markdown("<h4><FONT COLOR='#000000'>Revenue Based on region:</h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            ac=map_data.groupby(["geoNetwork_region"])["transactionRevenue"].sum()
            ac=pd.DataFrame(ac)
            fig = px.bar(ac, x=ac.index, y="transactionRevenue",color=ac.index,width=600, height=400,title="Region's Revenue")
            st.plotly_chart(fig)
        with c2:
            st.write("")
        with c3:
            
            ac=ac.sort_values("transactionRevenue",ascending=False)
            st.dataframe(ac)
        st.markdown("<h4><FONT COLOR='#000000'>Revenue based on source:</h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            ac=map_data.groupby(["latest_source"])["transactionRevenue"].sum()
            ac=pd.DataFrame(ac)
            fig = px.bar(ac, x=ac.index, y="transactionRevenue",color=ac.index,width=600, height=400,title="Source Revenue")
            st.plotly_chart(fig)
        with c2:
            st.write("")
        with c3:
            ac=ac.sort_values("transactionRevenue",ascending=False)
            st.dataframe(ac)
        st.markdown("<h3><FONT COLOR='#000000'>Revenue based on interactions:</h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            fig = px.line(map_data, x="num_interactions", y="transactionRevenue", color='latest_source',symbol="latest_source",width=600,height=400,title="Revenue based on no.of interactions")
            st.plotly_chart(fig)
        with c2:
            st.write("")
        with c3:
            ac=map_data.groupby(["latest_source"])["num_interactions"].sum()
            ac=pd.DataFrame(ac).sort_values("num_interactions",ascending=False)
            st.dataframe(ac)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4><FONT COLOR=#000000>Shape of Dataset</h4>""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Rows: </b>{shap[0]}""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Columns: </b>{shap[1]}""",unsafe_allow_html=True)
            st.write("""In this 100000 rows there are 90793 duplicates, for model building we can delete all duplicates from the dataset. So that classification will perform better""")
            st.write("""<h4><FONT COLOR=#000000>After Removing all Duplicates: </h5>""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Rows: </b>{shape[0]}""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Columns: </b>{shape[1]}""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4><FONT COLOR=#000000>Aggregate Values</h4>""",unsafe_allow_html=True)
            st.write("""<p>From this table we can take the mean, min, max, etc... for all the numerical value columns</p>""",unsafe_allow_html=True)
            st.dataframe(des)
        st.write(" ")
        st.markdown("""<h3><FONT COLOR='#000000'>Data Preprocessing</h3>""",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("""<h4><FONT COLOR:'#000000'>Null Percentage</h4>""",unsafe_allow_html=True)
            st.write("""<P>From this table there is no empty values. So we don't need to make any changes</p>""",unsafe_allow_html=True)
            st.dataframe(isnull,hide_index=True)
        with c2:
            st.markdown("""<h4><FONT COLOR:'#000000'>Sparsity Data</h4>""",unsafe_allow_html=True)
            st.write("""<p>In this sparsity table the columns having more than 50 percent of zero values we can delete that columns""",unsafe_allow_html=True)
            st.dataframe(zerodf,hide_index=True)
        with c3:
            st.write("")
            st.write("")
            st.write("")
            st.write("""<p><b>Note:</b> The column has_converted is the target columns so we should not make any changes and removed all non-numeric values.</p>""",unsafe_allow_html=True)
            st.dataframe(zerodf_pre,hide_index=True)
        st.markdown("""<h4><FONT COLOR:'#000000'>Outlier Detection: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.write("""<p>  It's an unusually extreme value that lies outside the typical range of values in a dataset. Identifying outliers is important in machine learning</p> """,unsafe_allow_html=True)
            st.plotly_chart(px.box(cls_data))
            st.write("")
        with c2:
            cls_data1=outlier(cls_data)
            st.write("""<p> We can treat the outliers by changing the values using boxcox method. After treating the outlier the mean,median and mode values will be in the plot</p> """,unsafe_allow_html=True)
            st.plotly_chart(px.box(cls_data1))
        st.markdown("""<h4><FONT COLOR:'#000000'>Distributation Curve: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.write("""<p>This graphs are ploted to show the distribution of values for induvial columns. Most of the columns are Right Skewed this is not normaly distributed</p>""",unsafe_allow_html=True)
            on = st.toggle('View Distribution curve',key="on1")
            if on:
                st.set_option('deprecation.showPyplotGlobalUse', False)              
                for i in cls_data.columns:
                    sns.set(style="whitegrid")
                    plt.figure(figsize=(10, 6),)
                    sns.displot(cls_data[i],kind='kde')
                    st.pyplot()
        with c2:
            st.write("""<p>Here we used boxcox and log1p method to make right skewd graph to normal distribution</p>""",unsafe_allow_html=True)
            on = st.toggle('View Distribution curve',key="on2")
            if on:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                for i in cls_data.columns:
                    sns.set(style="whitegrid")
                    plt.figure(figsize=(10, 6))
                    sns.displot(cls_data1[i],kind='kde')
                    st.pyplot()
                
        st.markdown("""<h4><FONT COLOR:'#000000'>Correlation Heatmap: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            cls_data2=cls_data1.copy()
            cls_data2.drop("has_converted",axis=1,inplace=True)
            st.write("""<p>Correlation heatmap for all continous variables that quantifies the degree to which two variables are related </p>""",unsafe_allow_html=True)
            st.write("")
            corr_data = cls_data2.corr()
            fig = px.imshow(corr_data,x=corr_data.columns,y=corr_data.columns,color_continuous_scale='Viridis', title='Correlation Heatmap')
            st.plotly_chart(fig)
        with c2:
            cls_data2.drop(["count_session","count_hit","historic_session","single_page_rate"],axis=1,inplace=True)
            st.write("""<p> From the previous correlation map the columns <b>count session,count hit, historic_session, single_page_rate</b> are having highest correaltion, we can remove that columns</p>""",unsafe_allow_html=True)
            corr_data = cls_data2.corr()
            fig = px.imshow(corr_data,x=corr_data.columns,y=corr_data.columns,color_continuous_scale='Viridis', title='Correlation Heatmap')
            st.plotly_chart(fig)
        st.markdown("""<h4><FONT COLOR:'#000000'>Feature Importance: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ran_class=RandomForestClassifier(n_estimators=20,random_state=44)
            over_data=cls_data2.copy()
            over_tar=cls_data1['has_converted']
            ran_class.fit(over_data,over_tar)
            val=ran_class.feature_importances_*100
            feature_df=pd.DataFrame({"Columns":over_data.columns,"Feature_percentage":val}).sort_values("Feature_percentage",ascending=False)
            st.dataframe(feature_df)
            over_data.drop(["sessionQualityDim","avg_visit_time","geoNetwork_longitude","geoNetwork_latitude"],axis=1,inplace=True)
        with c2:
            st.write("""<p>In this Feature Importance we can understand the importance of the columns in the dataset.</p>""",unsafe_allow_html=True)
            st.write("""<p>We can remove the sessionqualityDim, geonetwork latitude, geonetwork longititude, avg_visit_time columns from the dataset. It will be useful for the model building. </p>""",unsafe_allow_html=True)
        st.markdown("""<h4><FONT COLOR:'#000000'>Data Spreading:</h4>""",unsafe_allow_html=True)
        st.write("""<p>Data visualization that displays individual data points as markers on a two-dimensional graph. Each point on the graph represents the values of two variables, one plotted along the x-axis and the other along the y-axis.</p.""",unsafe_allow_html=True)
        on = st.toggle('View',key="on_button1")
        if on:
            d=cls_data2.copy()
            d["has_converted"]=cls_data1["has_converted"]
            d.drop(["sessionQualityDim","avg_visit_time","geoNetwork_longitude","geoNetwork_latitude"],axis=1,inplace=True)
            fig = px.scatter_matrix(map_data, dimensions=["avg_session_time","visits_per_day","num_interactions","time_on_site","transactionRevenue"], color="has_converted",width=1200,height=800)
            st.plotly_chart(fig)            
    with tab2:
        st.write("""<h3>Prediction</h4>""",unsafe_allow_html=True)
        st.write("""<p><b>NOTE: </b>Kindly enter only positive values and dont't enter negative value or zero.""",unsafe_allow_html=True)
        c1,c2,c3,c4,c5=st.columns(5)
        with c1:
            ses=st.number_input("Enter the average session value",)
        with c2:
            visit=st.number_input("Enter the visits per day:",)
        with c3:
            inter=st.number_input("Enter the number of interactions:")
        with c4:
            time=st.number_input("Enter the time on site value:")
        with c5:
            tras=st.number_input("Enter the transaction revenue:")
        if st.button("Predict",key="bt1"):
            lis=[ses,visit,inter,time]
            li=predict(lis,tras)
            st.write(f"""<b>Avereage Session Time: </b>{ses}""",unsafe_allow_html=True)
            st.write(f"""<b>Visits per day: </b>{visit}""",unsafe_allow_html=True)
            st.write(f"""<b>Number of interactions: </b>{inter}""",unsafe_allow_html=True)
            st.write(f"""<b>Time on site</b>: {time}""",unsafe_allow_html=True)
            st.write(f"""<b>ransaction Revenue</b>: {tras}""",unsafe_allow_html=True)
            if li[0]==0:
                st.write("""<h4><FONT COLOR='red'>User Not Converted</h4>""",unsafe_allow_html=True)
            else:
                st.write("""<h4><FONT COLOR='green'>User Converted</b></h4>""",unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.markdown("""<h3>Model Result</h3>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>Logistic Regression</h3>""",unsafe_allow_html=True)
            st.write("""<p>Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome. </p>""",unsafe_allow_html=True)
            st.write("""<p>Logistic regression is a useful analysis method for classification problems, where you are trying to determine if a new sample fits best into a category. As aspects of cyber security are classification problems, such as attack detection, logistic regression is a useful analytic technique.</p>""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4>Metrics Score</h4>""",unsafe_allow_html=True)
            with open("log_reg.pkl","rb") as lg:
                lg_model=pickle.load(lg)
            tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,lg_model)
            score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
            st.dataframe(score_df)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>Support Vector Machine</h3>""",unsafe_allow_html=True)
            st.write("""<p>A support vector machine (SVM) is a type of supervised learning algorithm used in machine learning to solve classification and regression tasks; SVMs are particularly good at solving binary classification problems, which require classifying the elements of a data set into two groups.</p>""",unsafe_allow_html=True)
            st.write("""<p>The sigmoid kernel is widely applied in neural networks for classification processes. The SVM classification with the sigmoid kernel has a complex structure and it is difficult for humans to interpret and understand how the sigmoid kernel makes classification decisions.</p>""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4>Metrics Score</h4>""",unsafe_allow_html=True)
            with open("svm_model.pkl","rb") as sv:
                sv_model=pickle.load(sv)
            tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,sv_model)
            score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
            st.dataframe(score_df)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>KNN Algorithm</h3>""",unsafe_allow_html=True)
            st.write("""<p>The k-nearest neighbors algorithm, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.</p>""",unsafe_allow_html=True)
            st.write("""<p>For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature.</p>""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4>Metrics Score</h3>""",unsafe_allow_html=True)
            with open("knn.pkl","rb") as lg:
                knn_model=pickle.load(lg)
            tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,knn_model)
            score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
            st.dataframe(score_df)
            
if opt=="Image Preprocessing":
    st.markdown("""<h3><FONT COLOR:'#000000'>Image Preprocessing</h3>""",unsafe_allow_html=True)
    select = option_menu(None,["About","Preprocess"],orientation="horizontal",key="image_side")
    if select=="About":
        st.markdown("""<h4>About</h4>""",unsafe_allow_html=True)
        st.write("""<p>Image preprocessing is a crucial step in computer vision and image analysis pipelines, aiming to enhance the quality of input images and facilitate more accurate and efficient processing by machine learning algorithms. This pre-processing involves a series of operations that address issues such as noise reduction, contrast adjustment, and normalization.</p>""",unsafe_allow_html=True)
        st.write("""<p> Common techniques include resizing images to a standard resolution, converting them to grayscale, and applying filters for smoothing or sharpening. Additionally, methods like histogram equalization can be employed to balance the distribution of pixel intensities. Image normalization ensures that pixel values fall within a specific range, often between 0 and 1, making images more suitable for machine learning models.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Grayscale Conversion</h4>""",unsafe_allow_html=True)
        st.write("""<p>Grayscale conversion is a fundamental image processing technique that involves transforming a color image into a black-and-white representation. In a grayscale image, each pixel is represented by a single intensity value, typically ranging from 0 (black) to 255 (white), with shades of gray in between. This conversion simplifies the image by removing color information, making it easier to analyze and reducing the computational complexity of subsequent tasks. The conversion process can be achieved through various methods, such as taking the average of the RGB (Red, Green, Blue) values or using weighted combinations to preserve certain color channels' contributions.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Resizing</h4>""",unsafe_allow_html=True)
        st.write("""<p>Resizing an image is a fundamental operation in image processing that involves altering its dimensions, either to reduce or increase its size. This process is crucial for various applications, including web development, computer vision, and machine learning. Resizing is often performed to meet specific requirements, such as fitting an image into a designated display area, reducing file size for efficient storage, or preparing data for training machine learning models with consistent input sizes. </p>""",unsafe_allow_html=True)
        st.write("""<p>Common resizing techniques include bilinear or bicubic interpolation, which calculate new pixel values based on the original image's surrounding pixels. When downscaling, these methods help maintain image quality by smoothing transitions between pixels. Conversely, when upscaling, they interpolate to create additional pixels and avoid a blocky appearance.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Bluring Image </h4>""",unsafe_allow_html=True)
        st.write("""<p>This operation is widely used in various applications, including photography, computer vision, and graphics. There are different types of blurring filters, such as Gaussian blur, median blur, and bilateral blur, each with its specific characteristics and use cases.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Gaussian Blur: </b> Applies a convolution with a Gaussian kernel to the image, smoothing out high-frequency noise and details. This is particularly useful for reducing noise in images and creating a soft, defocused effect.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Median Blur: </b> It replaces each pixel's value with the median value of its neighboring pixels. This technique is effective in preserving edges while removing salt-and-pepper noise from an image.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Bilateral Blur: </b> It considers both spatial and intensity differences between pixels, allowing for the preservation of edges while smoothing homogeneous regions. This makes it suitable for applications where maintaining important details is critical.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Contrast Image </h4>""",unsafe_allow_html=True)
        st.write("""<p>Contrast is a crucial aspect of visual perception and plays a significant role in the overall quality and interpretability of images. Low contrast can result in a dull or washed-out appearance, while high contrast can lead to overly pronounced differences between bright and dark regions.</p>""",unsafe_allow_html=True)
        st.write("""<p>Contrast enhancement is achieved through various methods, such as histogram equalization, stretching, and normalization. Histogram equalization redistributes pixel intensities across the entire dynamic range, enhancing the overall contrast by making optimal use of available intensity levels. Contrast stretching involves linearly scaling pixel values to span the full range of intensities, effectively stretching the distribution to increase contrast. Normalization adjusts pixel values to a standard range, often between 0 and 255, facilitating consistent contrast levels across different images.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Edge Detection </h4>""",unsafe_allow_html=True)
        st.write("""<p>The objective is to highlight areas where abrupt transitions or discontinuities exist, which often correspond to object boundaries or meaningful features. This process is essential for various computer vision tasks, including object recognition, image segmentation, and scene understanding. Edge detection is a crucial preprocessing step in computer vision applications, enabling subsequent analysis and interpretation of images. It allows algorithms to focus on relevant details and shapes, improving the efficiency and accuracy of tasks such as object recognition or image segmentation. </p>""",unsafe_allow_html=True)
        st.write("""<p>There are several edge detection algorithms, with some of the most widely used including the Sobel operator, Canny edge detector, and Prewitt operator. These algorithms typically involve convolving the image with specific convolution kernels to emphasize changes in intensity. The resulting gradient information helps identify the locations of edges and their orientations.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Sharping Image </h4>""",unsafe_allow_html=True)
        st.write("""<p> Image sharpening is widely used in photography, medical imaging, and various computer vision applications where the clarity of edges and details is crucial for accurate analysis and interpretation. The goal is to strike a balance that enhances visual quality without sacrificing the integrity of the original image. Sharpness is often lost during image acquisition or compression, and image sharpening helps to address this by enhancing the high-frequency components.</p>""",unsafe_allow_html=True)
        st.write("""<p>Various algorithms and filters are employed for image sharpening, with one of the common methods being the application of convolution kernels, such as the Laplacian or high-pass filters. These filters accentuate rapid changes in intensity, highlighting edges and fine details. Unsharp Masking (USM) is another popular technique where a blurred version of the original image is subtracted from the original, enhancing edges and producing a sharpened effect.</p>""",unsafe_allow_html=True)
        st.markdown("""<h4>Negative Image </h4>""",unsafe_allow_html=True)
        st.write("""<p>Negative images have both artistic and practical applications. Artistically, they can be used to evoke a surreal or abstract feel, offering a fresh perspective on familiar scenes. In practical terms, negative images are sometimes employed in medical imaging to enhance certain features or abnormalities that might not be as visible in the original positive image. The technique can also be applied in quality control and industrial inspections to highlight defects or anomalies in materials.</p>""",unsafe_allow_html=True)
        st.write("""<p>In a negative image, the colors are inverted, turning light areas dark and vice versa. This transformation involves changing each pixel's color values to their complementary values. For example, in an RGB image, the red, green, and blue channels are individually inverted. The result is a striking visual effect where the entire image appears as a photographic negative, resembling the look of traditional film negatives.</p>""",unsafe_allow_html=True)
    if select=="Preprocess":
        st.write("""<h5>Select any image</h5>""",unsafe_allow_html=True)
        input_file=st.file_uploader("Select an image: ", type=["jpg", "jpeg", "png"],key="upload1")
        if input_file is None:
            st.write("<p><b>Please select your Image</b></p>",unsafe_allow_html=True)
        else:
            image = Image.open(input_file)
            st.write(" ")
            st.markdown("""<h4>Original Image</h3>""",unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.image(image)
            with c2:
                fr,hi,wi,arr,ar_s,mo=image_details(image)
                st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                st.write("<b>Image RGB Array colors:</b>",[arr],unsafe_allow_html=True)
                st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
            gray_image = image.convert("L")
            st.write(" ")
            st.write("""<h5>Select any types of preprocessing techniques:</h5>""",unsafe_allow_html=True)
            processing=st.multiselect("Choose any preprocessing techniques:",["Gray Scale","Resizing Image","Bluring Image","Contrast Image","Edge Detection","Sharping Image","Negative Image","Brightness","Text in Image"])
            for i in processing:
                if i=="Gray Scale":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Gray Scale Conversation</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        gray_image = image.convert("L")
                        st.image(gray_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(gray_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Resizing Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Resizing Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        h=st.slider("Choose size", 0, 5000,400)
                        w=st.slider("Choose Width:",0,5000,200)
                        resized_image = gray_image.resize((h,w))
                        st.image(resized_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(resized_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Bluring Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Bluring Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        b=st.slider("Choose size", 0, 100,3)
                        blur_image = gray_image.filter(ImageFilter.GaussianBlur(radius=b)) 
                        st.image(blur_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(blur_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Contrast Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Contrast Image</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        c=st.slider("Choose value:",0,255,10)
                        process_1 = ImageEnhance.Contrast(gray_image)
                        process=process_1.enhance(c)
                        st.image(process)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(process)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Edge Detection":
                    st.write(" ")
                    st.markdown("""<h3>Edge detection</h3>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        g=st.slider("Choose value",0,20,5)
                        gray_edge = gray_image.filter(ImageFilter.FIND_EDGES)
                        edge_bright = ImageEnhance.Brightness(gray_edge)
                        edge_ = edge_bright.enhance(g)
                        st.image(edge_)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(edge_)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Sharping Image":
                    st.write(" ")
                    st.markdown("""<h3>Sharping Image:<h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        s=st.slider("Choose value",0,30,5)
                        sharp_img = ImageEnhance.Sharpness(image)
                        sharp=sharp_img.enhance(s)
                        st.image(sharp)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(sharp)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Brightness":
                    st.write(" ")
                    st.markdown("""<h3>Brightness:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        br=st.slider("Choose value",0,100,10)
                        edge_bright = ImageEnhance.Brightness(gray_image)
                        bright = edge_bright.enhance(br)
                        st.image(bright)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(bright)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Negative Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Negative Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        neg_image = ImageOps.invert(gray_image)
                        st.image(neg_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(neg_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Text in Image":
                    st.write(" ")
                    st.markdown("""<h3>Text in Image</h4>""",unsafe_allow_html=True)
                    reader = easyocr.Reader(['en'])
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result = reader.readtext(opencv_image,detail=0)
                    c1,c2=st.columns(2)
                    with c1:
                        st.image(image)
                    with c2:
                        if len(result)==0:
                            st.write("")
                            st.write("")
                            st.write("""<b>There is no text in the given image...</b>""",unsafe_allow_html=True)
                        else:
                            st.write("""Text from the Image""")
                            st.write(result)
                        
if opt=="Text Preprocessing":
    st.markdown("""<h3>Text Preprocessing</h3""",unsafe_allow_html=True)
    with st.expander("ABOUT"):
        st.write("""<p>Text preprocessing is an essential step in natural language processing (NLP) and machine learning tasks. It involves cleaning and transforming raw text data into a format that can be easily understood and analyzed by algorithms. </p>""",unsafe_allow_html=True)
        st.write("""<p><b>Tokenization: </b>Tokenization is a crucial step in text preprocessing where a text is split into individual units, which are often words or subwords. This process is fundamental for various natural language processing (NLP) tasks</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Stemming: </b>Stemming is a text normalization process that involves reducing words to their base or root form, known as a stem. The goal of stemming is to group words with the same meaning but different inflections or derivations under a common base form. This helps in reducing the dimensionality of the data and capturing the core meaning of words. </p>""",unsafe_allow_html=True)
        st.write("""<p><b>Lemmatization: </b>Lemmatization is another text normalization technique, like stemming, but it tends to be more linguistically accurate. The goal of lemmatization is to reduce words to their base or dictionary form, known as a lemma. Lemmatization considers the context and the meaning of a word to ensure that the resulting base form is a valid word in the language.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Stop Word Removal: </b>Stop word removal is a common text preprocessing step that involves removing commonly used words (stop words) from a text. These words, such as "the," "and," "is," are often very frequent but typically do not carry much meaning in a given context. Removing stop words can help reduce the dimensionality of the data and improve the efficiency of algorithms, as well as focus on more meaningful words for analysis.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Removing Special Characters: </b>Removing special characters is a common text preprocessing step that involves eliminating characters that are not letters, numbers, or whitespace. Special characters can include punctuation, symbols, and other non-alphanumeric characters.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Parts of Speech: </b>Parts of speech (POS) refer to the grammatical categories of words in a language, and they play a crucial role in understanding the structure and meaning of sentences. Common parts of speech include nouns, verbs, adjectives, adverbs, pronouns, prepositions, conjunctions, and interjections.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Word Cloud:</b>A word cloud is a popular and visually appealing way to represent the frequency or importance of words in a body of text. It visually emphasizes words that appear more frequently. Larger or bolder words are often used to indicate higher frequency or importance. </p>""",unsafe_allow_html=True)
        st.write("""<p><b>Sentiment Analysis Score: </b>Sentiment analysis involves determining the sentiment or emotional tone expressed in a piece of text. The sentiment is often categorized as positive, negative, or neutral. Sentiment analysis scores are numerical values assigned to the sentiment of the text. </p>""",unsafe_allow_html=True)
    tab1,tab2=st.tabs(["Text preprocess","Translate"])
    with tab1:
        o=st.radio("Select one option",options=["Browse image","Enter text"])
        if o=="Browse image":
            input_file=st.file_uploader("Select an image: ", type=["jpg", "jpeg", "png"],key="upload2")
            if input_file is None:
                st.write("""<p><b>Please select the image</b></p>""",unsafe_allow_html=True)
            else:
                image = Image.open(input_file)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                reader=easyocr.Reader(['en'])
                result = reader.readtext(opencv_image,detail=0)
                text_input=" ".join(result)
        if o=="Enter text":
            st.write("""<p>You can enter any text here:</p>""",unsafe_allow_html=True)
            text_input=st.text_area("Enter your text:")
        text_opt_select=st.multiselect("Select any preprocessing techniques:",["Word Tokenize","Stop Word Removal","Removing Special Characters","Stemming","Lemmatizer","Parts of Speech","Word Colud","Sentiment Analysis Score","Keyword Extraction"])
        if st.button("Process"):
            sample_tokens = word_tokenize(text_input)
            spl=[re.sub(r'[^a-zA-Z]','',i )  for i in sample_tokens]
            for i in text_opt_select:
                if i=="Word Tokenize":
                    st.markdown("""<h4>Word Tokenize</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Word Tokenize</h5>""",unsafe_allow_html=True)
                        st.write(sample_tokens)
                if i=="Stop Word Removal":
                    st.markdown("""<h4>Stop Word Removal</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Stop Word Removal</h5>""",unsafe_allow_html=True)
                        stop_words=stopwords.words('english')
                        stp_rem=[i for i in spl if i not in stop_words]
                        st.write(stp_rem)
                if i=="Removing Special Characters":
                    st.markdown("""<h4>Removing Special Characters</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Removing Special Characters</h5>""",unsafe_allow_html=True)
                        st.write(spl)
                if i=="Stemming":
                    st.markdown("""<h4>Stemming</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Stemming</h5>""",unsafe_allow_html=True)
                        stemmer = PorterStemmer()
                        stem=[stemmer.stem(i) for i in spl]
                        st.write(stem)
                if i=="Lemmatizer":
                    st.markdown("""<h4>Lemmatizer</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Lemmatizer</h5>""",unsafe_allow_html=True)
                        lemm = WordNetLemmatizer()
                        lem=[lemm.lemmatize(i) for i in spl]
                        st.write(lem)
                if i=="Parts of Speech":
                    st.markdown("""<h4>Parts of Speech</h4>""",unsafe_allow_html=True)
                    c1,c2,c3=st.columns(3)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Parts of Speech</h5>""",unsafe_allow_html=True)
                        tag_mapping = {'CC': 'Coordinating Conjunction','CD': 'Cardinal Digit','DT': 'Determiner','EX': 'Existential There','FW': 'Foreign Word','IN': 'Preposition or Subordinating Conjunction','JJ': 'Adjective','JJR': 'Adjective, Comparative','JJS': 'Adjective, Superlative','LS': 'List Item Marker','MD': 'Modal','NN': 'Noun, Singular or Mass','NNS': 'Noun, Plural','NNP': 'Proper Noun, Singular','NNPS': 'Proper Noun, Plural','PDT': 'Predeterminer','POS': 'Possessive Ending','PRP': 'Personal Pronoun','PRP$': 'Possessive Pronoun','RB': 'Adverb','RBR': 'Adverb, Comparative','RBS': 'Adverb, Superlative','RP': 'Particle','SYM': 'Symbol','TO': 'to','UH': 'Interjection','VB': 'Verb, Base Form','VBD': 'Verb, Past Tense','VBG': 'Verb, Gerund or Present Participle','VBN': 'Verb, Past Participle','VBP': 'Verb, Non-3rd Person Singular Present','VBZ': 'Verb, 3rd Person Singular Present','WDT': 'Wh-determiner','WP': 'Wh-pronoun','WP$': 'Possessive Wh-pronoun','WRB': 'Wh-adverb'}
                        pos_tagged_words = nltk.pos_tag(spl)
                        pdf=pd.DataFrame(pos_tagged_words,columns=["Text","Parts of Speech"])
                        st.dataframe(pdf)
                    with c3:
                        po=tag_mapping.values()
                        pk=tag_mapping.keys()
                        p=pd.DataFrame(po,pk)
                        st.dataframe(p)
                if i=="Word Colud":
                    st.markdown("""<h4>Word Colud</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Word Colud</h5>""",unsafe_allow_html=True)
                        w_c = WordCloud(width=1000,height=500,background_color="orange").generate(text_input)
                        plt.figure(figsize=(14,6))
                        plt.imshow(w_c)
                        plt.axis('off')
                        st.pyplot()
                if i=="Sentiment Analysis Score":
                    st.markdown("""<h4>Sentiment Analysis Score</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Sentiment Analysis Score</h5>""",unsafe_allow_html=True)
                        sent_ment = SentimentIntensityAnalyzer()
                        sent_score = sent_ment.polarity_scores(text_input)
                        score={}
                        for i in sent_score:
                            score[i]=sent_score[i]*100
                        st.write(score)
                        sc=score.values()
                        fig = px.histogram(sc,color=score.keys())
                        fig.update_layout(bargap=0.1)
                        st.plotly_chart(fig)
                if i=="Keyword Extraction":
                    st.markdown("""<h4>Keyword Extraction</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        st.write("""<h5>Original Text</h5>""",unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Keyword Extraction</h5>""",unsafe_allow_html=True)
                        vect = CountVectorizer(stop_words="english")
                        ext_val = vect.fit_transform(spl)
                        nf=vect.get_feature_names_out()
                        st.write(nf)
    with tab2:
        st.markdown("""<h3>Translation</h3>""",unsafe_allow_html=True)
        lan={'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian', 'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu', 'fil': 'Filipino', 'he': 'Hebrew'}
        src_lang=lan.values()
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>From:</h4>""",unsafe_allow_html=True)
            source=st.selectbox("Select the source language",src_lang)
            for i,j in lan.items():
                if source==j:
                    source_lang=i
                    break
        with c2:
            st.markdown("""<h4>To:</h4>""",unsafe_allow_html=True)
            des=st.selectbox("Select the destination language",src_lang)
            for i,j in lan.items():
                if des==j:
                    des_lang=i
                    break
        text_translate=st.text_input("Enter your text to translate:")
        if st.button("Translate",key="tras"):
            tran=googletrans.Translator()
            test_translated = tran.translate(text_translate,src=source_lang,dest=des_lang)
            st.write(f"""<h5>Original Sentence:</h5> {test_translated.origin}""",unsafe_allow_html=True)
            st.write("")
            st.write(f"""<h5>Translated Sentence:</h5> {test_translated.text}""",unsafe_allow_html=True)
            st.write("")
            st.write(f"""<h5>Sentence Pronunciation:</h5> {test_translated.pronunciation}""",unsafe_allow_html=True)
                
if opt=="Recommendation":
    st.markdown("""<h3>Course Recommendation</h3>""",unsafe_allow_html=True)
    tab1,tab2=st.tabs(["ABOUT","Recommend"])
    with tab1:
        st.markdown("""<h3>History</h4>""",unsafe_allow_html=True)
        st.write("""<p>Coursera was founded in 2012 by Stanford University computer science professors Andrew Ng and Daphne Koller. Ng and Koller started offering their Stanford courses online in fall 2011, and soon after left Stanford to launch Coursera. Princeton, Stanford, the University of Michigan, and the University of Pennsylvania were the first universities to offer content on the platform. In 2014 Coursera received both the Webby Winner (Websites and Mobile Sites Education 2014) and the People's Voice Winner (Websites and Mobile Sites Education) awards.</p>""",unsafe_allow_html=True)
        st.write("""<p>In March 2021, Coursera filed for an IPO. The nine-year-old company brought in roughly 293 million dollar in revenue for the fiscal year ended December 31 — a 59% growth rate from 2019, according to the filing. Net losses widened by roughly 20 million dolloar yearly, reaching 66.8 million dollar in 2020. Coursera spent 107 million dollar on marketing in 2020.</p>""",unsafe_allow_html=True)
        st.markdown("""<h3>Course</h3>""",unsafe_allow_html=True)
        st.write("""<p>Coursera courses last approximately four to twelve weeks, with one to two hours of video lectures a week. These courses provide quizzes, weekly exercises, peer-graded and reviewed assignments, an optional Honors assignment, and sometimes a final project or exam to complete the course. Courses are also provided on-demand, in which case users can take their time in completing the course with all of the material available at once. As of May 2015, Coursera offered 104 on-demand courses. They also provide guided projects which are short 2-3 hour projects that can be done at home.</p>""",unsafe_allow_html=True)
        st.markdown("""<h3>Degree and Certificate</h3>""",unsafe_allow_html=True)
        st.write("""<p>As of 2017, Coursera offers complete master's degrees. They first started with a Master's in Innovation and Entrepreneurship (MSIE) from HEC Paris and a Master's of Accounting (iMSA) from the University of Illinois but have moved on to offer a Master of Computer Science in Data Science and Master of Business Administration (iMBA), both from University of Illinois. Also as part of their MBA programs, there are some courses which are offered separately, which are included in the curriculum of specific MBAs when enrolling in classes such as their digital marketing courses.</p>""",unsafe_allow_html=True)
        st.write("""<p>Google, IBM, Meta and other well-known companies, launched various courses for professional certificates, allowing students to fill the workforce in various sectors, such as data analytics, IT support, digital marketing, UX design, project management, and Data science. According to Google, their courses are equivalent to 4 year degrees. They also offered 100,000 scholarships. Google and its 20+ partners will accept those certificates as 4-year degree equivalent.</p>""",unsafe_allow_html=True)
    with tab2:
        st.markdown("""<h4>Recommendation:</h4>""",unsafe_allow_html=True)
        st.write("""<p>Coursera offers a wide range of courses across various subjects. The availability of courses may change, and new courses may be added. To find the most up-to-date and specific information, I recommend visiting the official Coursera website. </p>""",unsafe_allow_html=True)
        course=pd.read_csv("Coursera.csv")
        course.drop_duplicates()
        le=LabelEncoder()
        course["course_name_label"]=le.fit_transform(course["Course Name"])
        name=course["Course Name"].values
        name=list(set(name))
        course_name=st.multiselect("Select any course:",name)
        if st.button("Recommend"):
            recommd(course,course_name)
