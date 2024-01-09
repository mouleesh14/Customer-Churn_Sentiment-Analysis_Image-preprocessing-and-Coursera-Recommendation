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
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(layout='wide',page_title="Final_Project")

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
def color_patt(sline_in,i_in):
    wch_colour_box = (0, 204, 102)
    fontsize = 25
    sline = sline_in #"text"
    lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
    i = i_in 
    htmlstr = f"""<p style='background-color: rgba({wch_colour_box[1]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[1]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 10px; 
                                padding-left: 12px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height: {fontsize * 1.5}px;'> <!-- Adjusted line height -->
                                <span style='color: black; font-size: {fontsize+5}px;'>{i}</span><br>
                                <span style='color: black; font-size: {fontsize}px; margin-top: 0;'>{sline}</span></p>"""

    st.markdown(lnk + htmlstr, unsafe_allow_html=True)

    
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
        "geoNetwork_region","bounce_rate"]
    cls_data1.drop(col_to_rem,axis=1,inplace=True)
    zero1=[]
    for z in cls_data1.columns:
        value=((cls_data1[z]==0).mean()*100).round(2)
        zero1.append(value)
    zero_df_pre=pd.DataFrame({"Column_name":cls_data1.columns,"Zero_Percentage":zero1}).sort_values("Zero_Percentage",ascending=False)
    return zero_df,zero_df_pre,cls_data1   
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

    
def getting_data():
    cls_data=pd.read_csv("classification_data.csv")
    sh=cls_data.shape
    is_null=(cls_data.isnull().mean()*100).round(2)
    is_null_df=pd.DataFrame({"Column_Name":is_null.index,"Null_Percentage":is_null.values}).sort_values("Null_Percentage",ascending=False)
    cls_data.drop_duplicates(inplace=True)
    sh1=cls_data.shape
    des=cls_data.describe()
    zero_df,zerodf_pre,cls_data1=zero_preprocessing(cls_data)
    return sh,sh1,des,is_null_df,zero_df,zerodf_pre,cls_data1

with st.sidebar:
    opt = option_menu("Final",["Prediction","Image Preprocessing","Text Preprocessing","Recommendation"],menu_icon="cast",styles={"container": {"padding":"4!important"},"nav-link": {"text-align":"left"},"nav-link-selected": {"background-color": "#C2452D"}})
    
if opt=="Prediction":
    st.markdown("<h2><FONT COLOR='#000000'>Predicton</h3>",unsafe_allow_html=True)
    shap,shape,des,isnull,zerodf,zerodf_pre,cls_data= getting_data()
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
        if li[0]==0:
            st.write("""<h4><FONT COLOR='red'>User Not Converted</h4>""",unsafe_allow_html=True)
        else:
            st.write("""<h4><FONT COLOR='green'>User Converted</b></h4>""",unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.markdown("""<h3>Model Result</h3>""",unsafe_allow_html=True)
    st.markdown("""<h4>Metrics Score</h4>""",unsafe_allow_html=True)
    with open("log_reg.pkl","rb") as lg:
        lg_model=pickle.load(lg)
    tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,lg_model)
    score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
    st.dataframe(score_df)
    st.markdown("""<h4>Metrics Score</h4>""",unsafe_allow_html=True)
    with open("svm_model.pkl","rb") as sv:
        sv_model=pickle.load(sv)
    tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,sv_model)
    score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
    st.dataframe(score_df)
    st.markdown("""<h4>Metrics Score</h3>""",unsafe_allow_html=True)
    with open("knn.pkl","rb") as lg:
        knn_model=pickle.load(lg)
    tr_a,tr_p,tr_r,tr_f,te_a,te_p,te_r,te_f=model_predict(over_data,over_tar,knn_model)
    score_df=pd.DataFrame({"Accuracy":[tr_a,te_a],"Precision":[tr_p,te_p],"Recall":[tr_r,te_r],"F1 Score":[tr_f,te_f]},index=["Training Score","Testing Score"])
    st.dataframe(score_df)
        
if opt=="Image Preprocessing":
    st.markdown("""<h3><FONT COLOR:'#000000'>Image Preprocessing</h3>""",unsafe_allow_html=True)
    
if opt=="Text Preprocessing":
    st.markdown("""<h3>Text Preprocessing</h3""",unsafe_allow_html=True)
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
if opt=="Recommendation":
    stmarkdown("""<h3>Course Recommendation</h3>""",unsafe_allow_html=True)

