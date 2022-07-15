#############################################
#GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

#Soru1 : persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv("datasets/persona.csv")
df.head()
df.info()
df.shape
df.columns
df.describe().T

#Soru2 : Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].unique()
df["SOURCE"].nunique()
df["SOURCE"].value_counts()


#Soru3: Kaç unique PRICE vardır?

df["PRICE"].unique()

#Soru4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

#Soru5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

#Soru6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY")["PRICE"].sum()

#Soru7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

#Soru8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

#Soru9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE")["PRICE"].mean()

#Soru10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()


###########################################################################
#GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında aortalama kazançlar nedir?
###########################################################################

df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean()


############################################
#GÖREV 3: Çıktıyı PRICE’a göre sıralayınız.
############################################

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"})
agg_df = agg_df.sort_values("PRICE",ascending=False)


################################################################
#GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
################################################################

agg_df=agg_df.reset_index()


##############################################################################
#GÖREV 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70
###############################################################################

age_categories = ["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"], bins = [0, 18, 23, 30, 40, 70], labels=age_categories)

agg_df.head()


##############################################################################
#GÖREV 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
# değişkenini oluşturmanız gerekmektedir.
###############################################################################

agg_df["customer_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].apply("_".join, axis=1)
agg_df["customer_level_based"] = agg_df["customer_level_based"].apply(lambda x: x.upper())
agg_df["customer_level_based"]


persona = agg_df.groupby("customer_level_based").agg({"PRICE":"mean"})
persona.reset_index(inplace=True)
persona.head()


##############################################################################
#GÖREV 7: Yeni müşterileri (personaları) segmentlere ayırınız.
# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre groupby yapıp price mean, max, sum’larını alınız).
###############################################################################

persona["SEGMENT"] = pd.qcut(persona["PRICE"],4,labels=["D","C","B","A"])
print(persona.groupby('SEGMENT').agg({'PRICE': ['mean', 'max','sum']}))
persona.head()

##############################################################################
#GÖREV 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
###############################################################################

new_user = "TUR_ANDROID_FEMALE_31_40"
persona[persona["customer_level_based"] == new_user]

new_user = "FRA_IOS_FEMALE_31_40"
persona[persona["customer_level_based"] == new_user]





