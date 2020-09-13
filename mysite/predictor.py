# I have created
from django.http import HttpResponse
import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import BaggingClassifier

def getKmers(sequence, size=6):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1



def result_class(request):
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
        doc2 = request.FILES['document2']
    human = pd.read_table(doc1)
    chimp = pd.read_table(doc2)
    human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
    human = human.drop('sequence', axis=1)
    chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
    chimp = chimp.drop('sequence', axis=1)
    human_texts = list(human['words'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])
    y_h = human.iloc[:, 0].values
    chimp_texts = list(chimp['words'])
    for item in range(len(chimp_texts)):
        chimp_texts[item] = ' '.join(chimp_texts[item])
    y_c = chimp.iloc[:, 0].values

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)
    X_chimp = cv.transform(chimp_texts)
    # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
    data1_shape=X.shape
    data2_shape=X_chimp.shape

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_h,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    conf_m1=pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))



    accuracy1, precision1, recall1, f11 = get_metrics(y_test, y_pred)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    y_pred_chimp = classifier.predict(X_chimp)
    # performance on chimp genes
    #print("Confusion matrix\n")
    conf_m2=pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted'))
    accuracy2, precision2, recall2, f12 = get_metrics(y_c, y_pred_chimp)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

    #print("see")

    # if f.mode=='r':
    #     contents=f.read()
    #     print(contents)
    # doc1=pd.read_table(doc1)
    params={'shape1':data1_shape,'shape2':data2_shape,'confu_matrix1':conf_m1,'confu_matrix2':conf_m2,'accuracy1':accuracy1,'precision1':precision1,'recall1':recall1,'f11':f11,'accuracy2':accuracy2,'precision2':precision2,'recall2':recall2,'f12':f12}
    return render(request, 'mysite/result_pred.html',params)

def result_newgene(request):
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
        doc2 = request.FILES['document2']
    human = pd.read_table(doc1)
    chimp = pd.read_table(doc2)
    human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
    human = human.drop('sequence', axis=1)
    chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
    chimp = chimp.drop('sequence', axis=1)
    human_texts = list(human['words'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])
    y_h = human.iloc[:, 0].values
    chimp_texts = list(chimp['words'])
    for item in range(len(chimp_texts)):
        chimp_texts[item] = ' '.join(chimp_texts[item])
    y_c = chimp.iloc[:, 0].values

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)
    X_chimp = cv.transform(chimp_texts)
    # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
    data1_shape=X.shape
    data2_shape=X_chimp.shape

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_h,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    conf_m1=pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))



    accuracy1, precision1, recall1, f11 = get_metrics(y_test, y_pred)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    y_pred_chimp = classifier.predict(X_chimp)
    # performance on chimp genes
    #print("Confusion matrix\n")
    # conf_m2=pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted'))
    # accuracy2, precision2, recall2, f12 = get_metrics(y_c, y_pred_chimp)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

    #print("see")
    res_gene=classifier.predict(X_chimp[0])
    # if f.mode=='r':
    #     contents=f.read()
    #     print(contents)
    # doc1=pd.read_table(doc1)
    params={'shape1':data1_shape,'shape2':data2_shape,'confu_matrix1':conf_m1,'accuracy1':accuracy1,'precision1':precision1,'recall1':recall1,'f11':f11,'ans':res_gene}
    return render(request, 'mysite/result_newgene_out.html',params)

def result_clin_var (request):

    #file_ = open(os.path.join(settings.BASE_DIR, 'train.csv'))

    df_numeric = pd.read_csv(os.path.join(settings.BASE_DIR, 'train.csv'), dtype={0: object, 38: str, 40: object})
    df_numeric = df_numeric[['AF_ESP', 'POS', 'AF_EXAC', 'AF_TGP', 'CLASS']]
    # splitting data into training and test sets
    df_numeric_predictors = df_numeric.drop(["CLASS"], axis=1)
    df_numeric_outcome = df_numeric["CLASS"]
    X_train, X_test, y_train, y_test = train_test_split(df_numeric_predictors, df_numeric_outcome, test_size=0.33,
                                                        random_state=42)
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
    df2_numeric = pd.read_csv(doc1, dtype={0: object, 38: str, 40: object})
    df2_numeric = df2_numeric[['AF_ESP', 'POS', 'AF_EXAC', 'AF_TGP', 'CLASS']]
    df2_testx = df2_numeric.drop(["CLASS"], axis=1)
    df2_testy = df2_numeric["CLASS"]
    models = [BaggingClassifier()]
    # Gather metrics here
    accuracy_by_model = {}

    # Train then evaluate each model
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        y2_pred = model.predict(df2_testx)
        model_name = model.__class__.__name__
        accuracy_by_model[model_name] = score
    acc_df = pd.DataFrame(list(accuracy_by_model.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy',
                                                                                                      ascending=False).reset_index(
        drop=True)
    acc_df.index = acc_df.index + 1
    conf_mat = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))
    str1=""
    count=2;
    for i in y2_pred:
        if(i==0):
            str1=str1+str(count)+" Your predicition variant type : Non-conflicting"+'\n'
        else:
            str1=str1+str(count)+" Your predicition variant type : Conflicting"+'\n'
        count=count+1
    params={'res':str1,'table':acc_df,'conf':conf_mat}

    return render(request,'mysite/result_clin_var.html',params)


def download (request):
    # df_numeric = pd.read_csv(os.path.join(settings.BASE_DIR, 'mysite\static\mysite\data\prain.csv'), dtype={0: object, 38: str, 40: object})
    # print(df_numeric.columns)
    # df_numeric.to_csv(os.path.join(settings.BASE_DIR, 'mysite\static\mysite\data\crain.csv'))
    # os.remove(os.path.join(settings.BASE_DIR, 'mysite\static\mysite\data\prain.csv'))
    if(request.method=='POST'):
        doc1 = request.FILES['document1']

    # cv_columns = {}
    #
    # with gzip.open(doc1, 'rt') as f:
    #     for metaline in f:
    #         if metaline.startswith('##INFO'):
    #             colname = re.search('ID=(\w+),',
    #                                 metaline.strip('#\n'))
    #             coldesc = re.search('.*Description=(.*)>',
    #                                 metaline.strip('#\n'))
    #             cv_columns[colname.group(1)] = coldesc.group(1).strip('"')

    cv_df = pd.read_csv(doc1, sep='\t', comment='#', header=None, usecols=[0, 1, 2, 3, 4, 7], dtype={0: object})

    def list_to_dict(l):
        """Convert list to dict."""
        return {k: v for k, v in (x.split('=') for x in l)}
    # convert dictionaries to columns
    cv_df = pd.concat([cv_df.drop([7], axis=1),
                       cv_df[7].str.split(';')
                      .apply(list_to_dict)
                      .apply(pd.Series)], axis=1
                      )
    #     # rename columns
    cv_df.rename(columns={0: 'CHROM',
                          1: 'POS',
                          2: 'ID',
                          3: 'REF',
                          4: 'ALT'},
                 inplace=True)
    cv_df['CLASS'] = 0
    cv_df.set_value(cv_df.CLNSIGCONF.notnull(), 'CLASS', 1)
    cv_df[['AF_ESP', 'AF_EXAC', 'AF_TGP']] = \
    cv_df[['AF_ESP', 'AF_EXAC', 'AF_TGP']].fillna(0)
    # select variants that have beeen submitted by multiple organizations.
    cv_df = \
        cv_df.loc[cv_df['CLNREVSTAT']
            .isin(['criteria_provided,_multiple_submitters,_no_conflicts',
                   'criteria_provided,_conflicting_interpretations'])]
    # Reduce the size of the dataset below
    cv_df.drop(columns=['ALLELEID', 'RS', 'DBVARID'], inplace=True)
    # drop columns that would reveal class
    cv_df.drop(columns=['CLNSIG', 'CLNSIGCONF', 'CLNREVSTAT'], inplace=True)
    # drop this redundant columns
    cv_df.drop(columns=['CLNVCSO'], inplace=True)
    cv_df.drop(columns=['ID']).to_csv(os.path.join(settings.BASE_DIR, 'mysite\static\mysite\data\crain.csv'),
                                      index=False)
    # os.remove(os.path.join(settings.BASE_DIR, 'mysite\static\mysite\data\clinvar_20190603.vcf.gz'))
    return render(request,'mysite/download.html')

def custom_result (request):
    if(request.method=='POST'):
        doc2 = request.FILES['document2']
    df_numeric = pd.read_csv(doc2, dtype={0: object, 38: str, 40: object})
    df_numeric = df_numeric[['AF_ESP', 'POS', 'AF_EXAC', 'AF_TGP', 'CLASS']]
    # splitting data into training and test sets
    df_numeric_predictors = df_numeric.drop(["CLASS"], axis=1)
    df_numeric_outcome = df_numeric["CLASS"]
    X_train, X_test, y_train, y_test = train_test_split(df_numeric_predictors, df_numeric_outcome, test_size=0.33,
                                                        random_state=42)
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
    df2_numeric = pd.read_csv(doc1, dtype={0: object, 38: str, 40: object})
    df2_numeric = df2_numeric[['AF_ESP', 'POS', 'AF_EXAC', 'AF_TGP', 'CLASS']]
    df2_testx = df2_numeric.drop(["CLASS"], axis=1)
    df2_testy = df2_numeric["CLASS"]
    models = [BaggingClassifier()]
    # Gather metrics here
    accuracy_by_model = {}

    # Train then evaluate each model
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        y2_pred = model.predict(df2_testx)
        model_name = model.__class__.__name__
        accuracy_by_model[model_name] = score
    acc_df = pd.DataFrame(list(accuracy_by_model.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy',
                                                                                                      ascending=False).reset_index(
        drop=True)
    acc_df.index = acc_df.index + 1
    conf_mat = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))
    str1=""
    count=2;
    for i in y2_pred:
        if(i==0):
            str1=str1+str(count)+" Your predicition variant type : Non-conflicting"+'\n'
        else:
            str1=str1+str(count)+" Your predicition variant type : Conflicting"+'\n'
        count=count+1
    params={'res':str1,'table':acc_df,'conf':conf_mat}

    return render(request,'mysite/custom_result.html',params)






