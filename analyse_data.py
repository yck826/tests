# -*- coding: utf-8 -*-
from sql_update import MSSQL
import sklearn
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
import datetime

def KNeigh_predict(predict_X):
    clf = joblib.load("train3_model.m")
    clf1 = joblib.load("train5_model.m")
    #
    # filename =  'train_data5.csv'
    # csv_Data = pd.read_csv(filename, encoding='gbk')
    # csv_Data.dropna(inplace=True)
    # X = csv_Data[['kdj_k','kdj_d','mfi','RSI_6','BollUp','bollmiddle','BollDown','lcap']]
    # X = [x for x in X.values]
    # #    print(X)
    # y = [round(x,3) for x in csv_Data['p_chg'].values]
    # neigh = KNeighborsRegressor(n_neighbors=8)
    # neigh.fit(X, y)
    # joblib.dump(neigh, "train5_model.m")
    # return neigh.predict(predict_X)
    return clf.predict(predict_X),clf1.predict(predict_X)

def get_result():
    st_date = str(datetime.datetime.now().date())
    sql = '''SELECT Stockcode,SMTime,p_chg,kdj_k,kdj_d,kdj_j,CCI5,mfi,RSI_6,BollUp,bollmiddle,BollDown,bias10,lcap FROM daydata_extend1 where SMTime ='%s' '''%st_date
    # sql ='''SELECT Stockcode,SMTime,PE,p_chg,DIFF,mfi,DEA,MACD,kdj_k,kdj_d,kdj_j,CCI5,atr6,RSI_6 FROM daydata_extend1 where SMTime = '2019-02-12';'''
    # select * from t_book, t_bookType where t_book.bookTypeId = t_bookType.id
    # df = pd.read_sql_query(sql, engine)
    ms = MSSQL(host="127.0.0.1", user="root", pwd="123456", db="StockDB_Test")
    SID, index = ms.Selsql(sql)
    # print(SID)
    df_index = []
    for i in range(len(index)):
        df_index.append(index[i][0])
    # print(df_index)
    SID = pd.DataFrame(list(SID), columns=df_index)
    predict = SID[
        ['Stockcode', 'SMTime', 'kdj_k', 'kdj_d', 'mfi', 'RSI_6', 'BollUp', 'bollmiddle', 'BollDown', 'lcap']].copy()
    predict = predict.dropna()
    predict_X = predict[['kdj_k', 'kdj_d', 'mfi', 'RSI_6', 'BollUp', 'bollmiddle', 'BollDown', 'lcap']].copy()
    # predict_X = predict_X.dropna()
    predict_X = [x for x in predict_X.values]
    f, f1 = KNeigh_predict(predict_X)
    predict1 = predict.copy()
    predict['predict'] = f
    predict['Stockcode'] = predict['Stockcode'].apply(lambda x: "'" + str(x))
    predict1['predict'] = f1
    predict1['Stockcode'] = predict['Stockcode'].apply(lambda x: "'" + str(x))

    predict.to_csv('D:\python_project\\analyse_shares_data\\'+str(datetime.datetime.now().date()) + '.csv', index=False, sep=',', encoding='gbk')
    predict1.to_csv('D:\python_project\\analyse_shares_data\\'+str(datetime.datetime.now().date()) + '-1.csv', index=False, sep=',', encoding='gbk')
    return True

# if __name__ == "__main__":
#     print(get_result())
#     #    print(y)
# #    h = KNeigh_predict(X,y)
#    st_date = str(datetime.datetime.now().date())
#    sql = '''SELECT Stockcode,SMTime,p_chg,kdj_k,kdj_d,kdj_j,CCI5,mfi,RSI_6,BollUp,bollmiddle,BollDown,bias10,lcap FROM daydata_extend1 where SMTime = '2019-2-21';'''
#    # sql ='''SELECT Stockcode,SMTime,PE,p_chg,DIFF,mfi,DEA,MACD,kdj_k,kdj_d,kdj_j,CCI5,atr6,RSI_6 FROM daydata_extend1 where SMTime = '2019-02-12';'''
#    # select * from t_book, t_bookType where t_book.bookTypeId = t_bookType.id
#    # df = pd.read_sql_query(sql, engine)
#    ms = MSSQL(host="127.0.0.1", user="root", pwd="123456", db="StockDB_Test")
#    SID,index = ms.Selsql(sql)
#    # print(SID)
#    df_index = []
#    for i in range(len(index)):
#        df_index.append(index[i][0])
#     # print(df_index)
#    SID = pd.DataFrame(list(SID),columns=df_index)
#    predict = SID[['Stockcode','SMTime','kdj_k','kdj_d','mfi','RSI_6','BollUp','bollmiddle','BollDown','lcap']].copy()
#    predict = predict.dropna()
#    predict_X = predict[['kdj_k','kdj_d','mfi','RSI_6','BollUp','bollmiddle','BollDown','lcap']].copy()
#    # predict_X = predict_X.dropna()
#    predict_X = [x for x in predict_X.values]
#    f,f1 = KNeigh_predict(predict_X)
#    predict1 = predict.copy()
#    predict['predict'] = f
#    predict['Stockcode'] = predict['Stockcode'].apply(lambda x:"'"+str(x))
#    predict1['predict'] = f1
#    predict1['Stockcode'] = predict['Stockcode'].apply(lambda x: "'" + str(x))
#
#    predict.to_csv(str(datetime.datetime.now().date())+'.csv', index=False, sep=',', encoding='gbk')
#    predict1.to_csv(str(datetime.datetime.now().date())+'-1.csv', index=False, sep=',', encoding='gbk')