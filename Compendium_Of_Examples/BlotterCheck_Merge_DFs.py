import numpy as np
import pandas as pd
import time
from openpyxl import load_workbook



start = time.time()


xls =  pd.ExcelFile("L:\Excel Sheets\BlotterCheck4.xls")
tsy_bbg = xls.parse("BBG_Form")
tsy_prom = xls.parse("Prom Form_TSY")
tsy_bbg = tsy_bbg.dropna(axis=0,how='all')
tsy_prom = tsy_prom.dropna(axis=0,how='all')


check = pd.merge(tsy_bbg,tsy_prom,how="left",left_index=True,right_index=True)


for o in ["CUSIP","SIDE","PRICE","AMOUNT","BROKER","SETTLE_DATE"]:
    for i in range(0,check.shape[0]):
        if check.ix[i,o+"_x"]==check.ix[i,o+"_y"]:
            check.ix[i,o+'_check']="ok"
        else:
            check.ix[i,o+'_check']="WRONG!"



check1 = pd.merge(tsy_prom,tsy_bbg,how="left",left_index=True,right_index=True)
for o in ["CUSIP","SIDE","PRICE","AMOUNT","BROKER","SETTLE_DATE"]:
    for i in range(0,check1.shape[0]):
        if check1.ix[i,o+"_x"]==check1.ix[i,o+"_y"]:
            check1.ix[i,o+'_check']="ok"
        else:
            check1.ix[i,o+'_check']="WRONG!"






#SAME THING FOR FUTURES-------------------------------------

fut_bbg = xls.parse("EMSX_Print")
fut_prom = xls.parse("Prom Form Fut Print")
fut_bbg = fut_bbg.dropna(axis=0,how='all')
fut_prom = fut_prom.dropna(axis=0,how='all')

check_fut = pd.merge(fut_bbg,fut_prom,how="left",left_index=True,right_index=True)
for o in ["CUSIP","SIDE","PRICE","AMOUNT","BROKER","SETTLE_DATE"]:
    for i in range(0,check_fut.shape[0]):
        if check_fut.ix[i,o+"_x"]==check_fut.ix[i,o+"_y"]:
            check_fut.ix[i,o+'_check']="ok"
        else:
            check_fut.ix[i,o+'_check']="WRONG!"


check1_fut = pd.merge(fut_prom,fut_bbg,how="left",left_index=True,right_index=True)
for o in ["CUSIP","SIDE","PRICE","AMOUNT","BROKER","SETTLE_DATE"]:
    for i in range(0,check1_fut.shape[0]):
        if check1_fut.ix[i,o+"_x"]==check1_fut.ix[i,o+"_y"]:
            check1_fut.ix[i,o+'_check']="ok"
        else:
            check1_fut.ix[i,o+'_check']="WRONG!"




writer = pd.ExcelWriter('L:\Excel Sheets\BlotterCheck_OUTPUT.xlsx', engine='xlsxwriter')
check.to_excel(writer, sheet_name='Tsy_Left')
check1.to_excel(writer, sheet_name='Tst_Right')
check_fut.to_excel(writer, sheet_name='Fut_Left')
check1_fut.to_excel(writer, sheet_name='Fut_Right')
writer.save()
print('time for all of data')
end4 = time.time()
print(end4 - start)
