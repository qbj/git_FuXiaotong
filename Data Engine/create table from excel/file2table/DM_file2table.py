"""
date: 2017-09-13

purpose: to convert file to table definition script for LUDP idl, cdl/udl

usage:
    1. put 1 input csv file into the same folder with file2table.py
    2. table columns are defined in the csv
    3. the columns are defined as string
    4. the extra column 'batch_number' will be attached
    5. only support


"""

import pandas as pd
import glob
import re
import os
import datetime



################################################ functions ################################################


#clean string
def cleanStr(s):
    s = s.strip().lower()
    s = re.sub('[^0-9a-zA-Z]+', '_', s)
    return s


###########################################################################################################



allFolders = os.listdir(".\\inbound\\")

# read each folders
for folder in allFolders:
    schema_name = folder
    allFiles = glob.glob(".\\inbound\\" + folder + "\\*.xlsx")

    # read each file
    for i in allFiles:
        file_name = i.replace(".\\inbound\\","").replace(".xlsx","").replace( schema_name + "\\","")
        file = pd.ExcelFile(i)
        table_name = cleanStr(file_name)

        #system info
        sys_info = "/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ \n"

        # initial script
        create_table_dm_sql = "\n"
        create_table_dm_sql += "create table " + schema_name + ".{@table} ( \n"
        create_table_dm_sql += "{@columns} , \n"
        create_table_dm_sql += "    batch_number string \n"
        create_table_dm_sql += "); \n"

        # final script
        final_create_table_sql = "\n" + create_table_dm_sql + "\n"

        # read each sheet
        for j in file.sheet_names:
            df_file = file.parse(j)
            if not df_file.empty:

                # get columns
                columns_sql = ""
                for k in df_file.columns:
                    column = cleanStr(k)
                    columns_sql += "    " + column + " string" + " ,\n"

                # remove last ','
                columns_sql = columns_sql[:-2]

        # final sql
        final_create_table_sql = final_create_table_sql.replace("{@table}",table_name)
        final_create_table_sql = final_create_table_sql.replace("{@columns}",columns_sql)

        # write file
        path = ".\\outbound\\" + schema_name + "\\"
        if not os.path.exists(path):
            os.makedirs(path)
        f_out = open(path + table_name + ".sql", 'w')
        f_out.write(final_create_table_sql)  # python will convert \n to os.linesep
        f_out.close()

        #archive
        path = ".\\archive\\" + schema_name + datetime.datetime.now().strftime("%y%m%d%H%M") + "\\"
        if not os.path.exists(path):
            os.makedirs(path)
        os.rename(i, path + file_name +".xlsx")

