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



################################################ functions ################################################


#clean string
def cleanStr(s):
    s = s.strip().lower()
    s = re.sub('[^0-9a-zA-Z]+', '_', s)
    return s


###########################################################################################################



# get all excel files
allFiles = glob.glob(".//*.xlsx")

# read each file
for i in allFiles:
    file_name = i.replace(".\\","").replace(".xlsx","")
    file = pd.ExcelFile(i)
    table_name = cleanStr(file_name)

    #system info
    sys_info = "/*** This is auto-generated script by FuXiaotong, for creating table in idl and cdl. ***/ \n"


    # initial script
    create_table_idl_sql = "-- prd idl \n"
    create_table_idl_sql += "create table prd_inbound.{@table}_idl ( \n"
    create_table_idl_sql += "{@columns} , \n"
    create_table_idl_sql += "    batch_number string \n"
    create_table_idl_sql += "); \n"

    create_table_cdl_sql = "-- prd cdl \n"
    create_table_cdl_sql += "create table prd_updated.{@table}_udl ( \n"
    create_table_cdl_sql += "{@columns} \n"
    create_table_cdl_sql += ") partitioned by (batch_number string) \n"
    create_table_cdl_sql += "STORED AS PARQUET; \n"

    verify_sql = "-- verify \n"
    verify_sql += "select * from prd_inbound.{@table}_idl \n"
    verify_sql += "union all \n"
    verify_sql += "select * from prd_updated.{@table}_udl \n"

    final_create_table_sql = sys_info + "\n" + create_table_idl_sql + "\n" + create_table_cdl_sql + "\n" + verify_sql

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
    path = ".\\create_table_scripts\\"
    if not os.path.exists(path):
        os.makedirs(path)
    f_out = open(path + table_name + ".sql", 'w')
    f_out.write(final_create_table_sql)  # python will convert \n to os.linesep
    f_out.close()

    #archive
    path = ".\\archive_table_file\\"
    if not os.path.exists(path):
        os.makedirs(path)
    os.rename(i, path+i.replace(".\\",""))

