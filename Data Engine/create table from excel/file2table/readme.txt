1. put excel files in directory "inbound\schema_name\"
2. each file stands for a table, file name is the table name. e.g. 'table1.xlsx' will create table 'table1'
3. sql script will be generated into '\outbound'
4. the input files will be archived into 'archive'
5. merge.sh is to unified all scripts into a single file.