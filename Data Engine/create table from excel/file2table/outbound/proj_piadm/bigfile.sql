/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_finance_e_o_detail ( 
    material_number string ,
    material_description string ,
    valuation_class string ,
    description_of_valuation_class string ,
    profit_center string ,
    product_family string ,
    quantity_of_stock string ,
    unit_of_stock string ,
    value_of_stock string ,
    aug_pup string ,
    total_value_by_pup string ,
    life_cycle string ,
    aging_qty string ,
    e_o string ,
    fin_q1 string ,
    fin_q2 string ,
    fin_q3 string ,
    fin_q4 string  , 
    batch_number string 
); 

/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_finance_e_o_sumary ( 
    q string ,
    yyyy string ,
    reg_e_o string ,
    broker_rev string ,
    scrap string ,
    true_up string ,
    free_buffer string  , 
    batch_number string 
); 

/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_mtm_model ( 
    mtm string ,
    model string ,
    partsnumber string ,
    description string ,
    qty string ,
    p string ,
    attachrate string ,
    cc string ,
    commoditycode string ,
    abc string  , 
    batch_number string 
); 

/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_planning_dashboard ( 
    order_status string ,
    product_line string ,
    pn string ,
    description string ,
    model string ,
    commodity string ,
    order_qty_ string ,
    plant string ,
    demand_type string ,
    updated_commit_eta_616e_ string  , 
    batch_number string 
); 

/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_pn_list ( 
    partno string ,
    model string ,
    pn_country string ,
    partfamily string ,
    description_chinese_ string ,
    description_english_ string ,
    cc string ,
    cc_desc string ,
    abc string ,
    ssdate string ,
    lifecycle string ,
    eol_notice_time string ,
    bac_code string ,
    usagerate string ,
    ltb_done string ,
    mpq string  , 
    batch_number string 
); 

/*** This is auto-generated script by FuXiaotong, for creating table in data mart. ***/ 

-- data mart 
create table proj_piadm.dm_piadm_pre_lock ( 
    p_n_ string ,
    family string ,
    subgeo string ,
    geo string ,
    color string ,
    emmc string ,
    voice_call string ,
    wireless_wan string ,
    shipping_country string ,
    shipment_total string ,
    pup_yymm string ,
    pup_yymm_1 string ,
    pup_yymm_2 string ,
    pup_yymm_3 string ,
    pup_yymm_4 string ,
    pup_yymm_5 string  , 
    batch_number string 
); 

cat ./* > bigfile.sql 2>&1