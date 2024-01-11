1 Dataset
raw data  download: dataset C and dataset D  https://www.dropbox.com/sh/ist4ojr03e2oeuw/AAD5NkpAFg1nOI2Ttug3h2qja?dl=0 
initial events：raw data  sets are transferred to initial events，dataset C --> data/events_initial_A、dataset D -->data/events_initial_B

2 Dataprocess

step1: cal_faults.py read raw data  faults.csv from dataset C and dataset D, construct offline_data_set_info.json和online_data_set_info.json 

step2: runtime evn configuration, log4j 

step3: use the method proposed in this paper to preprocess event data for train & test

step4: DataSetGraphSimGenerator.py split train & test dataset 
 

3 Method

#--------------- fault KG constuction-----
KBConstruction.py  

#---------------KGroot------------------------
graph_sim_dej_X.py


#-------KGroot without GCN--------------
graph_sim_no_gcn_dej_X.py


#---------KGroot without KG--------------

graph_sim_no_kb_dej_X.py
