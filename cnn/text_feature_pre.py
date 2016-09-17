import numpy as np
import sys, getopt

origin_f = "../data/data_format2/train_format2.csv"
target_f = "../output/cnn_features.txt"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"hm:")
except getopt.GetoptError:
    print 'text_feature_pre.py -m <mode>'
    print 'mode: predict/offline'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-m' and arg == 'predict':
        origin_f = "../data/data_format2/test_format2.csv"
print 'Extract feature from %s to %s.' % (origin_f, target_f)

ACT_CLICK = '0'
ACT_ADDCART = '1'
ACT_BUY = '2'
ACT_FAVOURITE = '3'

def extract_feature(u_data, fo):
    if len(u_data["merchant"]) < 1:return
    for merchant in u_data["merchant"]:
        fo.write("%s,%s,%s,%s,%s#%s#%s#%s\n" % (\
                merchant[0], u_data["uid"], merchant[1], u_data["age"], u_data["gender"],\
                ','.join(sorted(u_data['brand'])), ','.join(sorted(u_data['purchase_brand'])), ','.join(sorted(u_data['purchase_category']))\
                )
        )
            
    
u_data = {"uid":0, "age": 0, "gender": 0, "merchant": set(),  "brand": set(), "purchase_category":set(), "purchase_brand": set()}
idx = 0
with open(origin_f, "r") as f, open(target_f, "w") as f1:
    f.readline()
    current_uid = -1
    for line in f:
        uid, age, gender, merchant, label, log = line.rstrip().split(",")
        if uid != current_uid:
            extract_feature(u_data, f1) 
            u_data = {"uid":uid, "age": age, "gender": gender, "merchant": set(),  "brand": set(), "purchase_category":set(), "purchase_brand": set()}
            current_uid = uid

        if label != '-1':
            u_data["merchant"].add((label, merchant))
            
        for mlog in log.split("#"):
            try:
                item, cat, brand, ts , action = mlog.split(":")
            except:
                #sys.stdout.write(">")
                #sys.stdout.flush()
                continue
            if action != ACT_BUY:
                u_data['brand'].add('%s_%s' % (ts, brand))
            if action == ACT_BUY:
                u_data['purchase_category'].add('%s_%s' % (ts, cat))
                u_data['purchase_brand'].add('%s_%s' % (ts, brand))
    extract_feature(u_data, f1)
            

