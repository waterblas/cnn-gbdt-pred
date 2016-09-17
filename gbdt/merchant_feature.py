from collections import Counter
import numpy as np
import sys, getopt


origin_f1 = "../data/data_format1/train_format1.csv"
origin_f2 = "../data/data_format2/train_format2.csv"
target_f = "../output/merchant_feature_offline.pkl"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"hm:")
except getopt.GetoptError:
    print 'merchant_feature.py -m <mode>'
    print 'mode: predict/offline'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-m' and arg == 'predict':
        origin_f1 = "../data/data_format1/test_format1.csv"
        origin_f2 = "../data/data_format2/test_format2.csv"
        target_f = "../output/merchant_feature.pkl"
print 'Extract feature from (%s, %s) to %s.' % (origin_f1, origin_f2, target_f)

MONTH_RANGE =6
features = {}
fst_feature = {}
with open(origin_f1) as f:
    for line in f:
        _, merchant, _ = line.rstrip().split(',')
        if merchant not in fst_feature:
            features[merchant] = None
            fst_feature[merchant] = {
                'action':[0] * 4,
                'action_month': [0] * 4 * 6,
                'days': [set(), set(), set(), set()],
                'users': [set(), set(), set(), set()],
                'items': [set(), set(), set(), set()],
                'user_days': [set(), set(), set(), set()],
            }

print len(fst_feature)
def count_action(month_arr):
    res = np.zeros(16).reshape(4,4)
    for i in range(0,4):
        at = np.array(month_arr[6*i:6*(i+1)])
        res[i] = [np.mean(at), np.std(at), np.max(at), np.median(at)]
    return list(res.reshape(16))

def count_days(days_arr):
    month_res = np.zeros(24).reshape(4,6)
    for i in range(0,4):
        for record in days_arr[i]:
            month_res[i][int(record[:2]) % MONTH_RANGE] += 1
    res = np.zeros(16).reshape(4,4)
    for i in range(0,4):
        res[i] = [np.mean(month_res[i]), np.std(month_res[i]), np.max(month_res[i]), np.median(month_res[i])]
    return list(res.reshape(16))

def count_user(user_arr):
    user_res = np.zeros(24).reshape(4,6)
    for i in range(0,4):
        for record in user_arr[i]:
            record = record.split('_')[-1]
            user_res[i][int(record[:2]) % MONTH_RANGE] += 1
    res = np.zeros(16).reshape(4,4)
    for i in range(0,4):
        res[i] = [np.mean(user_res[i]), np.std(user_res[i]), np.max(user_res[i]), np.median(user_res[i])]
    return list(res.reshape(16))

def count_item(item_arr):
    res = np.zeros(16).reshape(4,4)
    for i in range(0,4):
        c = Counter()
        for ele in item_arr[i]:
            c[ele.split('_')[0]]+=1
        at = np.array(list(c.values()))
        if len(at) > 0:
            res[i] = [np.mean(at), np.std(at), np.max(at), np.median(at)]
    return list(res.reshape(16))

def count_user_days(user_days_arr):
    res = np.zeros(24).reshape(4,6)
    for i in range(0,4):
        c = Counter()
        for ele in user_days_arr[i]:
            c[ele.split('_')[0]]+=1
        at = np.array(list(c.values()))
        if len(at) > 0:
            res[i] = [len(c), len(user_days_arr[i]), np.mean(at), np.std(at), np.max(at), np.median(at)]
    return list(res.reshape(24))

def count_repeat_user_ratio(user_arr):
    res = np.zeros(4)
    for i in range(0,4):
        c = Counter()
        idx = 0
        for ele in user_arr[i]:
            c[ele.split('_')[0]]+=1
        for ele in c:
            if c[ele] > 2:
                idx += 1
        if len(c) > 0 :
            res[i] = idx * 1.0 / len(c)
    return list(res)
    
with open(origin_f2, "r") as f:
    f.readline()
    for idx, line in enumerate(f):
        if idx % 70000 == 0:
            sys.stdout.write(">")
            sys.stdout.flush()
        user_id, age_range, gender, merchant_id, label, activity_log= line.rstrip().split(",")
        if len(activity_log) < 1 or merchant_id not in fst_feature:
            continue
        for ele in activity_log.split('#'):
            try:
                item_id, category_id, brand_id, time_stamp, action_type = ele.split(':')
            except:
                print line
                raise Exception
            fst_feature[merchant_id]['action'][int(action_type)] += 1
            fst_feature[merchant_id]['action_month'][int(action_type) * 6 + (int(time_stamp[:2]) % MONTH_RANGE)] += 1
            fst_feature[merchant_id]['days'][int(action_type)].add(time_stamp)
            fst_feature[merchant_id]['users'][int(action_type)].add('%s_%s' % (user_id, time_stamp))
            fst_feature[merchant_id]['items'][int(action_type)].add('%s_%s' % (user_id, item_id))
            fst_feature[merchant_id]['user_days'][int(action_type)].add('%s_%s' % (user_id, time_stamp))
    for ele in fst_feature:
        feature = fst_feature[ele]['action'] + fst_feature[ele]['action_month']+  count_action(fst_feature[ele]['action_month'])\
        + [len(fst_feature[ele]['days'][i]) for i in range(4)] + count_days(fst_feature[ele]['days']) + count_user(fst_feature[ele]['users'])\
        + count_item(fst_feature[ele]['items']) + count_user_days(fst_feature[ele]['user_days']) + count_repeat_user_ratio(fst_feature[ele]['users'])
        features[ele] = feature

import cPickle as pickle
with open(target_f, 'w') as f:
    pickle.dump(features, f, 2)

