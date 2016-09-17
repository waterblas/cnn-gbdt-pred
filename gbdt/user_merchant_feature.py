import numpy as np
from collections import Counter
from datetime import datetime
import math
import sys, getopt


ACT_CLICK = '0'
ACT_ADD2CART = '1'
ACT_BUY = '2'
ACT_FAVOUR = '3'
MONTH_RANGE = 6

origin_f = "../data/data_format2/train_format2.csv"
target_f = "../output/user_merchant_feature_offline.txt"
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"hm:")
except getopt.GetoptError:
    print 'user_merchant_feature.py -m <mode>'
    print 'mode: predict/offline'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-m' and arg == 'predict':
        origin_f = "../data/data_format2/test_format2.csv"
        target_f = "../output/user_merchant_feature.txt"
print 'Extract feature from %s to %s.' % (origin_f, target_f)


class UmFeature(object):
    
    def __init__(self):
        self.date_format = "%m%d"
        self.reset()

    def reset(self):
        self.buy_log = []
        self.click_log = []
        self.add2cart_log = []
        self.favourite_log = []
    
    def parse_log(self, log):
        records = log.split('#')
        for record in records:
            item_id, category_id, brand_id, time_stamp, action_type = record.split(":")
            if action_type == ACT_CLICK:
                self.click_log.append([item_id, category_id, brand_id, time_stamp])
            elif action_type == ACT_BUY:
                self.buy_log.append([item_id, category_id, brand_id, time_stamp])
            elif action_type == ACT_ADD2CART:
                self.add2cart_log.append([item_id, category_id, brand_id, time_stamp])
            elif action_type == ACT_FAVOUR:
                self.favourite_log.append([item_id, category_id, brand_id, time_stamp])

    def action_num(self, log):
        action_count = [0]*MONTH_RANGE
        before_1111 = 0
        for ele in log:
            item_id, category_id, brand_id, time_stamp = ele
            action_count[int(time_stamp[:2]) % MONTH_RANGE] += 1
            if time_stamp != "1111":
                before_1111 += 1
        feature = list(action_count) + [before_1111, len(log), np.mean(action_count), np.std(action_count), np.max(action_count), np.median(action_count)]
        return feature

    def days_num(self, log):
        days_count = [0] * MONTH_RANGE
        days = set()
        for ele in log:
            item_id, category_id, brand_id, time_stamp = ele
            days.add(time_stamp)
        for ele in days:
            days_count[int(ele[:2]) % MONTH_RANGE] += 1
        return [len(days), np.mean(days_count), np.std(days_count), np.max(days_count), np.median(days_count)]

    def distinct_num(self, log):
        item_set = set()
        brand_set = set()
        cat_set = set()
        item_set_before1111 = set()
        brand_set_before1111 = set()
        cat_set_before1111 = set()
        for ele in log:
            item_id, category_id, brand_id, time_stamp = ele
            item_set.add(item_id)
            brand_set.add(brand_id)
            cat_set.add(category_id)
            if time_stamp != "1111":
                item_set_before1111.add(item_id)
                brand_set_before1111.add(brand_id)
                cat_set_before1111.add(category_id)
        return [len(item_set), len(brand_set), len(cat_set), len(item_set_before1111), len(brand_set_before1111), len(cat_set_before1111)]

    def item_detail(self, log):
        if len(log) < 1:
            return [0] * 4
        item_count = Counter()
        day_count = Counter()
        days_set = set()
        for ele in log:
            item_id, category_id, brand_id, time_stamp = ele
            item_count[item_id] += 1
            days_set.add("%s_%s" % (item_id, time_stamp))
        for ele in days_set:
            item, _ = ele.split("_")
            day_count[item] += 1
        item_num_3 = 0
        day_num_3 = 0
        for ele in item_count:
            if item_count[ele] > 2:
                item_num_3 += 1
        for ele in day_count:
            if day_count[ele] > 2:
                day_num_3 += 1
        feature = [max(item_count.values()), max(day_count.values()), item_num_3, day_num_3]
        return feature
            
    def brand_detail(self, log):
        if len(log) < 1:
            return [0] * 4
        brand_count = Counter()
        day_count = Counter()
        days_set = set()
        for ele in log:
            brand_id, category_id, brand_id, time_stamp = ele
            brand_count[brand_id] += 1
            days_set.add("%s_%s" % (brand_id, time_stamp))
        for ele in days_set:
            brand, _ = ele.split("_")
            day_count[brand] += 1
        brand_num_3 = 0
        day_num_3 = 0
        for ele in brand_count:
            if brand_count[ele] > 2:
                brand_num_3 += 1
        for ele in day_count:
            if day_count[ele] > 2:
                day_num_3 += 1
        feature = [max(brand_count.values()), max(day_count.values()), brand_num_3, day_num_3]
        return feature

    def repeat_time(self):
        alllog = self.buy_log + self.click_log + self.favourite_log + self.add2cart_log
        time_set = set()
        for ele in alllog:
            brand_id, category_id, brand_id, time_stamp = ele
            time_set.add(time_stamp)
        if len(time_set) < 2:
            return [0]*3
        else:
            time_section = [datetime.strptime(ele, self.date_format) for ele in sorted(time_set)]
            time_distance = np.zeros(len(time_section)-1)
            for i in range(len(time_distance)):
                time_distance[i] = (time_section[i+1] - time_section[i]).days
            return [np.mean(time_distance), np.std(time_distance), np.max(time_distance)]
            

    def count(self):
        alllog = [self.buy_log, self.click_log, self.favourite_log, self.add2cart_log]
        not_lead_log = [self.click_log, self.favourite_log, self.add2cart_log] 
        feature = []
        for log in alllog:
            feature += self.action_num(log)
            feature += self.days_num(log)
            feature += self.distinct_num(log)

        for log in not_lead_log:
            feature += self.item_detail(log)
            feature += self.brand_detail(log)
        feature += self.repeat_time()
        return feature
            

with open(origin_f, 'r') as f, open(target_f, 'w') as f1:
    f.readline()
    ex = UmFeature()
    for idx, line in enumerate(f):
        if idx % 70000 == 0:
            sys.stdout.write(">")
            sys.stdout.flush()
        user_id, age_range, gender, merchant_id, label, activity_log= line.rstrip().split(",")
        if label == '-1' or len(activity_log) < 1 :
            continue
        ex.parse_log(activity_log)
        feature = ex.count()
        feature = ','.join([str(ele) for ele in feature])
        f1.write("%s,%s,%s,%s,%s#%s\n" % (user_id, merchant_id, label, age_range, gender, feature))
        ex.reset()

