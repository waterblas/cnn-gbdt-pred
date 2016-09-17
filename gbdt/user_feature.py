import numpy as np
from collections import Counter
from datetime import datetime
import math
import sys


ACT_CLICK = '0'
ACT_ADD2CART = '1'
ACT_BUY = '2'
ACT_FAVOUR = '3'
MONTH_RANGE = 6

origin_f = "../data/data_format1/user_log_format1.csv"
target_f = "../output/user_feature.txt"


class Features(object):
    def __init__(self, uid):
        self.uid = uid
        self.reset(uid)
        self.date_format = "%m%d"
    def reset(self, uid):
        self.buy_log = []
        self.click_log = []
        self.add2cart_log = []
        self.favourite_log = []
        self.uid = uid

    def user_action(self, log):
        action_count = [0]*MONTH_RANGE
        for ele in log:
            mid = int(ele.split(',')[-1][:2])
            action_count[mid % MONTH_RANGE] += 1
        return [len(log), np.mean(action_count), np.std(action_count), np.max(action_count), np.median(action_count)] + action_count

    def merchant_days_with_action(self, log):
        
        days_count = np.zeros(MONTH_RANGE)
        merchant_count = np.zeros(MONTH_RANGE)
        days = set()
        merchants = set()
        for ele in log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            days.add(time_stamp)
            merchants.add('%s_%s' % (seller_id, time_stamp))
        for ele in days:
            days_count[int(time_stamp) % MONTH_RANGE] += 1
        for ele in merchants:
            merchant_count[int(ele.split('_')[-1][:2]) % MONTH_RANGE] += 1
        days_feature = [len(days), np.mean(days_count), np.std(days_count), np.max(days_count), np.median(days_count)]
        merchant_feature = [len(merchants), np.mean(merchant_count), np.std(merchant_count), np.max(merchant_count), np.median(merchant_count)]
        return days_feature + merchant_feature

    def item_merchant(self, log):
        if len(log) < 1:
            return [0] * 10
        item_count = Counter()
        days_count = Counter()
        item_set = set()
        days_set = set()
        for ele in log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            item_set.add('%s_%s' % (item_id, seller_id))
            days_set.add('%s_%s' % (time_stamp, seller_id))
        for ele in item_set:
            item_id, mid = ele.split('_')
            item_count[mid] += 1
        for ele in days_set:
            time_stamp, mid = ele.split('_')
            days_count[mid] += 1
        merchant_item = np.array(item_count.values())
        merchant_days = np.array(days_count.values())
        item_feature = [np.mean(merchant_item), np.std(merchant_item), np.max(merchant_item), np.median(merchant_item)]
        days_feature = [len(merchant_item), np.sum(merchant_days), np.mean(merchant_days), np.std(merchant_days), np.max(merchant_days), np.median(merchant_days)]
        return item_feature + days_feature

    def repeat_action(self, log):
        merchant_set = set()
        merchant_count = Counter()
        for ele in log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            merchant_set.add("%s_%s" % (seller_id, time_stamp))
        for ele in merchant_set:
            mid = ele.split('_')[0]
            merchant_count[mid] += 1
        repeat_merchant = set()
        for ele in merchant_count:
            if merchant_count[ele] > 1:
                repeat_merchant.add(ele)
        repeat_days = set()
        all_days = set()
        for ele in log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            if seller_id in repeat_merchant:
                repeat_days.add(time_stamp)
            all_days.add(time_stamp)
        ratio_merchant = len(repeat_merchant) * 1.0 / len(merchant_count) if len(merchant_count) > 0 else 0
        ratio_days = len(repeat_days) * 1.0 / len(all_days) if len(all_days) > 0 else 0
        return [ratio_merchant, ratio_days]

    def distinct_action(self, log):
        merchant_set = set()
        item_set = set()
        brand_set = set()
        cat_set = set()
        for ele in log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            merchant_set.add(seller_id)
            item_set.add(item_id)
            brand_set.add(brand_id)
            cat_set.add(cat_id)
        return [len(merchant_set), len(item_set), len(brand_set), len(cat_set)]

    def repeat_time(self):
        features = [0] * 3
        all_log = [self.click_log, self.add2cart_log, self.favourite_log]
        time_set = set()
        purchase_time_set = set()
        for log in all_log:
            for ele in log:
                item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
                time_set.add(time_stamp)
        for ele in self.buy_log:
            item_id,cat_id,seller_id,brand_id,time_stamp = ele.split(',')
            time_set.add(time_stamp)
            purchase_time_set.add(time_stamp)
        if len(time_set) < 2:
            features[0] = 0
        else:
            time_section = [datetime.strptime(ele, self.date_format) for ele in sorted(time_set)]
            time_distance = np.zeros(len(time_section)-1)
            for i in range(len(time_distance)):
                time_distance[i] = (time_section[i+1] - time_section[i]).days
            features[0] = np.mean(time_distance)
        if len(purchase_time_set) < 2:
            features[1] = 0
            features[2] = 0
        else:
            purchase_time_section = [datetime.strptime(ele, self.date_format) for ele in sorted(purchase_time_set)]
            purchase_time_distance = np.zeros(len(purchase_time_section)-1)
            for i in range(len(purchase_time_distance)):
                purchase_time_distance[i] = (purchase_time_section[i+1] - purchase_time_section[i]).days
            features[1] = math.log(1+np.mean(purchase_time_distance))
            features[2] = math.log(1+purchase_time_distance[-1])
        return features
            

    def count(self):
        log = [self.click_log, self.add2cart_log, self.buy_log, self.favourite_log]
        user_feature = []
        for ele in log:
            tmp = self.user_action(ele) + self.merchant_days_with_action(ele)+ self.item_merchant(ele)+ self.repeat_action(ele)+ self.distinct_action(ele)
            user_feature += tmp
        user_feature += self.repeat_time()
        return user_feature


    def save(self, f):
        user_feature = self.count()
        f.write("%s\t%s\n" % (self.uid, ','.join([str(ele) for ele in user_feature])))

    def add(self, item_id,cat_id,seller_id,brand_id,time_stamp,action_type):
        if action_type == ACT_CLICK:
            self.click_log.append("%s,%s,%s,%s,%s" % (item_id,cat_id,seller_id,brand_id,time_stamp))
        elif action_type == ACT_ADD2CART:
            self.add2cart_log.append("%s,%s,%s,%s,%s" % (item_id,cat_id,seller_id,brand_id,time_stamp))
        elif action_type == ACT_BUY:
            self.buy_log.append("%s,%s,%s,%s,%s" % (item_id,cat_id,seller_id,brand_id,time_stamp))
        elif action_type == ACT_FAVOUR:
            self.favourite_log.append("%s,%s,%s,%s,%s" % (item_id,cat_id,seller_id,brand_id,time_stamp))
        
    

with open(origin_f, 'r') as f, open(target_f, 'w') as f1:
    featurs = Features(0)
    f.readline()
    for idx, line in enumerate(f):
        if idx % 500000 == 0:
            sys.stdout.write(">")
            sys.stdout.flush()
        user_id,item_id,cat_id,seller_id,brand_id,time_stamp,action_type = line.rstrip().split(',')
        if user_id != featurs.uid:
            featurs.save(f1)
            featurs.reset(user_id)

        featurs.add(item_id,cat_id,seller_id,brand_id,time_stamp,action_type)
    featurs.save(f1)

            
            


