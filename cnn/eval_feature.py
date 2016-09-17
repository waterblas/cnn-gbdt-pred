import numpy as np
import cPickle as pickle
import gzip
from collections import Counter
import sys, getopt


def generate_dep(orgin_f, tmp_f, id2num_f, predict_mode):
    ag2idx = {}
    m2idx = {}
    b2idx = {}
    pb2idx = {}
    pc2idx = {}
    m_index = 1
    b_index = 1
    pb_index = 1
    pc_index = 1
    ag_index = 1
    with open(orgin_f, 'r') as f, open(tmp_f, 'w') as f2:
        for line in f:
            info, brand, purchase_brand, purchase_cat = line.rstrip().split('#')
            label, uid, merchant, age, gender  = info.split(',')
            # predict mode
            if label == '':
                label = 2
            ball = []
            pball = []
            pcall = []
            ag = "a%s_%s" % (age, gender)
            if len(brand) > 0:
                for ele in brand.split(','):
                    ts, child_ele = ele.split('_')
                    ball.append(child_ele)
            if len(purchase_brand) > 0:
                for ele in purchase_brand.split(','):
                    ts, child_ele = ele.split('_')
                    pball.append(child_ele)
            if len(purchase_cat) > 0:
                for ele in purchase_cat.split(','):
                    ts, child_ele = ele.split('_')
                    pcall.append(child_ele)
            if merchant not in m2idx:
                m2idx[merchant] = m_index
                m_index += 1
            if ag not in ag2idx:
                ag2idx[ag] = ag_index
                ag_index += 1

            for ele in ball:
                if ele not in b2idx:
                    b2idx[ele] = b_index
                    b_index += 1
            for ele in pball:
                if ele not in pb2idx:
                    pb2idx[ele] = pb_index
                    pb_index += 1
            for ele in pcall:
                if ele not in pc2idx:
                    pc2idx[ele] = pc_index
                    pc_index += 1
            f2.write('%s#%s#%s#%s#%s#%s#%s\n' % (uid, label, ag, merchant, ",".join(ball), ",".join(pball), ",".join(pcall)))
    if predict_mode:
        return
    with open(id2num_f, 'w') as f:
        pickle.dump([ag2idx, m2idx, b2idx, pb2idx, pc2idx], f, protocol=2)

def convert2vec(tmp_f, serail_f, id2num_f):
    brand_size = 80
    purchase_brand_size = 30
    purchase_cat_size = 30

    um_arr = []
    ag_arr = []
    m_arr = []
    b_arr = []
    pc_arr = []
    pb_arr = []
    y_train = []
    with open(id2num_f, 'r') as f:
        ag2idx, m2idx, b2idx, pb2idx, pc2idx = pickle.load(f)
        print "ag:%s, m:%s, b:%s, pb:%s, pc:%s" % (len(ag2idx), len(m2idx), len(b2idx), len(pb2idx), len(pc2idx))

    with open(tmp_f, 'r') as f:
        lines = f.readlines()
    line2ids = np.random.permutation(len(lines))
    for k in line2ids:
        line = lines[k]
        uid, label, ag, merchant, brand_s, purchase_brand_s, purchase_cat_s = line.rstrip().split('#')
        b_tmp = map(lambda x: b2idx[x] if x in b2idx else 0, brand_s.split(',')[:brand_size])
        pb_tmp = map(lambda x: pb2idx[x] if x in pb2idx else 0, purchase_brand_s.split(',')[:purchase_brand_size])
        pc_tmp = map(lambda x: pc2idx[x] if x in pc2idx else 0, purchase_cat_s.split(',')[:purchase_cat_size])
        while len(b_tmp) < brand_size:
            b_tmp.append(0)
        while len(pb_tmp) < purchase_brand_size:
            pb_tmp.append(0)
        while len(pc_tmp) < purchase_cat_size:
            pc_tmp.append(0)
        um_arr.append("%s_m%s" % (uid, merchant))
        ag_arr.append(ag2idx[ag] if ag in ag2idx else 0) 
        m_arr.append(m2idx[merchant] if merchant in m2idx else 0) 
        b_arr.append(b_tmp)
        pb_arr.append(pb_tmp)
        pc_arr.append(pc_tmp)
        y_train.append(np.array(int(label)).astype('int32'))
    x_train = [um_arr, np.array(ag_arr, dtype='int'), np.array(m_arr, dtype='int'), np.array(b_arr, dtype='int'), np.array(pb_arr, dtype='int'), np.array(pc_arr, dtype='int')]
    y_train = np.array(y_train, dtype='int')
    with open(serail_f, 'w') as f:
        pickle.dump((x_train, y_train), f, 2)


if __name__ == '__main__':
    argv = sys.argv[1:]
    predict_mode = False 
    try:
        opts, args = getopt.getopt(argv,"hm:")
    except getopt.GetoptError:
        print 'eval_feature.py -m <mode>'
        print 'mode: predict/offline'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-m' and arg == 'predict':
            predict_mode = True
    # feature generate
    generate_dep('../output/cnn_features.txt', '../output/tmp_feature.txt', '../output/id2num.pkl', predict_mode)
    convert2vec('../output/tmp_feature.txt', '../output/serial_feature.pkl', '../output/id2num.pkl')



