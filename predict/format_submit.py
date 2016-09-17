import shutil

office_set = set()
with open("../data/data_format1/test_format1.csv", "r") as f:
    f.readline()
    for line in f:
        user_id, merchant_id, _ = line.rstrip().split(",")
        office_set.add("%s,%s" % (user_id, merchant_id))
print len(office_set)

pred_set = set()
with open("./submission_pre.csv", "r") as f:
    for line in f:
        user_id, merchant_id, _ = line.rstrip().split(",")
        pred_set.add("%s,%s" % (user_id, merchant_id))
print len(pred_set)

diff = list(office_set.difference(pred_set))
print len(diff)

shutil.copyfile("./submission_pre.csv", "submission.csv")
with open("./submission.csv", "a") as f:
    for ele in diff:
        f.write("%s,%s\n" % (ele, 0))
