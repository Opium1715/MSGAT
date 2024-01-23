import argparse
import time
import csv
import pickle
import operator
import datetime
import os

# 使用的数据集 默认是小的取样样本 可选择yoochoose,diginetica#
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset')
opt = parser.parse_args()
print('USING DATASET is {}'.format(opt.__getattribute__('dataset')))

diginetica_path = "D:/dataSet/dataset-train-diginetica/train-item-views.csv"

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'
elif opt.dataset == 'diginetica':
    dataset = diginetica_path

print("-----  开始处理数据集 @ %ss" % datetime.datetime.now())

with open(dataset, 'r') as f:
    if opt.dataset == 'yoochoose':
        # 读取.dat文件并给每列置一个具体的列名
        reader = csv.DictReader(f, delimiter=',', fieldnames=["SessionID", "Timestamp", "ItemID", "Category"])
    else:
        reader = csv.DictReader(f, delimiter=";")
    session_clicks = {}
    session_date = {}
    ctr = 0
    cur_id = -1
    cur_date = None
    # count for limit
    count_limit = 0
    for data in reader:
        # print(data)
        session_id = data["sessionId"]
        if cur_date and cur_id != session_id:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(cur_date[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(cur_date, '%Y-%m-%d'))
            session_date[cur_id] = date
        cur_id = session_id
        if opt.dataset == 'yoochoose':
            item = data["itemID"]
        else:
            item = data["itemId"], int(data["timeframe"])
        cur_date = ''
        if opt.dataset == 'yoochoose':
            cur_date = data["Timestamp"]
        else:
            cur_date = data['eventdate']

        if session_id in session_clicks:
            session_clicks[session_id].append(item)
        else:
            session_clicks[session_id] = [item]
        ctr += 1
        count_limit += 1
        # if count_limit > 100000:  # 104w
        #     break
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(cur_date[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(cur_date, '%Y-%m-%d'))
        for i in list(session_clicks):
            sorted_clicks = sorted(session_clicks[i], key=operator.itemgetter(1))
            session_clicks[i] = [c[0] for c in sorted_clicks]
    session_date[cur_id] = date
print("----- 读取数据 @ %ss" % datetime.datetime.now())

# 过滤长度为1的session
for s in list(session_clicks):
    if len(session_clicks[s]) == 1:
        del session_clicks[s]
        del session_date[s]

# 记录每一个商品出现的次数
iid_counts = {}
for s in session_clicks:
    seq = session_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(session_clicks)

for s in list(session_clicks):
    curseq = session_clicks[s]
    # 从某一条session中筛选出满足出现次数大于等于5的item
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    # 如果一次session中满足出现次数大于5的item总数小于2那么剔除该次session, 否则保留
    if len(filseq) < 2:
        del session_clicks[s]
        del session_date[s]
    else:
        session_clicks[s] = filseq

# 分割测试集
# 找出最大的时间戳
sorted_session_date = sorted(session_date.items(), key=operator.itemgetter(1))
maxdate = sorted_session_date[len(sorted_session_date) - 1][1]

# 分割出7天数据作为测试集
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # 一天共有 86400 seconds
else:
    splitdate = maxdate - 86400 * 7

print('分割时间戳点:{}'.format(splitdate))
dates = list(session_date.items())
train_sess = filter(lambda x: x[1] < splitdate, dates)
test_sess = filter(lambda x: x[1] > splitdate, dates)

# 通过时间戳对session进行排序
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))
# print('训练集session数量：{}'.format(len(train_sess)))
# print('测试集session数量：{}'.format(len(test_sess)))
print(" 完成训练集与测试集分割 @ %ss" % datetime.datetime.now())

item_dict = {}


def get_train():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in train_sess:
        seq = session_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq.append(item_dict[i])
            else:
                outseq.append(item_ctr)
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:
            continue
        train_ids.append(s)
        train_dates.append(date)
        train_seqs.append(outseq)
    # print(item_ctr)
    return train_ids, train_dates, train_seqs


def get_test():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in test_sess:
        seq = session_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = get_train()
tes_ids, tes_dates, tes_seqs = get_test()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print('训练样本：{}'.format(len(tr_seqs)))
print('测试样本：{}'.format(len(te_seqs)))

all = 0
for seq in tr_seqs:
    all += len(seq)
for seq in te_seqs:
    all += len(seq)
print("序列平均长度{}".format(all / (len(tr_seqs) + len(te_seqs))))
# all = 0
# for seq in tra_seqs:
#     all += len(seq)
# for seq in tes_seqs:
#     all += len(seq)
# print('序列平均长度: {}'.format(all / (len(tra_seqs) + len(tes_seqs) * 1.0)))

if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.pkl', 'wb'))
    pickle.dump(tes, open('diginetica/test.pkl', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.pkl', 'wb'))
    # for name, data in {'train': tr_seqs, 'test': te_seqs}.items():
    #     with open(f'diginetica/{name}.txt', 'wb') as datafiles:
    #         pickle.dump(data, )
    #         wr.writerows(data)

elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))
    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

print('数据预处理完成于 @ %ss' % datetime.datetime.now())
