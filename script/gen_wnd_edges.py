#coding=utf-8
import os
import sys
import pickle
import logging
from tqdm import tqdm
from collections import namedtuple, Counter

data_path = "dataset/electronics"

reviews_file = os.path.join(data_path, "reviews-info")
train_file = os.path.join(data_path, "local_train_splitByUser")
reviews_cache_file = os.path.join(data_path, "cache", "rating.pkl")
edge_output_file_v1 = os.path.join(data_path, "graph", "edges_train.csv")

if not os.path.exists(os.path.join(data_path, "cache")): 
    os.mkdir(os.path.join(data_path, "cache"))
if not os.path.exists(os.path.join(data_path, "graph")): 
    os.mkdir(os.path.join(data_path, "graph"))
        
        
SAMPLE_SEP = "\t"
SEQ_SEP = "\x02"

SeqItem = namedtuple("SeqItem", ["mid", "cate", "ts"])
Edge = namedtuple("Edge", ["frm", "to"])
WEdge = namedtuple("EdgeEx", ["frm", "to", "freq"])

Rating = namedtuple("Rating", ["uid", "mid", "score", "ts"])
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format="%(asctime)s[%(levelname)s] %(message)s")

def load_ratings(reviews_file, reviews_cache_file = None):
    if reviews_cache_file is not None and os.path.exists(reviews_cache_file):
        logging.info("loading reviews from cache: %s", reviews_cache_file)
        with open(reviews_cache_file, "rb") as fi:
            return pickle.load(fi)
    rating_map = {}
    total_reviews = 0
    logging.info("load ratings from {}".format(reviews_file))
    # with open(reviews_file, encoding="utf8") as fi:
    with open(reviews_file) as fi:
        for i, line in tqdm(enumerate(fi)):
            [uid, mid, score, ts] = line.split("\t")
            rating = Rating(uid, mid, float(score), int(ts))
            total_reviews += 1
            rating_map[(uid, mid)] = rating
    total_ratings = len(rating_map)
    logging.info("total reviews: %d, total ratings: %d", total_reviews, total_ratings)

    if reviews_cache_file is not None:
        with open(reviews_cache_file, "wb") as fo:
            pickle.dump(rating_map, fo)
        logging.info("caching reviews to %s", reviews_cache_file)
    return rating_map


def check_order(item_list):
    last_ts = 0
    for item in item_list:
        ts = item.ts
        if ts >= last_ts:
            last_ts = ts
            continue
        else:
            return False
    return True

def get_pred_edges(dataset_file, rating_map):
    edges = []
    logging.info("loading pred edges from %s", dataset_file)
    # with open(dataset_file, encoding="utf8") as fi:
    with open(dataset_file) as fi:
        for i, line in tqdm(enumerate(fi)):
            [label, uid, mid, cate, his_mid_list, his_cate_list] = line.strip().split(SAMPLE_SEP)
            if label == "0":
                continue
            if his_mid_list.strip() == "":
                continue
            his_mid_list = his_mid_list.split(SEQ_SEP)
            his_cate_list = his_cate_list.split(SEQ_SEP)
            his_item_list = [SeqItem(hmid, hcate, rating_map[(uid, hmid)].ts) for hmid, hcate in zip(his_mid_list, his_cate_list)]
            item_list = his_item_list + [SeqItem(mid, cate, rating_map[(uid, mid)].ts)]
            if not check_order(item_list):
                logging.error("[INVALID ORDER]: {}".format(his_item_list))
                continue
            edge = Edge(item_list[-2].mid, item_list[-1].mid) # edge directly connection last his item and target item.
            edges.append(edge)
    return set(edges)

def gen_full_edges(train_file, rating_map, check_triangle=False):
    edge_counter = Counter()
    # with open(train_file, encoding = "utf8") as fi:
    with open(train_file) as fi:
        error_cnt = 0
        user_cache = {}
        logging.info("loading reviews from %s", train_file)
        for i, line in tqdm(enumerate(fi)):
            [label, uid, mid, cate, his_mid_list, his_cate_list] = line.strip().split(SAMPLE_SEP)
            if label == "0":
                continue
            if his_mid_list.strip() == "":
                continue
            his_mid_list = his_mid_list.split(SEQ_SEP)
            his_cate_list = his_cate_list.split(SEQ_SEP)

            his_item_list = [SeqItem(hmid, hcate, rating_map[(uid, hmid)].ts) for hmid, hcate in zip(his_mid_list, his_cate_list)]
            item_list = his_item_list + [SeqItem(mid, cate, rating_map[(uid, mid)].ts)]

            if not check_order(item_list):
                error_cnt += 1
                logging.error("[INVALID ORDER]: {}".format(item_list))
                continue
            if uid not in user_cache:
                user_cache[uid] = []
            user_cache[uid].extend(item_list)
        logging.info("error count: %d", error_cnt)
        logging.info("couting edges...")
        for i, (uid, item_list) in tqdm(enumerate(user_cache.items())):
            item_list = sorted(list(set(item_list)), key=lambda x:x.ts)
            if len(item_list) < 2:
                continue
            if len(item_list) == 2:
                edge = Edge(item_list[0].mid, item_list[1].mid)
                edge_counter.update([edge])
                continue
            else:
                item_cnt = len(item_list)
                for j in range(item_cnt - 2):
                    [a, b, c] = item_list[j: j + 3]
                    edge_counter.update([Edge(a.mid, b.mid), Edge(a.mid, c.mid), Edge(b.mid, c.mid)])
    edges = [WEdge(edge.frm, edge.to, cnt) for edge, cnt in list(edge_counter.most_common())]
    logging.info("total edges: %d", len(edges))
    return edges


def main():
    rating_map = load_ratings(reviews_file, reviews_cache_file)

    train_pred_edges = get_pred_edges(train_file, rating_map)
    logging.info("total train pred edges: %d", len(train_pred_edges))
    for edge in list(train_pred_edges)[:5]:
        logging.info(edge)
    
    full_edges = gen_full_edges(train_file, rating_map, check_triangle=True)

    logging.info("writing edges to %s", edge_output_file_v1)
    cnt = 0
    with open(edge_output_file_v1, "w") as fo:
        for we in tqdm(full_edges):
            e = Edge(we.frm, we.to)
            if True: #if e not in test_pred_edges:
                line = "{}\t{}\t{}\n".format(we.frm, we.to, we.freq)
                cnt += 1
                fo.write(line)
    logging.info("done! total edges: %d", cnt)
    

def check_edges():
    voc = {}
    with open(data_path+"/mid_voc.csv", encoding = "utf8") as fi:
        for line in fi:
            [name, id] = line.strip().split("\t")
            voc[name] = id

    logging.info("checking mids...")
    missing_cnt = 0
    with open(edge_output_file_v1, encoding = "utf8") as fi:
        for line in tqdm(fi):
            [from_mid, to_mid, freq] = line.strip().split("\t")
            if from_mid not in voc or to_mid not in voc:
                missing_cnt += 1
    logging.info("number of oov mids in %s: %d", edge_output_file_v1, missing_cnt)

    
if __name__ == '__main__':
    main()
    #check_edges()
