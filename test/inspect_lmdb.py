import lmdb
import pickle

lmdb_path = '/home/joao.lima/experiments/rhythm_classifier/data/MSP/rtm_feats.lmdb'
env = lmdb.open(lmdb_path, readonly=True)

with env.begin() as txn:
    entry = pickle.loads(txn.get(b'MSP-PODCAST_0002_0033.wav'))
    cursor = txn.cursor()
    for key, value in cursor:
        entry = pickle.loads(value)
        print(entry['beats'])
        break

env.close()