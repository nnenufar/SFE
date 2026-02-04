import lmdb
import pickle
import numpy as np 

lmdb_path = '/hadatasets/joao.lima/data/quick_samples/out/test_envPad/rtm_feats.lmdb'
env = lmdb.open(lmdb_path, readonly=True)

with env.begin() as txn:
    entry = pickle.loads(txn.get(b'audios_arctic_a0006.wav'))
    cursor = txn.cursor()
    for key, value in cursor:
        print(f'Key: {key}\n')
        entry = pickle.loads(value)

        feats = entry['feats']
        print(f'Feats type: {type(feats)} of {type(feats[0])}')
        print(f'Feats shape: {np.array(feats).shape}\n')

        beats = entry['beats']
        print(f'Beats type: {type(beats)} of {type(beats[0])}')
        print(f'Num Beats: {len(beats)}\n')

        envSpec = entry['envelope_spectrum']
        print(f'EnvSpec type: {type(envSpec)}')
        print(f'EnvSpec shape: {envSpec.shape}')

        freqs = entry['spectrum_freq_bins']
        print(f'Freqs type: {type(freqs)}')
        print(f'Freqs shape: {freqs.shape}')

        intervals = entry['intervals']
        print(f'Intervals type: {type(intervals)} of {type(intervals[-1])}')
        print(f'Num Intervals: {len(intervals)}\n')

        break

env.close()