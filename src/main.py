import librosa
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import lmdb
import pickle
from rhythm import RTM

parser = argparse.ArgumentParser()

parser.add_argument('-in', '--input_dir', type=str, required=True, help='directory with audio files')
parser.add_argument('-out', '--output_dir', type=str, required=True, help='directory to save output')
parser.add_argument('-sr', '--sampling_rate', type=int, required=True, help='audio sampling rate, automatic resampling is performed if specified sr is different from native')
parser.add_argument('-l', '--bandpass_left', type=int, default=700, help='bandpass filters left cutoff')
parser.add_argument('-r', '--bandpass_right', type=int, default=1300, help='bandpass filters right cutoff')
parser.add_argument('-env', '--envelope_method', type=str, choices=['bark', 'vocallic_energy'], help='envelope extraction method')
parser.add_argument('-D', '--max_dur', type=int, default=10, help='maximum duration (in seconds) of audio files to be processed')
parser.add_argument('-d', '--min_dur', type=int, default=2, help='minimum duration (in seconds) of audio files to be processed')
parser.add_argument('-mfcc', '--compute_mfcc', action='store_true', help='if used, MFCC features will be computed and stored in the database')
parser.add_argument('-tg', '--save_textgrids', action='store_true', help='if used, textgrid will be saved in [output_dir]/textgrids')
parser.add_argument('-w', '--save_filtered', action='store_true', help='if used, filtered audios will be saved in [output_dir]/filtered_wavs')
parser.add_argument('-plt', '--save_plots', action='store_true', help='if used, plots will be saved in [output_dir]/plots')

args = parser.parse_args()

processor = RTM(sr = args.sampling_rate, env_method = args.envelope_method, bandpass_left=args.bandpass_left, bandpass_right=args.bandpass_right)

# Gather all .wav files in the input directory including subdirs
file_list = [str(p) for p in Path(args.input_dir).rglob('*.wav')]
gt_base_path = os.path.join(args.input_dir, 'textgrids_gt')
os.makedirs(args.output_dir, exist_ok=True)
out_file = os.path.join(args.output_dir, "rtm_feats.lmdb")
map_size = 100 * 1024**3
env = lmdb.open(out_file, map_size=map_size, writemap=True, map_async=True)

# Check if the .lmdb file already exists
existing_keys = set()
if os.path.exists(out_file):
    with lmdb.open(out_file, readonly=True, lock=False) as env_read:
        with env_read.begin() as txn_read:
            existing_keys = set(k.decode('utf-8') for k in txn_read.cursor().iternext(values=False))

txn = env.begin(write=True)
commit_interval = 2000
durations = []
num_skipped = 0

for i, file_path in enumerate(tqdm(file_list, desc="Processing audios")):
    identifier = "_".join(file_path.split("/")[-2:])

    # Skip if identifier already exists in the database
    if identifier in existing_keys:
        continue

    waveform, sr = librosa.load(file_path, sr=args.sampling_rate)
    duration = len(waveform) / sr

    if duration <= args.min_dur or duration >= args.max_dur:
       num_skipped += 1
       continue
    else:
      durations.append(duration)
      beats = processor.process(waveform,
                                identifier,
                                write_dir = args.output_dir,
                                gt_tg_dir = gt_base_path,
                                write_tgs = args.save_textgrids,
                                write_filtered = args.save_filtered,
                                write_plots = args.save_plots)
      
      entry = {
          'key': identifier,                                  # str   
          'envelope': beats['envelope'],                      
          'beats': beats['beat_frames'],                      # list(int)
          'envelope_spectrum': beats['envelope_spectrum'],    # np.array array_shape (num_freq_bins, )
          'spectrum_freq_bins': beats['spectrum_freq_bins'],
          'intervals': beats['beat_intervals'],               # list(float)
          'dur': duration
      }

      if args.compute_mfcc:
          entry['mfcc_features'] = beats['mfcc_features']      # list(array) array_shape (num_beats, num_mfccs)
      
      serialized_entry = pickle.dumps(entry)
      txn.put(identifier.encode('utf-8'), serialized_entry)
      
      # Commit periodically to avoid memory issues
      if (i + 1) % commit_interval == 0:
          txn.commit()
          txn = env.begin(write=True)
  
txn.commit()
env.close()

with open(os.path.join(args.output_dir, 'metadata.txt'), 'w') as f:
    num_processed = len(durations)
    f.write(f"Number of processed audio files: {num_processed}\n")
    f.write(f"Sampling Rate: {args.sampling_rate}\n")
    f.write(f"Bandpass Filter: {args.bandpass_left} Hz - {args.bandpass_right} Hz\n")
    f.write(f"Maximum Audio Duration: {args.max_dur} seconds\n")
    if num_processed > 0:
        f.write(f"Total Duration of Processed Audios: {np.sum(durations):.2f} seconds\n")
        f.write(f"Average Duration of Processed Audios: {np.mean(durations):.2f} seconds\n")
        f.write(f"Standard Deviation of Durations: {np.std(durations):.2f} seconds\n")
    else:
        f.write("No audio files were processed (all skipped or already in database).\n")