import librosa
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import lmdb
import pickle
from beat_detector import BD

parser = argparse.ArgumentParser()

parser.add_argument('-in', '--input_dir', type=str, required=True, help='directory with audio files')
parser.add_argument('-out', '--output_dir', type=str, required=True, help='directory to save output')
parser.add_argument('-sr', '--sampling_rate', type=int, required=True, help='audio sampling rate, automatic resampling is performed if specified sr is different from native')
parser.add_argument('-l', '--bandpass_left', type=int, default=700, help='bandpass filters left cutoff')
parser.add_argument('-r', '--bandpass_right', type=int, default=1300, help='bandpass filters right cutoff')
parser.add_argument('-tg', '--save_textgrids', action='store_true', help='if used, textgrid will be saved in [output_dir]/textgrids')
parser.add_argument('-w', '--save_filtered', action='store_true', help='if used, filtered audios will be saved in [output_dir]/filtered_wavs')
parser.add_argument('-plt', '--save_plots', action='store_true', help='if used, plots will be saved in [output_dir]/plots')

args = parser.parse_args()

bd = BD( sr = args.sampling_rate, bandpass_left=args.bandpass_left, bandpass_right=args.bandpass_right)

# Script will only read .wav files in the specified directory, ignoring subdirectories

file_list = [f"{args.input_dir}/{f}" for f in os.listdir(f'{args.input_dir}') if os.path.isfile(f"{args.input_dir}/{f}") and f.split('.')[-1] == 'wav']
gt_base_path = os.path.join(args.input_dir, 'textgrids_gt')

os.makedirs(args.output_dir, exist_ok=True)
out_file = os.path.join(args.output_dir, "rtm_feats.lmdb")
map_size = 100 * 1024**3
env = lmdb.open(out_file, map_size=map_size, writemap=True, map_async=True)

txn = env.begin(write=True)
commit_interval = 2000

for i, file_path in enumerate(tqdm(file_list, desc="Processing audios")):
    waveform, sr = librosa.load(file_path, sr=args.sampling_rate)
    beats = bd.detect_beats(waveform)
    
    # Generate a unique ID
    identifier = os.path.basename(file_path)
    
    entry = {
        'key': identifier,
        'beats': beats['beat_frames'],
        'envelope_spectrum': beats['envelope_spectrum'],
        'feats': beats['features'],
        'intervals': beats['beat_intervals']
    }
    
    serialized_entry = pickle.dumps(entry)
    txn.put(identifier.encode('utf-8'), serialized_entry)
    
    # Commit periodically to avoid memory issues
    if (i + 1) % commit_interval == 0:
        txn.commit()
        txn = env.begin(write=True)

    if args.save_textgrids:
      tg_path = os.path.join(args.output_dir, 'textgrids')
      os.makedirs(tg_path, exist_ok=True)
      out_file = os.path.join(tg_path, identifier[:-4] + ".TextGrid")
      bd.save_textgrid(beats[0], out_file)
    

    if args.save_filtered:
      filtered_path = os.path.join(args.output_dir, 'filtered_wavs')
      os.makedirs(filtered_path, exist_ok=True)
      out_file = os.path.join(filtered_path, identifier[:-4] + ".wav")
      bd.write_wav(out_file)

    if args.save_plots:
      gt_path = os.path.join(gt_base_path, Path(identifier).stem + '.TextGrid')
      plots_path = os.path.join(args.output_dir, 'plots')
      os.makedirs(plots_path, exist_ok=True)
      out_file = os.path.join(plots_path, Path(identifier).stem + '.png')
      if os.path.isfile(gt_path):
        bd.plot_filtered(out_file, gt_path)
      else:
        bd.plot_filtered(out_file)

      plots_feats_path = os.path.join(args.output_dir, 'plots_feats')
      os.makedirs(plots_feats_path, exist_ok=True)
      out_file = os.path.join(plots_feats_path, Path(identifier).stem + '.png')
      bd.plot_feats(out_file)
  
txn.commit()
env.close()