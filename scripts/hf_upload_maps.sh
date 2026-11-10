#!/usr/bin/env bash
# Upload a maps folder to the Hub dataset abshetty/floz-diag under its tag:  scripts/hf_upload_maps.sh data/floz-diag/<tag>
set -a; . ~/.env; set +a
python3 -c "
import sys; from huggingface_hub import HfApi
d=sys.argv[1].rstrip('/'); t=d.split('/')[-1]
HfApi().upload_folder(folder_path=d, path_in_repo=t, repo_id='abshetty/floz-diag', repo_type='dataset', ignore_patterns=['.mapsdone'], commit_message=f'maps {t}')
print('UPLOADED', t)" "$1"
