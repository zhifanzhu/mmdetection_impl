" set tags+=../mmcv/.tags
set tags+=~/.conda/envs/mmlab/lib/python3.7/site-packages/mmcv-0.2.7-py3.7.egg/mmcv/.tags
set tags+=~/.conda/envs/mmlab/lib/python3.7/site-packages/torch/.tags

set wildignore+=data/**

let dirs='.,'.system("find . -maxdepth 1 -type d | cut -d/ -f2 | grep '^[^.]' | grep -v 'data' | awk '{print}' ORS='/**,'")
let &path=dirs
" Equivalatent
" exe "set path=".dirs
