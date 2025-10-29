#!/bin/sh
#SBATCH --job-name=ani-figure_1
#SBATCH --time=00-00:10:00
#SBATCH --clusters=mesopsl2
#SBATCH --qos=mesopsl2_hi_normal
#SBATCH --account=castellana_hi
#SBATCH --partition=hi
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1


echo "Entering the virtual environment .. "
source ~/miniconda3/bin/activate myenv
echo " .. done."


FIGURE_NAME='figure_1'
RUN=$HOME/"paper_ale/figures/"$FIGURE_NAME

cd $RUN
rm -rf *.mp4 *.pdf

python3 animate_$FIGURE_NAME.py 

ffmpeg -i animation_large_$FIGURE_NAME.mp4 -c:v libx265 -tag:v hvc1 -crf 23 -preset slow -pix_fmt yuv420p animation_$FIGURE_NAME.mp4

