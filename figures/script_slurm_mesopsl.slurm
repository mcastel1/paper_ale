#!/bin/sh
#SBATCH --job-name=ani-figure_3
#SBATCH --time=01-00:0:00
#SBATCH --clusters=mesopsl2
#SBATCH --qos=mesopsl2_hi_normal
#SBATCH --account=castellana_hi
#SBATCH --partition=hi
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1


echo "Entering the virtual environment .. "
source ~/miniconda3/bin/activate myenv
echo " .. done."


FIGURE_NAME='figure_3'
RUN=$HOME/"paper_ale/figures/"$FIGURE_NAME

cd $RUN
rm -rf *.mp4 *.pdf

python3 animate.py 

ffmpeg -i animation_${FIGURE_NAME}.mp4 -c:v libx265 -tag:v hvc1 -crf 23 -preset slow -pix_fmt yuv420p animation_${FIGURE_NAME}_small.mp4

