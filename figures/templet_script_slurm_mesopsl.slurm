#!/bin/sh
#SBATCH --job-name=ani-FIGURE_TO_PLOT
#SBATCH --time=04-00:0:00
#SBATCH --clusters=mesopsl2
#SBATCH --qos=mesopsl2_def_long
#SBATCH --account=castellana
#SBATCH --partition=def
#SBATCH --ntasks-per-node=1
#SBATCH--mem=50G
#SBATCH --nodes=1


echo "Entering the virtual environment .. "
source ~/miniconda3/bin/activate myenv
echo " .. done."


FIGURE_NAME='FIGURE_TO_PLOT'
RUN=$HOME/"paper_ale/figures/"$FIGURE_NAME

cd $RUN
rm -rf *.mp4 *.pdf

python3 animate.py 

ffmpeg -i animation_${FIGURE_NAME}.mp4 -c:v libx265 -tag:v hvc1 -crf 23 -preset slow -pix_fmt yuv420p animation_${FIGURE_NAME}_small.mp4

