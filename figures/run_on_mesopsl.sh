#!/bin/bash
# run with ./run_on_mesopsl [name of folder on local] 
# run with ./run_on_mesopsl.sh figure_14

clear
clear

# define the paths 
ROOT_PATH=/Users/michelecastellana/Documents/paper_ale/
FIGURES_PATH=$ROOT_PATH'figures/'
FIGURE_PATH=$FIGURES_PATH''$1
MODULES_PATH=/Users/michelecastellana/Documents/python_modules
OUT=mesopslt    

echo 'root path = ' $ROOT_PATH
echo 'figures path = ' $FIGURES_PATH
echo 'figure path = ' $FIGURE_PATH


# remove old stuff on the remote
# ssh $OUT "rm -rf /obs/mcastellana/paper_ale/figures/"$1
ssh $OUT "mkdir -p paper_ale/figures/"$1
ssh $OUT 'rm -rf /obs/mcastellana/paper_ale/figures/'"$1"'/slurm*'
ssh $OUT 'rm -rf /obs/mcastellana/paper_ale/figures/'"$1"'/*.mp4'
ssh $OUT 'rm -rf /obs/mcastellana/paper_ale/figures/'"$1"'/*.pdf'

# sync all modules necessary to make the plot or animation
rsync -avz --delete \
  --exclude='f*' \
  --exclude='dFdl*' \
  --exclude='F_el*' \
  --exclude='v_bar*' \
  --exclude='phi_*' \
  --exclude='u_el_dot*' \
  --exclude='u_msh_dot*' \
  --exclude='def_v_bar*' \
  --exclude='def_phi*' \
  --exclude='def_sigma*' \
  --exclude='sigma_*' \
  --exclude='*.msh' \
  --exclude='.DS_Store' \
  --exclude='*.xdmf' \
  --exclude='*.h5' \
  --exclude='*.mp4' \
  --exclude='*.pdf' \
  --exclude='*.pyc' \
  "$FIGURE_PATH/" mesopslt:paper_ale/figures/"$1"
rsync -av --delete $MODULES_PATH/* mesopslt:paper_ale/figures/modules 
rsync -av --delete $ROOT_PATH/*.tex mesopslt:paper_ale
rsync -av --delete $FIGURES_PATH/*.tex mesopslt:paper_ale/figures

# replace FIGURE_TO_PLOT with the name of the actual figure to plot in the slurm script 
sed 's/FIGURE_TO_PLOT/'$1'/g' templet_script_slurm_mesopsl.slurm > script_slurm_mesopsl.slurm
rsync -av --delete script_slurm_mesopsl.slurm mesopslt:paper_ale/figures/$1

ssh "$OUT" "bash -l -c 'cd paper_ale/figures/$1 && sbatch script_slurm_mesopsl.slurm'"