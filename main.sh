######START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N run_train
#$ -M 16ee234.megh@nitk.edu.in #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=40G
#$ -l gpu=1
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
export CUDA_VISIBLE_DEVICES 
source activate dss
module unload gcc
module load gcc/5.2.0

# create experiment dump repo
mkdir -p ./exp/deepercluster/

# run unsupervised feature learning
python main.py \
--dump_path ./exp/deepercluster/ \
--data_path ./data/clipart/train \
--size_dataset 20000 \
--workers 4 \
--sobel false \
--lr 0.1 \
--wd 0.00001 \
--nepochs 100 \
--batch_size 48 \
--reassignment 3 \
--dim_pca 4096 \
--super_classes 4 \
--rotnet true \
--k 1 \
--warm_restart false \
--use_faiss true \
--niter 10 \
--world-size 64 \
--dist-url DIST_URL



##--pretrained PRETRAINED \
		 
