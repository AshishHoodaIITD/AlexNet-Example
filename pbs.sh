#!/bin/sh
### Set the job name
#PBS -N test
### Set the project name, your department dc by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $cs1140425@iitd.ac.in
####
#PBS -l select=1:ngpus=1:ncpus=4
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=09:00:00

#PBS -l software=
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd /home/cse/btech/cs1140254/DeepLearn/
#job 
python test.py > output.txt
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
