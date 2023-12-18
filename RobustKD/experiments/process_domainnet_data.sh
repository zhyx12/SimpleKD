if [ ! -d '/dev/shm/zhangyx' ]; then
  mkdir /dev/shm/zhangyx || exit
  cd /dev/shm/zhangyx || exit
fi

####
dataset_1='clipart'
dataset_2='infograph'
dataset_3='painting'
dataset_4='quickdraw'
dataset_5='real'
dataset_6='sketch'


if [ ! -d '/dev/shm/zhangyx/${dataset_1}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_1}.tar /dev/shm/zhangyx
  tar xf ./${dataset_1}.tar
  rm /dev/shm/zhangyx/${dataset_1}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_2}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_2}.tar /dev/shm/zhangyx
  tar xf ./${dataset_2}.tar
  rm /dev/shm/zhangyx/${dataset_2}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_3}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_3}.tar /dev/shm/zhangyx
  tar xf ./${dataset_3}.tar
  rm /dev/shm/zhangyx/${dataset_3}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_4}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_4}.tar /dev/shm/zhangyx
  tar xf ./${dataset_4}.tar
  rm /dev/shm/zhangyx/${dataset_4}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_5}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_5}.tar /dev/shm/zhangyx
  tar xf ./${dataset_5}.tar
  rm /dev/shm/zhangyx/${dataset_5}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_6}' ]; then
  cp /data/zhyx12/SSDA_Dataset/domainnet/${dataset_6}.tar /dev/shm/zhangyx
  tar xf ./${dataset_6}.tar
  rm /dev/shm/zhangyx/${dataset_6}.tar
fi
# pip install fastda -U
###