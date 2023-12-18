if [ ! -d '/dev/shm/zhangyx' ]; then
  mkdir /dev/shm/zhangyx || exit
  cd /dev/shm/zhangyx || exit
fi

####
dataset_1='Art'
dataset_2='Clipart'
dataset_3='Product'
dataset_4='Real'

if [ ! -d '/dev/shm/zhangyx/${dataset_1}' ]; then
  cp /gdata1/zhangyx/DataSet/office_home/${dataset_1}.tar /dev/shm/zhangyx
  tar xf ./${dataset_1}.tar
  rm /dev/shm/zhangyx/${dataset_1}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_2}' ]; then
  cp /gdata1/zhangyx/DataSet/office_home/${dataset_2}.tar /dev/shm/zhangyx
  tar xf ./${dataset_2}.tar
  rm /dev/shm/zhangyx/${dataset_2}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_3}' ]; then
  cp /gdata1/zhangyx/DataSet/office_home/${dataset_3}.tar /dev/shm/zhangyx
  tar xf ./${dataset_3}.tar
  rm /dev/shm/zhangyx/${dataset_3}.tar
fi
if [ ! -d '/dev/shm/zhangyx/${dataset_4}' ]; then
  cp /gdata1/zhangyx/DataSet/office_home/${dataset_4}.tar /dev/shm/zhangyx
  tar xf ./${dataset_4}.tar
  rm /dev/shm/zhangyx/${dataset_4}.tar
fi
###