if [ $HOME == '/' ]; then
  if [ -d "/ghome/zhangyx" ]; then
    export HOME=/ghome/zhangyx
    echo 'HOME is '${HOME}
  elif [ -d "/ghome/huxm" ]; then
    export HOME=/ghome/huxm
    echo "HOME is "${HOME}
  fi
elif [ $HOME == '/root' ]; then
  if [ -d "/home/zhangyx" ]; then
    export HOME=/home/zhangyx
    echo 'HOME is '${HOME}
  else
    export HOME='/code'
  fi
else
  :
fi