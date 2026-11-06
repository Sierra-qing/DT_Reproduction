# 使用方法：source reproduction/activate_official.sh
# 仅激活独立的官方环境，并补充 mujoco-py 要求的动态库搜索路径。
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate decision-transformer-gym || return
_DT_OFFICIAL_MUJOCO_BIN=/home/ubuntu/.mujoco/mujoco200/bin
case ":${LD_LIBRARY_PATH-}:" in
  *":${_DT_OFFICIAL_MUJOCO_BIN}:"*) ;;
  *) export LD_LIBRARY_PATH="${_DT_OFFICIAL_MUJOCO_BIN}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
unset _DT_OFFICIAL_MUJOCO_BIN
