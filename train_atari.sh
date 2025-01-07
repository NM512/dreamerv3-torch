#!/usr/bin/env bash

############################################
# Default parameters
############################################
LOG_FOLDER_DEFAULT="original/seed_0"
GPU_COUNT_DEFAULT=9

############################################
# Parse command-line arguments
############################################
LOG_FOLDER="$LOG_FOLDER_DEFAULT"
GPU_COUNT="$GPU_COUNT_DEFAULT"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --logfolder)
      LOG_FOLDER="$2"
      shift 2
      ;;
    --gpucount)
      GPU_COUNT="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized parameter: $1"
      echo "Usage: $0 [--logfolder FOLDER] [--gpucount N]"
      exit 1
      ;;
  esac
done

echo "Using LOG_FOLDER='${LOG_FOLDER}'"
echo "Using GPU_COUNT='${GPU_COUNT}'"

############################################
# List of tasks (Atari environments)
############################################
TASKS=(
  ##adventure
  ##air_raid
  alien
  amidar
  assault
  asterix
  #asteroids
  #atlantis
  bank_heist
  battle_zone
  #beam_rider
  #berzerk
  #bowling
  boxing
  breakout
  ##carnival
  #centipede
  chopper_command
  crazy_climber
  #defender
  demon_attack
  #double_dunk
  ##elevator_action
  #enduro
  #fishing_derby
  freeway
  frostbite
  gopher
  #gravitar
  hero
  #ice_hockey
  jamesbond
  ##journey_escape
  kangaroo
  krull
  kung_fu_master
  #montezuma_revenge
  ms_pacman
  #name_this_game
  #phoenix
  #pitfall
  pong
  ##pooyan
  private_eye
  qbert
  #riverraid
  road_runner
  #robotank
  seaquest
  #skiing
  #solaris
  #space_invaders
  #star_gunner
  #surround
  #tennis
  #time_pilot
  #tutankham
  up_n_down
  #venture
  #video_pinball
  #wizard_of_wor
  #yars_revenge
  #zaxxon
)

############################################
# Initialize array to track PIDs per GPU
############################################
declare -a GPU_PIDS
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  GPU_PIDS[$gpu]=0
done

############################################
# Function to start a single training run
############################################
start_training() {
  local task=$1
  local gpu_id=$2

  echo "Starting atari_${task} on cuda:${gpu_id}"
  python3 dreamer.py \
    --configs atari100k \
    --task atari_"${task}" \
    --logdir ./log_atari100k/"${LOG_FOLDER}"/"${task}" \
    --device cuda:"${gpu_id}" &

  # Store the PID of the newly launched process
  GPU_PIDS[$gpu_id]=$!
}

############################################
# Main loop over tasks
############################################
for task in "${TASKS[@]}"; do
  while true; do
    # Check each GPU to see if its PID is still running
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
      
      # If GPU_PIDS[gpu] == 0, it means no process is running there
      if [ "${GPU_PIDS[$gpu]}" -eq 0 ]; then
        # GPU is free, start training here
        start_training "$task" "$gpu"
        break 2
      fi

      # If GPU_PIDS[gpu] != 0, check if that PID is still alive
      if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
        # Process on this GPU is finished, so the GPU is free
        start_training "$task" "$gpu"
        break 2
      fi
    done

    # If we did not break, it means all GPUs are busy, so wait a bit and check again
    sleep 2
  done
done

############################################
# Wait for all remaining training to finish
############################################
wait
echo "All training runs have completed!"
