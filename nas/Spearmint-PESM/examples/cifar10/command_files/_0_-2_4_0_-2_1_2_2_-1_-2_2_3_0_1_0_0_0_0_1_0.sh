sleep 5; WID=$(xdotool search --name 10.217.132.118 | head -1); xdotool windowfocus $WID ; xdotool mousemove 783 584 click 1 ; sleep 1 ; cd /home/lile/Projects/git_repo/hacone/movidius; export WORKON_HOME=/home/lile/Projects/.virtualenvs; source /usr/local/bin/virtualenvwrapper.sh; workon movidius; make all MODEL_DIR=/home/lile/Projects/git_repo/hacone/models_trained/_0_-2_4_0_-2_1_2_2_-1_-2_2_3_0_1_0_0_0_0_1_0/ LOG_FILE=/home/lile/Projects/git_repo/hacone/measurements/gpu_on_movidius/_0_-2_4_0_-2_1_2_2_-1_-2_2_3_0_1_0_0_0_0_1_0.profile ; sleep 15 ; xdotool windowfocus $WID; xdotool mousemove 856 582 click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 784 626 click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 531 495;  xdotool click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 182 188;  xdotool click 1 ; ssh User@10.217.132.118 'filename=$(ls C:/Users/User/movidius -t | head -1); sleep 1; mv "C:/Users/User/movidius/$filename" C:/Users/User/movidius/_0_-2_4_0_-2_1_2_2_-1_-2_2_3_0_1_0_0_0_0_1_0.csv';echo "file done" ; sleep 2