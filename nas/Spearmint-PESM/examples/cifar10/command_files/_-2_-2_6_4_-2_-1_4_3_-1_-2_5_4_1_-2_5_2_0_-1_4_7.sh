sleep 1; WID=$(xdotool search --name 10.217.132.138 | head -1); xdotool windowactivate $WID ; xdotool mousemove 1032 681 click 1; sleep 1 ; xdotool mousemove 783 584 click 1 ; sleep 1 ; python3 /home/lile/Projects/git_repo/hacone/movidius/mvNCProfile.py -s 12 -network /home/lile/Projects/git_repo/hacone/outputs/random_experiments_on_movidius/_-2_-2_6_4_-2_-1_4_3_-1_-2_5_4_1_-2_5_2_0_-1_4_7/frozen_graph.pb -in input -on CifarNet/Predictions/Reshape_1 >/home/lile/Projects/git_repo/hacone/measurements/random_experiments_on_movidius/_-2_-2_6_4_-2_-1_4_3_-1_-2_5_4_1_-2_5_2_0_-1_4_7.profile ; sleep 5 ; xdotool windowfocus $WID; xdotool mousemove 856 582 click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 784 626 click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 531 495;  xdotool click 1 ; sleep 1; xdotool windowfocus $WID; xdotool mousemove 182 188;  xdotool click 1 ; ssh User@10.217.132.138 'filename=$(ls C:/Users/User/movidius -t | head -1); sleep 1; mv "C:/Users/User/movidius/$filename" C:/Users/User/movidius/_-2_-2_6_4_-2_-1_4_3_-1_-2_5_4_1_-2_5_2_0_-1_4_7.csv';echo "file done" ; sleep 2