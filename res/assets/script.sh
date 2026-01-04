#!/usr/bin/env bash

#https://superuser.com/a/556031
# -ss: skip first 10s, -t: duration 5s
# setpts=0.5*PTS makes it 2x faster (1/0.5 = 2)
# fps=10 keeps the output at 10 frames per second (dropping the intermediate frames)


ffmpeg -y -ss 1  -t 33 \
    -i house.mp4 \
    -vf "setpts=0.24*PTS,fps=10,scale=820:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 house.gif

ffmpeg -y -ss 0 -t 10 -i orbit.mp4 \
    -vf "setpts=1*PTS,fps=10,scale=820:-1:flags=lanczos,reverse,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 orbit.gif

ffmpeg -y -i house.gif -i orbit.gif \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
    -map "[v]" comparison.gif