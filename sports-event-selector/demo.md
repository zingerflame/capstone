Video preprocessing
1) greyscale
2) greyscale delta
100 -> 20

VLM Teacher
Input (Video)
Output VLM teacher Array(2-3 Key Frames, Associated events) "ground truth"

--> train our models


Deep Learning Model (student)
Input (Video)
Output Array(2-3 Key Frames) (actual output BOOL arry)

CNN -> CNN -> Transformer 

10-frames per window, 1 stride.

6 truth max 

(1-10), (2-11), (3-12) 3 = avg(truth of (1-10,3) ,...) > 0.65 = 1 (maybe hard threshold depending on results)

(1-100)  boolean array of length 100 tells us whether or not each point is a key frame

comparing the distance of each item to the teacher's results.

Punish it by exceeding the number of teacher results -> 2 from 100 and we select 4 from 100

[0,1,0,1,1,1,0,0,0,0,0,1,0,1]

VLM finetuning
Input (2-3 Key Frames)
Output (Events)