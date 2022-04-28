# Kinetics-TPS-evaluation
## Evaluation Criteria

The participants are required to provide two types of results. (1) Part State Parsing Result: For each frame in a test video, the participants should provide the predicted boxes of human instances, the predicted boxes of body parts as well as the predicted part state of each body part box. Note that, to reduce uploading burden, we will evaluate these results on the sampled frames of each test video (where the sampling interval is 5-frame). Hence, we encourage participants to provide the results on these frames. (2) Action Recognition Result: The participants should also provide the predicted action for each test video.

Since our goal is to leverage part state parsing for action recognition, we develop a new evaluation metric for this task, where we use Part State Correctness as condition for evaluating action recognition accuracy in a test video.

### Definition of Part State Correctness (PSC)

1. Matching humans in each frame: For each ground truth human box, we find the predicted human box according to IoU, where IoU between this prediction and ground truth box is maximum and should be > HUMAN_IOU_THRESH. If there is no matched prediction, the state correctness of all the body parts (w.r.t., this ground truth human box) is 0.
2. Matching body parts in each human: For a matched prediction of human box, we further compare the predicted boxes and ground truth of each body part. Suppose that, the number of predicted boxes is N_p for a body part. If at least one predicted box is overlapped with the ground truth part box (IOU > PART_IOU_THRESH) and the predicted state of this box has the same tag of the ground truth part, we set the state correctness of this part as 1/N_p.
We set HUMAN_IOU_THRESH as 0.5 and set PART_IOU_THRESH as 0.3.
3. Part State Correctness (PSC) of a test video: For each frame, we first average the state correctness of all the parts in this frame. Then, we can obtain PSC of a test video by averaging all frame-level results. Our PSC can reflect that how many parts can be reliably localized with correct state of gesture.

### Action Recognition Conditioned on PSC

1. For each test video, its video prediction can be seen as the correct one, if and only if its PSC is > PART_STATE_CORRECT_THRESH and its video prediction is the same as the ground truth action class. In this case, we can compute action recognition accuracy (top-1) under the condition of PART_STATE_CORRECT_THRESH.
2. We set PART_STATE_CORRECT_THRESH from 0 to 1, where the step size is 0.0001. Then we plot the curve of (PART_STATE_CORRECT_THRESH, the conditioned video accuracy). Then, we calculate the area under this curve as average video accuracy. We use this as the final evaluation metric (The result will be rounded up to 6 decimal places).

## Submissions
To submit your results to the leaderboard, you must construct a submission zip file that contains the files with the following format.

1. The format of the predicted human box is [lefttop_x, lefttop_y, rightbottom_x, rightbottom_y].
2. The format of the predicted part box is [lefttop_x, lefttop_y, rightbottom_x, rightbottom_y]. The predicted state of each part is a tag that is one of part state classes.
3. The video prediction is a tag that is one of action classes.
4. For each frame, please name the json file as same as the corresponding image file, such as “img_00001.jpg” and “img_00001.json”.
5. In order to prevent the submission files from being too large, we evaluate the results of part state parsing on the sampled frames of each test video (The sampling interval is 5-frame). Hence, the participants are only required to sumbit the results of img_00001.json, img_00006.json and so on. The results of other frames in a test video are not necessary and will not be considered for evaluation.
6. In each frame, the participants are allowed to submit at most 10 predicted human boxes. For a human box, the participants are allowed to submit at most 5 proposals for each body part in this human(at most 10 body parts in a human instance), where each proposal contains the predicted box and the state tag of the corresponding body part.
7. Please upload your result folder as a zip file.


### Detailed Submission Format
#### Part Result Format (json)
pred_part_result.json
```json
{
    "video_name": {
        "label_name": {
            "humans": [
                {
                    "number": 1,
                    "parts": {
                        "part_name" : {
                            "number": 1,
                            "box": [BOX1, BOX2, ...],
                            "verb": [PART STATE1, PART STATE2, ...],
                            "name": part_name,
                        }
                    }
                },
                ...
            ]
        },
        ...
    },
    ...
}
```
example
```json
{
    "video_name": {
        "img_00001.json": {
            "humans": [
                {
                    "number": 1,
                    "parts": {
                        "left_arm": {
                            "number": 1,
                            "box": [
                                [275,91,300,165],
                                [260,85,310,155]
                            ],
                            "verb": [
                                "unbend",
                                "bend"
                            ],
                            "name": "left_arm",
                        },
                        "right_leg": {
                            "number": 2,
                            "box": [
                                [296,188,317,264],
                                [266,199,312,254]
                            ],
                            "verb": [
                                "step_on",
                                "unbend"
                            ],
                            "name": "right_leg",
                        },
                        ...
                    }
                }
                ...
            ]
        },
        "img_00006.json": {
            ...
        },
        ...
    },
    "video_name": {
        ...
    },
    ...
}
```

### Video Result Format (json)
#### pred_vid_result.json
```json
{
    "video_name": "predicted_class",
    ...
}
```
example
```json
{
    "Y0-KQvJjKAw_000057_000067": "predicted_class",
    "hRisIK4NSds_000096_000106": "predicted_class",
    "sFWnb5LJEbw_000000_000010": "predicted_class",
    "95vwx9AidR8_000205_000215": "predicted_class",
    ...
}
```
