# AI usage

## Tools I used, and for which functions

I used ChatGPT to understand the task requirements and help draft audit() and the metric functions in `metrics.py`. I also used claude to review edge cases involving missing values, data leakage, input validation, and threshold selection.

## One place where the AI was wrong, incomplete, or not what the spec said

While testing `best_threshold()`, I tried a score equal to positive infinity. The function returned `math.inf`, but `confusion_at()` still flagged that case because `float("inf") >= float("inf")` evaluates to `True`. I discovered this by running a targeted test and comparing the result with the specification. This revealed an ambiguity: `math.inf` is described as the threshold for flagging nothing, but the general rule says to flag scores greater than or equal to the threshold. The specification does not explicitly explain how infinite scores should be handled.

## The function I wrote without AI, and why

I have not yet completed a function entirely without AI assistance. Before submitting, I plan to implement `confusion_at()` independently so I can practise counting true positives, false positives, true negatives, and false negatives. I will test my implementation using a small example whose results I calculate by hand.