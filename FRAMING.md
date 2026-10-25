# Framing: Rawaj Pay mule detection

## Decision

Rawaj Pay must decide which transfers to send to its five fraud analysts for investigation. The threshold should keep the number of alerts within the team's actual daily review capacity while still detecting mule activity. False positives waste analysts' time on legitimate transfers, while false negatives allow suspicious transfers to go undetected and may lead to financial and compliance losses. The task does not specify a numeric daily capacity or error costs, so the team should agree on these before choosing a final operating threshold.

## Metric

Precision@K is useful when the analysts can review only a fixed number of transfers: it measures how many of the top-ranked transfers are actual mule cases. Recall shows how many labelled mule cases are found, and expected cost can balance false positives against false negatives when the business supplies their relative costs. Accuracy alone may be misleading because the classes may be imbalanced and it treats both types of error as equally important. Choose the threshold using labelled validation data and the agreed workload and cost constraints.

## Baseline

The current process without a model is not specified in the task, so it should be documented with the fraud team rather than assumed. A baseline could be the existing rules or manual-review process, evaluated on the same labelled data. A model or threshold strategy should improve on that baseline in useful mule detection and/or error cost without exceeding the analysts' review capacity.

## Classical ML or LLM

For the core task of scoring structured transaction data using labelled examples, classical supervised machine learning is the sensible starting point, with simple rules as a baseline. It can produce risk scores that can be evaluated with precision, recall, and expected cost and used to rank transfers for review. An LLM may help with unstructured investigation notes or case summaries, but it should not be the primary transaction scorer without evidence that it improves the decision. The final choice should be checked against the decision table discussed in the session.
