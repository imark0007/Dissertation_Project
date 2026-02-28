# SOC Alert Triage Workflow Example

## Scenario
A Dynamic GNN alert fires on a 5-window sequence of IoT traffic.

## Step 1: Analyst receives SIEM alert
The SIEM dashboard shows a new alert with:
- **event.kind**: alert
- **rule.name**: IoT Dynamic GNN Detector
- **ml.score**: 0.97 (high confidence)
- **threat.indicator.confidence**: high

## Step 2: Review explanation
The alert includes an explanation bundle:
- **Top features**: `Rate` (importance: 2.14), `Srate` (1.87), `Header_Length` (1.52)
- **Top nodes**: node_12 (importance: 3.41), node_7 (importance: 2.89)

The analyst sees that the detection was driven by unusually high packet rates
and abnormal header lengths — consistent with a DDoS flood pattern.

## Step 3: Cross-reference with context
The analyst checks:
- Are the top-contributing flows from known IoT devices? (node mapping)
- Does the time window correlate with other alerts? (SIEM timeline)
- Is the traffic pattern consistent with known attack signatures? (threat intel)

## Step 4: Decision
Based on the high model score (0.97), clear feature explanations pointing to
flood-like traffic patterns, and corroboration from other SIEM events, the
analyst escalates the alert for incident response.

## Value of Explainability
Without explanations, the analyst would see only "malicious / score 0.97" and
would need to manually inspect raw flow logs. The top-feature and top-node
explanations reduce triage time by directing attention to the most relevant
traffic characteristics, supporting faster and more confident decisions.
