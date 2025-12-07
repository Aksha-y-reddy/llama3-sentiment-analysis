Completed: Cell_Phones_and_Accessories Baseline
I have successfully completed training the first baseline model on the Cell_Phones_and_Accessories category. Here are the results:
Overall Performance
Accuracy: 76.7%
Macro Precision: 0.776
Macro Recall: 0.771
Macro F1: 0.763
Per-Class Performance
Class
Precision
Recall
F1
Support
Negative
0.671
0.890
0.765
327
Neutral
0.734
0.531
0.616
343
Positive
0.925
0.891
0.907
330

Key Observations
Strong Negative Class Detection (89% recall): This is particularly important for our poisoning research, as attacks typically target the minority negative class. The model demonstrates a robust capability to identify negative sentiment.
Balanced Performance: Unlike the imbalanced experiments we discussed previously, this model shows no majority class bias due to the balanced sampling strategy (50,000 samples per class).
Neutral Class Challenge: The neutral class shows lower recall (53%), which is expected as neutral sentiment is inherently ambiguous. The confusion matrix shows neutral reviews are often classified as negative (40%) or positive (6%).
Training Stability: The model showed consistent improvement with no overfitting (validation loss: 1.218, training loss: 1.210).
Training Configuration
Training data: 150,000 balanced samples (50K per class)
Method: QLoRA (4-bit quantization)
Training time: ~6 hours on A100 80GB
Epochs: 1
