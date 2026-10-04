export const statusText = (status: string) => ({ queued: 'Waiting', running: 'Analyzing', completed: 'Ready', failed: 'Failed', cancelled: 'Cancelled' }[status] || status)
export const taskText = (task: string) => task === 'descriptive' ? 'Exploratory analysis' : task === 'clustering' ? 'Discover similar groups' : task === 'classification' ? 'Category prediction' : 'Numeric prediction'
export const goalText = (run: { task: string; target: string; objective?: string }) => run.task === 'descriptive' ? run.objective || 'Explore the evidence' : run.task === 'clustering' ? 'Discover similar groups' : `Predict ${run.target.replaceAll('_', ' ')}`

export const metrics: Record<string, { title: string; explanation: string }> = {
  accuracy: { title: 'Accuracy', explanation: 'The share of test examples with the correct predicted category. Higher is better.' },
  balanced_accuracy: { title: 'Balanced accuracy', explanation: 'Recall averaged equally across classes, so minority outcomes are not overwhelmed by the majority.' },
  positive_precision: { title: 'Positive precision', explanation: 'The share of predicted positive outcomes that are correct; the positive label is recorded.' },
  positive_recall: { title: 'Positive recall', explanation: 'The share of actual positive outcomes detected by the classifier.' },
  f1: { title: 'F1 score', explanation: 'Balances precision and recall, weighted by class size. Higher is better.' },
  f1_macro: { title: 'Balanced F1', explanation: 'F1 averaged equally across categories, so small classes matter too. Higher is better.' },
  precision: { title: 'Precision', explanation: 'How often predicted categories are correct, averaged by class size. Higher is better.' },
  recall: { title: 'Recall', explanation: 'How many examples of each category the model finds, averaged by class size. Higher is better.' },
  roc_auc: { title: 'ROC AUC', explanation: 'How well the model ranks positive examples above negative ones. Higher is better.' },
  pr_auc: { title: 'Precision–recall score', explanation: 'Average precision across recall thresholds, useful for imbalanced binary outcomes. Higher is better.' },
  mae: { title: 'Average error (MAE)', explanation: 'The average absolute difference between predicted and actual values, in target units. Lower is better.' },
  rmse: { title: 'Prediction error (RMSE)', explanation: 'An error measure in target units that puts more weight on large mistakes. Lower is better.' },
  r2: { title: 'R² score', explanation: 'Performance relative to predicting the test-set mean. One is ideal; values can be negative.' },
  silhouette: { title: 'Group separation', explanation: 'Silhouette measures how distinct the groups are, from −1 to +1. Higher is better.' },
  davies_bouldin: { title: 'Group overlap', explanation: 'The Davies–Bouldin index measures overlap relative to separation. Lower is better.' },
  calinski_harabasz: { title: 'Separation ratio', explanation: 'The Calinski–Harabasz index compares between-group and within-group variation. Higher is better.' },
  clusters: { title: 'Groups found', explanation: 'The number of non-empty groups assigned to the held-out test rows.' },
}
