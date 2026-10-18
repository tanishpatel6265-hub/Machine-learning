from sklearn.metrics import roc_curve
fpr , tpr , thresholds = roc_curve(y_train_5 , y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90 , fpr_90 = tpr[idx_for_threshold_at_90] , fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6,5))
plt.plot(fpr , tpr , linewidth = 2 ,label = "ROC curve")
plt.plot([0,1] , [0,1] , 'k:' , label = "Random classidier ROC curve")
plt.plot([fpr_90] , [tpr_90] , "ko", label = "threshold for 90% precision")
plt.grid()
plt.legend(loc = "lower right" , fontsize = 12)
plt.xlabel('Fasle Positive Rate (fall- out)' , colour= "#333333")
plt.ylabel('True Positive rate (Recall)' )

save_fig("roc_curve_plot")