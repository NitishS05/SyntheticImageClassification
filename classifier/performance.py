
def performance_graph():
	import numpy as np
	import matplotlib.pyplot as plt

	# data to plot
	n_groups= 4
	index = np.arange(n_groups)
	CNN= (0.9851, 0.97, 0.99, 0.98)
	SVM= (0.9259, 0.92, 0.92, 0.93)
	DT= (0.9629, 0.97, 0.96, 0.96)

	# create plot
	fig, ax = plt.subplots()
	bar_width = 0.20
	opacity = 0.5

	plt.bar(index, SVM, bar_width,alpha=opacity,color='r', label='SVM')
	plt.bar(index+bar_width, DT, bar_width,alpha=opacity,color='g', label='DT')
	plt.bar(index+bar_width+bar_width, CNN, bar_width,alpha=opacity,color='b', label='CNN')

	plt.xlabel('Parameters')
	plt.ylabel('Scores')
	plt.title('Performance analysis of Model')
	plt.xticks(index+bar_width, ("Accuracy","Precision","Recall","F1-score"))
	plt.legend()

	plt.tight_layout()
	plt.show()

	return