import numpy as np

#read data into numpy arrays
def prep_data():
	f = open('train.txt','r')
	train_data = f.readlines()
	f.close()

	f = open('test.txt','r')
	test_data = f.readlines()
	f.close()

	train_idx = []
	test_idx = []
	train_feats = []
	test_feats = []
	train_labels = []
	test_labels = []

	for data in train_data:
		data = (data.replace("\n","")).split(",")
		train_idx.append(data[0])
		train_labels.append(data[-1])
		train_feats.append(np.array(data[1:-1],dtype=np.float64))

	for data in test_data:
		data = (data.replace("\n","")).split(",")
		test_idx.append(data[0])
		test_labels.append(data[-1])
		test_feats.append(np.array(data[1:-1],dtype=np.float64))

	train_feats = np.vstack(train_feats)
	test_feats = np.vstack(test_feats)
	train_labels = np.array(train_labels,dtype=np.int)
	test_labels = np.array(test_labels,dtype=np.int)
	train_idx = np.array(train_idx,dtype=np.int)
	test_idx = np.array(test_idx,dtype=np.int)

	return train_idx, train_feats, train_labels, test_idx, test_feats, test_labels

def normalize_train(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	return (data - mean)/std, mean, std

def normalize_test(data, mean, std):
	return (data - mean)/std

def compute_l1(train_feats, test_feats):
	num_train = train_feats.shape[0]
	num_test = test_feats.shape[0]
	l1_dist = np.zeros((num_test,num_train))
	for i in range(num_test):
		l1_dist[i,:] = np.sum(np.absolute(train_feats - test_feats[i,:]), axis = 1)
	return l1_dist

#vectorized implementation
def compute_l2_vec(train_feats, test_feats):
	#sqrt((x-y)**2) = sqrt(x**2 + y**2 - 2x*y)
	sq_sum_train = np.sum(np.square(train_feats), axis=1)
	sq_sum_test = np.sum(np.square(test_feats), axis=1)
	inner_product = np.dot(test_feats, train_feats.T)
	return np.sqrt(sq_sum_train - 2 * inner_product + sq_sum_test.reshape(-1,1))

def compute_l2(train_feats, test_feats):
	num_train = train_feats.shape[0]
	num_test = test_feats.shape[0]
	l2_dist = np.zeros((num_test,num_train))
	for i in range(num_test):
		l2_dist[i,:] = np.sqrt(np.sum(np.square(train_feats - test_feats[i,:]), axis=1))
	return l2_dist	

def most_common(l):
	#make a list with contains how many times each label appeared in the list, like a hash map
	counts = [0 for j in range(max(l)+1)]
	for i in l:
		counts[i] += 1
	labels = []
	max_count = max(counts)
	for i in range(len(counts)):
		if counts[i] == max_count:
			labels.append(i)
	return labels
	
def predict(dist, train_labels, k):
	tags = [0,1,2,3,4,5,6,7]
	num_test = dist.shape[0]
	pred = np.zeros(num_test, dtype=np.int)
	for i in range(dist.shape[0]): 
		#retreive the closest k class labels
		k_labels = train_labels[np.argsort(dist[i,:])[:k]].tolist()
		closest_labels = most_common(k_labels)
		if len(closest_labels) == 1:
			pred[i] = closest_labels[0]
		else:
			#break tie by taking the one with least distance from test
			#find first_occurance of the class in the sorted list(by distance) k_labels	
			first_occ = [k_labels.index(j) for j in closest_labels] #index of the closest example from each tied class
			pred[i] = k_labels[min(first_occ)]
	return pred

def train_acc(train_feats, trian_labels, d_type, k):
	#performing leave-one-out cross validation
	pred = [] 
	val_labels = []
	if d_type == 'l1':
		dist = compute_l1(train_feats, train_feats)
 	else:
 		dist = compute_l2(train_feats, train_feats) 
	#perform leave-one-out cross-validation
	for i in range(dist.shape[0]):
		#remove the label and computed dist of the actual example and add it to val_labels and delete that from labels for classification
		rem_labels = trian_labels.copy()
		val_labels.append(train_labels[i])
		rem_labels = np.delete(rem_labels,i)
		val_dist = dist[i,:].copy()
		val_dist = np.delete(val_dist, i).reshape(1,-1)
		pred.append(predict(val_dist, rem_labels, k)[0])	
	pred = np.array(pred, dtype=np.int)
	val_labels = np.array(val_labels, dtype=np.int)

	return accuracy(val_labels, pred)

def accuracy(true_labels, pred):
	total_correct = np.sum(pred == true_labels)
	accuracy = float(total_correct) / true_labels.shape[0]
	return accuracy

def estimate_mean_std(train_feats, train_labels):
	mean = []
	std = []
	labels = [1,2,3,5,6,7]
	for label in labels:
		#print train_feats[train_labels == label].shape
		mean.append(np.mean(train_feats[train_labels == label], axis=0))
		std.append(np.std(train_feats[train_labels == label], axis=0))
	return np.vstack(mean), np.vstack(std)

def eval_gaussian(feats, mean, std):
	#if std =0, just making it inf, so it doesnt give me a warning
	zero_indxs = std == 0
	std[std == 0] = float("inf")

	exp_term = np.square(feats - mean) / (2 * np.square(std))
	denominator = 1 / (np.sqrt(2 * np.pi) * std)
	prob_density = denominator * np.exp(-1 * exp_term)
	#set the features with zero variance with zero conditional probability
	prob_density[zero_indxs] = 0 
	return prob_density

def compute_scores(train_feats, test_feats, feat_mean, feat_std, prior_probs):
	#as we have six classes
	num_train = train_feats.shape[0]
	num_test = test_feats.shape[0]
	num_classes = 6

	train_scores = np.zeros((num_train, num_classes))
	test_scores = np.zeros((num_test, num_classes))

	for i in range(num_train):
		train_scores[i,:] = np.sum(np.log(eval_gaussian(train_feats[i,:], feat_mean, feat_std)), axis=1) + prior_probs
	for i in range(num_test):
		test_scores[i,:] = np.sum(np.log(eval_gaussian(test_feats[i,:], feat_mean, feat_std)), axis=1) + prior_probs

	return train_scores, test_scores

def naive_bayes_predict(scores):
	num_examples = scores.shape[0]
	tags = np.array([1,2,3,5,6,7], dtype=np.int)
	#find the index of the class with max score, i.e. the last column of the sorted indexes by value
	sorted_idx = np.argsort(scores)[:,-1].tolist()
	return tags[sorted_idx]

def compute_priors(feats, class_labels):
	labels = [1,2,3,5,6,7]
	prior_probs = []
	for label in labels:
		prior_probs.append((feats[class_labels == label]).shape[0])
	prior_probs = np.array(prior_probs, dtype=np.float64)
	return prior_probs / feats.shape[0]

#to supress divide by zero warnings
np.seterr(divide='ignore')
#read data
train_idx, train_feats, train_labels, test_idx, test_feats, test_labels = prep_data()

#Naive Bayes
print "Naive Bayes Classifier"
print "----------------------"
print
#compute mean and variance for each feature given a class -> parametric density estimation of gaussians
feat_mean, feat_std = estimate_mean_std(train_feats, train_labels)
#calculate prior probabilties for each class
prior_probs = compute_priors(train_feats, train_labels)
#calculate scores for each class
train_scores, test_scores = compute_scores(train_feats, test_feats, feat_mean, feat_std, prior_probs)
#predict the class with max score
train_pred = naive_bayes_predict(train_scores)
test_pred = naive_bayes_predict(test_scores)

print "Naive Bayes Training accuracy: %.4f %%" % (accuracy(train_labels, train_pred) * 100)
print "Naive Bayes Test accuracy: %.4f %%" % (accuracy(test_labels, test_pred) * 100)
print

#KNN
print "K Nearest Neighbors Classifier"
print "------------------------------"
print
k_values = [1,3,5,7]

#normalize training data
train_feats, mean, std = normalize_train(train_feats)

#normalize test data
test_feats = normalize_test(test_feats, mean, std)

#calculate l1 distance
l1_dist = compute_l1(train_feats, test_feats)

#calculate l2 distance
l2 = compute_l2(train_feats, test_feats)

for k in k_values:
	print "##############"
	print "K = ", k
	print "##############"
	print
	#predict the class
	pred_l1 = predict(l1_dist, train_labels, k)

	#calculate accuracy
	print "L1 training accuracy: %.4f %%" % (train_acc(train_feats, train_labels, 'l1', k) * 100)
	print "L1 test accuracy: %.4f %%" % (accuracy(test_labels, pred_l1) * 100)
	print
	#predict the class
	pred_l2 = predict(l2, train_labels, k)
	
	#calculate accuracy
	print "L2 training accuracy: %.4f %%" % (train_acc(train_feats, train_labels, 'l2', k) * 100)
	print "L2 test accuracy: %.4f %%" % (accuracy(test_labels, pred_l2) * 100)
	print