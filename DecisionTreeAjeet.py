import numpy as np

class Node:
    def __init__(self, feature_index= None, 
                 threshold= None,
                 left = None, 
                 right = None, 
                 information_gain = None,
                 value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        
        self.value = value
        
class DecisionTreeClassifier:
    def __init__(self, max_depth= 4, min_sample_split = 2):
        
        self.root = None
        
        
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def information_gain(self, parent,left_child, right_child, mode ="entropy" ):
        
        weight_l = len(left_child) / len(parent)
        print(weight_l)
        weight_r = len(right_child)  / len(parent)
        if mode =="gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(left_child) + weight_r * self.gini_index(right_child))
        else:
            gain = self.entropy(parent)  - (weight_l * self.entropy(left_child)  + weight_r * self.entropy(right_child))
        return gain
    
    
    
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.information_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
        
    def fit(self, X, y):
        df = np.concatenate((X, y), axis = 1)
        self.root = self.build_tree(df)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions 
    
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            # print(feature_val)
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right) 
    
    # def make_predictions(self, X, tree):
    #     if tree.value != None: return tree.value
        
    #     feature_val = X[tree.feature_index]
    #     if feature_val <= tree.threshold: return self.make_predictions(X, tree.left)
    #     else: return self.make_predictions(X, tree.right)
        
    def build_tree(self, df, current_depth=0):
        X, y = df[:, :-1], df[:, -1].reshape(-1, 1)
        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_sample_split and current_depth <= self.max_depth:
            best_split = self.get_best_split(df, n_samples, n_features)
            if best_split["information_gain"] > 0:
                left_subtree = self.build_tree(best_split['left_dataset'], current_depth + 1)
                right_subtree = self.build_tree(best_split['right_dataset'], current_depth + 1)
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split["information_gain"]
                )

        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
        print(leaf_value)
        return Node(value=leaf_value)
         
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        print(Y)
        return max(Y, key=Y.count)
    

    def get_best_split(self, df, n_samples,n_features):
        best_split = {}
        
        max_info_gain = -float("inf")
        
        for feature_index in range(n_features):
            feature_values = df[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_split, right_split = self.split(df, threshold, feature_index)
                if len(left_split) >0 and  len(right_split) > 0: 
                    y, left_y , right_y = df[:, -1], left_split[:, -1], right_split[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "entropy")
                    if(curr_info_gain > max_info_gain):
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] =left_split
                        best_split["right_dataset"] = right_split 
                        best_split["information_gain"]= curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split                
    
    def split(self, df, threshold, feature_index):
        left_split  = np.array([row for row in df if row[feature_index] <= threshold])
        right_split = np.array([row for row in df if row[feature_index] >= threshold])
        return left_split, right_split