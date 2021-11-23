#Importing Libraries
import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Loading data
iris=datasets.load_iris()
X,y=iris.data,iris.target

#Spliting data into train and test categories
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

#Class Naive Bayes 
class naive_bayes():
    #Separating classes for mean and standard deviation calculation
    def separate_classes(self):
        l0=[]
        l1=[]
        l2=[]
        for i in range(len(y_train.tolist())):
            if y_train[i]==0:
                l0.append(X_train[i])
            elif y_train[i]==1:
                l1.append(X_train[i])
            else:
                l2.append(X_train[i])
        return l0,l1,l2
    
    #Calculation of mean and standard deviation depending on class target
    def mean_std(self,x):
        l0,l1,l2=self.separate_classes()
        mean_x=0
        std_x=0
        if x==0:
            mean_x=np.mean(l0,axis=0)
            std_x=np.std(l0,axis=0)
        elif x==1:
            mean_x=np.mean(l1,axis=0)
            std_x=np.std(l1,axis=0)
        else:
            mean_x=np.mean(l2,axis=0)
            std_x=np.std(l2,axis=0)
        return mean_x,std_x

    #Guassian Distribution Calculation
    def guassian_distribution(self,p,x,flag):
        k=1/np.sqrt(2*math.pi)
        mean_x,std_x=self.mean_std(x)
        express= math.exp(-((p-mean_x[flag])**2 / (2 * std_x[flag]**2 )))
        p_x=k*(1/std_x[flag])*express
        return p_x

    #Calculation of conditional probabilities depending on class target
    def con_prob_x(self,p,x):
        con_prob_f1=self.guassian_distribution(p[0],x,0)
        con_prob_f2=self.guassian_distribution(p[1],x,1)
        con_prob_f3=self.guassian_distribution(p[2],x,2)
        con_prob_f4=self.guassian_distribution(p[3],x,3)
        return con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4

    #Training: calculation of probabilities and updation of prior probabilities
    def training(self):
        prior_prob0=1/3
        prior_prob1=1/3
        prior_prob2=1/3
        for i in X_train:
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,0)
            prob0=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
    
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,1)
            prob1=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
            
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,2)
            prob2=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
            
            prior_prob0=prob0
            prior_prob1=prob1
            prior_prob2=prob2
        return prior_prob0,prior_prob1,prior_prob2

    #Testing on unseen dataset with max of probability as estimates
    def test(self):
        estimates=[]
        prior_prob0,prior_prob1,prior_prob2=self.training()
        for i in X_test:
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,0)
            prob0=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,1)
            prob1=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
            con_prob_f1,con_prob_f2,con_prob_f3,con_prob_f4=self.con_prob_x(i,2)
            prob2=np.log(con_prob_f1*con_prob_f2*con_prob_f3*con_prob_f4)
            
            prior_prob0=prob0
            prior_prob1=prob1
            prior_prob2=prob2
            
            if prob0>prob1 and prob0>prob2:
                estimates.append(0)
            elif prob1>prob0 and prob1>prob2:
                estimates.append(1)
            else:
                estimates.append(2)
        return estimates

    #Accuracy calculation
    def accuracy(self,y_test):
        estimates=self.test()
        correct=0
        for i in range(len(estimates)):
            if estimates[i]==y_test[i]:
                correct=correct+1
        acc=(correct*100)/(len(estimates))
        return acc

test=naive_bayes()
print("Accuracy : ",test.accuracy(y_test))





