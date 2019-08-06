# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:58:42 2019

@author: Mallory
with code from Chris' office hours content 
"""

"""
1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?
"""

#import libraries
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict  
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from itertools import product

#Read the two first two lines of the file.
with open('C:/Users/Mallory/Documents/SMU/Machine Learning/HW 2/claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())
    
#Colunn names that will be used in the below function, np.genfromtxt
names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]


#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt('C:/Users/Mallory/Documents/SMU/Machine Learning/HW 2/claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

''' View the Imported Claims Data '''

# print dtypes and field names
print(CLAIMS.dtype)
# Notice the shape differs since we're using structured arrays.
print(CLAIMS.shape)

# subsetting for specific values
# subset for a specific row.
print(CLAIMS[0])
# subset for a specific value
print(CLAIMS[0][1])
# view the names
print(CLAIMS.dtype.names)
# subset into a column
print(CLAIMS['MemberID'])
# subset into a column and a row value
print(CLAIMS[0]['MemberID'])

''' Find the J-codes using two methods: find() and startswith() '''

# a text string for the letter we are searching for
J = 'J'
J = J.encode()

''' Commenting this out becuase it finds too many Jcodes! Find is 
the better method '''
# Try startswith() on CLAIMS
#JcodeIndexes = np.flatnonzero(np.core.defchararray.startswith(CLAIMS['ProcedureCode'], J, start=1, end=2)!=-1)
np.set_printoptions(threshold=500, suppress=True)
# Using those indexes, subset CLAIMS to only Jcodes
#Jcodes = CLAIMS[JcodeIndexes]
#print(Jcodes)
#print(Jcodes.size)


# Try find() on CLAIMS
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], J, start=1, end=2)!=-1)
#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]
print(Jcodes)
print(Jcodes.size)

# ensure the names are correct
print(Jcodes.dtype.names)
# verify how the data is stored
print(type(Jcodes))
print(Jcodes.shape)

''' Question 1 A '''
print("The number of claim lines that have Jcodes is", Jcodes.size,".")

''' Question 1 B '''
I = 'I'
I = I.encode()
InNetworkIndexes = np.flatnonzero(np.core.defchararray.find(Jcodes['InOutOfNetwork'], I, start=1, end=2)!=-1)
InNetworkJcodes = Jcodes[InNetworkIndexes]
print(InNetworkJcodes.size)
print(InNetworkJcodes.dtype.names)
InNetwork_Amount=np.sum(InNetworkJcodes['ProviderPaymentAmount'])
print("The amount paid for J-codes to providers for 'in network' claims is ",round(InNetwork_Amount,2),"dollars.")

''' Question 1 C 
     C. What are the top five J-codes based on the payment to providers?
'''

''' Sorting the Jcodes '''
# Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')

# Reverse the sorted Jcodes (A.K.A. in descending order)
Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]

# What are the top five J-codes based on the payment to providers?

# We still need to group the data
print(Sorted_Jcodes[:10])

# You can subset it...
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes2 = Sorted_Jcodes['ProcedureCode']

# recall their data types
Jcodes2.dtype
ProviderPayments.dtype

# get the first three values for Jcodes
Jcodes2[:3]

# get the first three values for ProviderPayments
ProviderPayments[:3]

# Join arrays together
arrays = [Jcodes2, ProviderPayments]

# https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

# What does the result look like?
print(Jcodes_with_ProviderPayments[:3])

Jcodes_with_ProviderPayments.shape

# GroupBy JCodes using a dictionary
JCode_dict = {}

# Aggregate with Jcodes - code  modified from a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
    if aJCode[0] in JCode_dict.keys():
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]

# sum the JCodes
np.sum([v1 for k1,v1 in JCode_dict.items()])

# create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
# and then sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True))
    
# view the results        
print(JCodes_PaymentsAgg_descending)

# find the top 5 Jcodes!
list(JCodes_PaymentsAgg_descending.items())[:5]
top5=list(JCodes_PaymentsAgg_descending)[:5]
print(top5)
top5_sum=round(np.sum(list(JCodes_PaymentsAgg_descending.values())[:5]),2)
print(top5_sum)

print("The top five J-codes based on the payment to providers are",top5,".")
print("These top 5 J-codes pay" ,top5_sum, "dollars to the providers.")



'''
2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.
'''

''' Question 2 A '''

# view the shape
print(Jcodes.shape)
Jcodes[:3]
print(Jcodes.dtype.names)

# create indeces for provider payment values and subset the Jcodes
Zero_Payment=Jcodes[Jcodes['ProviderPaymentAmount'] <= 0]
print(Zero_Payment.shape)
Positive_Payment=Jcodes[Jcodes['ProviderPaymentAmount'] > 0]
print(Positive_Payment.shape)

# get the unique counts for providers with payment>0
unique, counts = np.unique(Positive_Payment['ProviderID'], return_counts=True)
Positive_Counts=np.asarray((unique, counts)).T
print(Positive_Counts)
Positive_Counts.shape
print(Positive_Counts.dtype.names)

# convert positive counts to matrix
Positive_Counts_final=np.asmatrix(Positive_Counts)
Positive_Counts_final.shape
print(Positive_Counts_final)

# get the unique counts for providers with payment=0
unique2, counts2 = np.unique(Zero_Payment['ProviderID'], return_counts=True)
Zero_Counts=np.asarray((unique2, counts2)).T
print(Zero_Counts)
Zero_Counts.shape

# convert the numpy array to matrix
Zero_Counts2=np.asmatrix(Zero_Counts)
type(Zero_Counts2)
Zero_Counts2.shape
print(Zero_Counts2)

# drop the providers that have no paid J Code claims (no matching Provider ID in the positive_counts)
drop_index = [7, 14]
Zero_Counts_final = np.delete(Zero_Counts2, drop_index, axis=0)
Zero_Counts_final.shape
print(Zero_Counts_final)

# convert back to array
Zero_Counts_final = np.squeeze(np.asarray(Zero_Counts_final))
type(Zero_Counts_final)
Positive_Counts_final = np.squeeze(np.asarray(Positive_Counts_final))
type(Positive_Counts_final)


# now the matrices are the same size, so they can be merged
both=np.concatenate((Positive_Counts_final, Zero_Counts_final), axis=1)
both.shape
# delete third column of the array (the extra ID's)
both = np.delete(both, 2, 1)  
both.shape
both
both.dtype

# grab the columns that you want out of both so you can manually write them into your own lists
    # did this method because could not extract out the data from the array into a list using numpy
ID = both[:,0]
Positive = both[:,1]
Zero = both[:,2]
Zero.dtype
#Zero.astype(int) this doesn't work to change the dtype :/
Zero.shape
Positive.shape
Positive
Zero
ID

# manually create the arrays for plotting from the results above
Zero_manual=[8710, 9799, 13947, 539, 6703, 67, 2545, 322, 601, 49, 1170, 449, 46]

Positive_manual=[74,1786,895,8,1228,4,302,415,561,5,740,43,7]

ID_manual=["FA0001387001","FA0001387002","FA0001389001","FA0001389003",
           "FA0001411001","FA0001411003","FA0001774001","FA0004551001",
           "FA1000014001","FA1000014002","FA1000015001","FA1000015002","FA1000016001"]

fig=plt.figure()
ax1 = fig.add_subplot(211)
#ax1 = fig.add_subplot(figsize=(200, 100))
ax1.scatter(ID_manual, Positive_manual, s=10, c='b', marker="s", label='Paid JCODE Claims')
ax1.scatter(ID_manual,Zero_manual, s=10, c='r', marker="o", label='Unpaid JCODE Claims')
plt.xticks(rotation=90)

plt.legend(loc='upper right');
fig.suptitle('Paid and Unpaid Jcode Claims by Provider ID', fontsize=12)
plt.xlabel('Provider ID', fontsize=10)
plt.ylabel('Claims', fontsize=10)

plt.show()

''' Question 2 B ''' 
# What insights can you suggest from the graph?

print('Question 2B: From the scatterplot, it looks like most providers have more unpaid claims than paid claims. In the case of four providers,'
      'the difference between the paid and unpaid claims is in the order of thousands. This may be because certain Jcodes' 
      ' may not require any payments from providers. These claims may be small claims, that are either waiting to be paid' 
      ' or that will not be paid, but are still listed as claims. For the other nine providers, the delta between paid and unpaid claims is much smaller.')


''' Question 2 C ''' 
# Based on the graph, is the behavior of any of the providers concerning? Explain.
print('Question 2 C: The fact that four providers have significantly more unpaid claims than paid claims is '
      'a little concerning. Unfortunately, I do not have enough knowledge about the individual claim numbers to '
      'know whether this is normal, good, or bad. People may be filing claims even though they know insurance wont '
      'pay them, whether because the claims are too small or if they are outside the insurance coverage. The insurance '
      'companies hopefully arent sitting on thousands of unpaid claims.')




'''
3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

     D. What data attributes are predominately influencing the rate of non-payment?
'''

''' Question 3 A '''
# What percentage of J-code claim lines were unpaid?

# count the total number of Jcode claims
total=np.add(Zero_Payment.shape,Positive_Payment.shape)
total

# count the number of unpaid Jcode claims
Zero_total=Zero_Payment.shape
Zero_total

# find the percentage of unpaid claims
perc_unpaid=(Zero_total/total)*100

# extract out the number, since it is stored in an array
perc_unpaid=perc_unpaid[0]
print('The percentage of Jcode claims that are unpaid is', round(perc_unpaid,2), 'percent.')

''' Question 3 B '''
# Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

# work with the Sorted_Jcodes created earlier
print(Sorted_Jcodes.dtype.names)

# unpaid row indexes  
unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)
# paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)

# index the jcodes
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]
Paid_Jcodes = Sorted_Jcodes[paid_mask]

# these are still structured numpy arrays
print(Unpaid_Jcodes.dtype.names)
print(Unpaid_Jcodes[0])

print(Paid_Jcodes.dtype.names)
print(Paid_Jcodes[0])

# view the dtype descriptions 
print(Paid_Jcodes.dtype.descr)
print(Unpaid_Jcodes.dtype.descr)

# create labels to be used for the classifier!
# create a new column and data type for both structured arrays
new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

print(new_dtype1)
print(new_dtype2)

# create a new structured array with labels

# first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype2)

# check the shape
Unpaid_Jcodes_w_L.shape
Paid_Jcodes_w_L.shape

# view the data
 # it just has the labels
print(Unpaid_Jcodes_w_L)
print(Paid_Jcodes_w_L)

# copy the columns/data over for Unpaid
Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

# assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

# view the data
print(Unpaid_Jcodes_w_L)

# Do the same for the Paid data

# copy the data over
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
 # since these are the paid claims, the label is 0 (not unpaid!) 
Paid_Jcodes_w_L['IsUnpaid'] = 0

#Look at the data..
print(Paid_Jcodes_w_L)

# combine the rows of paid and unpaid claims together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)

# verify the shape
Jcodes_w_L.shape

# look at the transition between the rows around row 44961
    # this is where the unpaid claims end and the piad claims begin
    # the rows must be shuffled before using classifers in sklearn
print(Jcodes_w_L[44955:44968])

# view the names
Jcodes_w_L.dtype.names

# apply the random shuffle to shuffle the data
np.random.shuffle(Jcodes_w_L)

# view that is 1 to 0 label transition is no longer there
print(Jcodes_w_L[44957:44965])

# check that the columns are still in the correct order
Jcodes_w_L

# prep the data for use with sklearn
label =  'IsUnpaid'
# features after Removing V1 and Diagnosis Code
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DenialReasonCode','PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType']

numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']

# convert features to list, then to np.array 
    # this step is important for sklearn to use the data from the structured NumPy array

# separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())

#  check your sklearn version. the new version does not require the label encorder before the OHE
import sklearn
print(sklearn.__version__)

# Now use the OneHotEncoder to encode the categorical features
     # subet if there is a memory error
ohe = OneHotEncoder(sparse=False) # makes it easier to read
Mcat = ohe.fit_transform(Mcat)

# if you want to go back to the original mappings.
ohe.inverse_transform(Mcat)
ohe_features = ohe.get_feature_names(cat_features).tolist()

# shape of the matrix categorical columns that were OneHotEncoded   
Mcat.shape
Mnum.shape

# concatenate the columns
M = np.concatenate((Mcat, Mnum), axis=1)
# get the labels from the other data
L = Jcodes_w_L[label].astype(int)

# check the final shapes
M.shape
L.shape

# Death to GridSearch!!!
    # code from previous HW 1
    
# folds used for cross validation
n_folds = 5

# view the cross validation object
kf = KFold(n_splits=n_folds)
print(kf)

# pack the arrays together into "data"
data = (M,L,n_folds)

# view the data
print(data)

# functions for the grid search
# "run" function runs all the classifiers on the data
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # Establish the cross validation 
  ret = {} # classic explication of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): # We're interating through train and test indexes by using kf.split
                                                                      # from M and L.
                                                                      # We're simply splitting rows into train and test rows
                                                                      # for our five folds.    
    clf = a_clf(**clf_hyper) # unpack paramters into clf if they exist   # this gives all keyword arguments except 
                                                                            # for those corresponding to a formal parameter
                                                                            # in a dictionary.
                                                                                   
    clf.fit(M[train_index], L[train_index])   # First param, M when subset by "train_index", 
                                                 # includes training X's. 
                                                 # Second param, L when subset by "train_index",
                                                 # includes training Y.                               
    pred = clf.predict(M[test_index])         # Using M -our X's- subset by the test_indexes, 
                                                 # predict the Y's for the test rows.    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret


# A dictionary where scores are kept by model and hyper parameter combinations
# this is necessary because otherwise the results are overwritten when "run" executes
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) # Since we have a number of k-folds for each classifier...
                           # We want to prevent unique k1 values due to different "key" values
                           # when we actually have the same classifer and hyper parameter settings.
                           # So, we convert to a string                       
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')       
        # Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) # append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] # create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
# function that runs through the hyperparameter combinations and matches it with the clf name
def myHyperParamSearch(clfsList,clfDict):  
    for clf in clfsList:    
    # check if values in clfsList are in clfDict ... 
        clfString = str(clf)     
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters   
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperParams = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperParams) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results

# Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

# the classifier combinations you want to run stored in a list
clfsList = [RandomForestClassifier]
#clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier] 

# dictionary of the classifiers with the different hyperparameters that you want to run
clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3]}}
#clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3], "max_depth": [3,5,8]},                                      
#'LogisticRegression': {"tol": [0.001,0.01,0.1], "penalty": ['l1','l2'], "solver": ['liblinear','saga']},
#'KNeighborsClassifier': {"n_neighbors": np.arange(3, 8), "weights": ['uniform', 'distance'], "algorithm": ['ball_tree', 'kd_tree', 'brute']}}

# Run myHyperSetSearch and print the results
myHyperParamSearch(clfsList,clfDict)    
print(clfsAccuracyDict)    

print('Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach: I chose the Random Forest classifier model.' 
      ' I chose a classification model because the target is a binary'
      ' response, IsUnpaid (1 for unpaid claims and 0 for paid claims). I chose a RF because RF models are one of my favorite'
      ' machine learning methods and usually perform well. I used the grid search code from HW 1 to iterate through several'
      ' combinations of the min_samples_split and n_jobs hyperparameters. From this we can identify the most optimal hyperparameter'
      ' combination by measuring the model accuracy. Accuracy was chosen as the metric because it is a well known and simple metric'
      ' that measures the percentage of correct predictions. Accuracy is often not the best metric because it is sensitive to '
      ' unbalanced data sets. It is often better to use F1 score. However, we are using just accuracy for now to keep the analysis simple.')

''' Question 3 C ''' 
# How accurate is your model at predicting unpaid claims?

# Plot the Random Forest Classifier accuracy results
RF={ k:v for k,v in clfsAccuracyDict.items() if 'RandomForestClassifier' in k }
labels, data = [*zip(*RF.items())]  # 'transpose' items to parallel key, value lists

plt.boxplot(data)
plt.xticks([])
plt.title("Box Plots for Random Forest Hyperparamters",fontsize=20)
plt.xlabel('Hyperparameter Combinations',fontsize=10)
plt.ylabel('Accuracy',fontsize=10)
plt.savefig('RandomForest_Results.png',bbox_inches = 'tight')
plt.show()

# average accuracy
avg_acc=np.mean(data)

# view the optimal hyperparameter combination based off the box plot results
print(list(RF.keys())[4])

print('The Random Forest model is consistently 99% accurate as shown by the box plots. The average accuracy from the grid search'
      ' models is' ,np.round(avg_acc,4)*100, 'percent. The model does extremely well'
      ' at predicting unpaid claims, perhaps too well! It may be worth looking at another model validation metric'
      ' such as F1 Score in a subsequent analysis. From looking at the box plots, the best performing RF model'
      ' has min_samples_split=3 and n_jobs=2. However, all the models have extremely high accuracy so any of the'
      ' hyperparameters could be used with success.')

''' Question 3 D ''' 
# What data attributes are predominately influencing the rate of non-payment?

# here I used a logistic regression model to perform recursive feature elimination (RFE) to select the five most important features
 # chose LR just to use a different model
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 5)
fit = rfe.fit(M, L)
print('Num Features:' ,fit.n_features_, '.')
print('Selected Features:', fit.support_,'.')
print('Feature Ranking:',fit.ranking_,'.')

print('It looks like the five most influential features are those with the label True in the' 
      ' above Selected Features or with the Ranking 1 in the Feature Ranking list. We can use these index values' 
      ' to go back and match the feature names. We could also play with the model and use different numbers of'
      ' important features to search for! We could also visualize the RF model trees in additonal analysis to see'
      ' where the RF model is spliting.')
