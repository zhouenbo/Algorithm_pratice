
def computeAccPrecisionRecall(labels, scores, threshold):
    predictions = [1 if score > threshold else 0 for score in scores]
    # print(predictions)
    # for i in range(len(scores)):
    #     if scores[i]>threshold:
    #         predictions[i]=1
    #     else:
    #         predictions[i]=0
    # print(predictions)


    confMatrix = [[0]*2 for _ in range(2)]
    for i in range(len(labels)):
        if labels[i]==predictions[i]:
            if labels[i]==1:
                confMatrix[0][0]+=1
            else:
                confMatrix[1][1]+=1
        else:
            if labels[i] == 1:
                confMatrix[0][1] += 1
            else:
                confMatrix[1][0] += 1

    print(confMatrix)
    accuracy = (confMatrix[0][0]+confMatrix[1][1])/(sum(confMatrix[0])+sum(confMatrix[1]))
    precision = confMatrix[0][0]/(confMatrix[0][0]+confMatrix[1][0])
    recall = confMatrix[0][0]/(confMatrix[0][0]+confMatrix[0][1])
    return accuracy, precision, recall

if __name__=="__main__":
    a = [1,1,1,0,1,0,0,0]
    b = [0.2,0.6,0.3,0.49,0.2,0.6,0.3,0.49]
    c = 0.5

    #a = "12 35 60 200 10204 532566"
    ans = computeAccPrecisionRecall(a,b,c)
    print(ans)


    print("--------------------------------------------------")
    from sklearn.metrics import confusion_matrix
    # >> > y_true = [2, 0, 2, 2, 0, 1]
    # >> > y_pred = [0, 0, 2, 2, 0, 2]
    predictions = [1 if score > c else 0 for score in b]
    print(predictions)
    confMatrix = confusion_matrix(a, predictions)
    print(confMatrix)

    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    # >> > y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    # >> > y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    print(precision_recall_fscore_support(a, predictions))