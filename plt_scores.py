import matplotlib.pyplot as plt 
scores = []
with open('scores.txt', 'r') as file:
    file.readline()
    scores = [ int(score) for score in file ]


    

k = 100
sliding_avg = []
avg = sum(scores[0:k])/k
for i in range(len(scores) - k):
    sliding_avg.append(avg)
    avg = avg + scores[i+k]/k - scores[i]/k
plt.plot(scores)
plt.plot(sliding_avg)
plt.show()


