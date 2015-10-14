from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satruntime.csv', request_time_slot=600, conf_per_slot=100)
regret_period = 600*72

n = 100
u = 0
v = 0
w = 0

#for i in range(0, n):
#  r.shuffle_data()
#  u +=  max(r.run_method('lex', regret_period))
#  v +=  max(r.run_method('rand', regret_period))
#  w +=  max(r.run_method('randsort', regret_period))

#print u/n, v/n, w/n

r.shuffle_data()
x = r.run_method('randw', regret_period)
print max(x)
