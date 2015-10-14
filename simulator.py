from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satruntime.csv', request_time_slot=600, conf_per_slot=100)
regret_period = 600*72

u = 0
v = 0
w = 0
for i in range(0, 150):
  r.shuffle_data()
  u +=  max(r.run_method('lex', regret_period))
  v +=  max(r.run_method('rand', regret_period))
  w +=  max(r.run_method('randsort', regret_period))

print u/100, v/100, w/100
