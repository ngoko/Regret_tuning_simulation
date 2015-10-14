from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satruntime.csv', request_time_slot=600, conf_per_slot=100)
regret_period = 600*72
r.shuffle_data()
#v = r.run_method('lex', regret_period)
#print max(v)
v = r.run_method('rand', regret_period)
print max(v)
