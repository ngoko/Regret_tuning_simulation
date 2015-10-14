from regret_tuning import RegretTuning

r = RegretTuning(datafile='./satruntime.csv', request_time_slot=700, conf_per_slot=1)
regret_period = 700*10
v = r.run_method('lex', regret_period)
print v
v = r.run_method('rand', regret_period)
print v
