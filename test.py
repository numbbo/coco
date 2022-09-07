# coding: utf-8
import cocoex
suite = cocoex.Suite("bbob-constrained", "", "")

count1 = 0
fail1 = []
count2 = 0
fail2 = []

for problem in suite:
    if not all(problem.constraint(problem.initial_solution) < 0):
        count1 += 1
        fail1.append(problem.name)
    if not (all(-5 < problem.initial_solution) and all(problem.initial_solution < 5)):
        count2 += 1
        fail2.append(problem.name)
    

if __name__ == "__main__":
    barstr = "-" * 50    

    for count, fail, name in ((count1, fail1, "FEASIBLE"), (count2, fail2, "INBOUNDS")):
        if count:
            print(barstr)
            print("FAIL TO BE FEASIBLE")
            print(barstr)
            for f in fail: print(f);
