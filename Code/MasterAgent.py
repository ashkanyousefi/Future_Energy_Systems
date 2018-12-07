from dqn import *

def main(dnum):
    total = 0
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    tr = True
    model = GymDQNLearner(True, dnum)
    import time
    start_time = time.time()
    model.train()
    time_cost = time.time() - start_time
    total += time_cost
    with open("./logs/dqn_group_%s_runtime.txt" % dnum, "w+") as f:
        f.write(str(total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnum', type=int, default=2)
    main(args.dnum)
    count = 1

