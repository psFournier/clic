import tensorflow as tf
import numpy as np
from utils.util import build_logger
from env_wrappers.registration import make
from docopt import docopt
from agents import Dqn2, DqnEpsGreedy
import time

help = """

Usage: 
  main.py --env=<ENV> --agent=<AGENT> [options]

Options:
  --seed SEED              Random seed
  --eps VAL                [default: 1]
  --inv_grad YES_NO        Gradient inversion near action limits [default: 1]
  --max_steps VAL          Maximum total steps [default: 300000]
  --ep_steps VAL           Maximum episode steps [default: 200]
  --ep_tasks VAL           Maximum episode tasks [default: 1]
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/continuous/log/local/]
  --eval_freq VAL          Logging frequency [default: 2000]
  --margin VAL             Large margin loss margin [default: 1]
  --gamma VAL              Discount factor [default: 0.99]
  --batchsize VAL          Batch size [default: 64]
  --wimit VAL              Weight for imitaiton loss with imitaiton [default: 1]
  --rnd_demo VAL           Amount of stochasticity in the tutor's actions [default: 0]
  --network VAL            network type [default: 0]
  --filter VAL             network type [default: 0]
  --prop_demo VAL             network type [default: 0.02]
  --freq_demo VAL             network type [default: 5000]
  --her VAL                [default: 0]
  --lrimit VAL             network type [default: 0.001]
  --rndv VAL               [default: 0]
  --demo VAL               [default: -1]
  --demoorder VAL          [default: 0]
  --features VAL           [default: 2,3,4,5,6]
  --tutoronly VAL          [default: -1]
  --initq VAL              [default: -100]
  --layers VAL             [default: 400,300]
"""

if __name__ == '__main__':

    args = docopt(help)

    logger = build_logger(args)
    env, wrapper = make(args['--env'], args)
    env_test, wrapper_test = make(args['--env'], args)

    seed = args['--seed']
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)
        env_test.seed(seed)

    if args['--agent'] == '2':
        agent = Dqn2(args, wrapper, logger)
    elif args['--agent'] == '3':
        agent = DqnEpsGreedy(args, wrapper, logger)
    # if args['agent'] == 'ddpg':
    #     agent = DDPG(args, env, env_test, logger)
    # elif args['agent'] == 'td3':
    #     agent = TD3(args, env, env_test, logger)
    # if args['--agent'] == 'qoff':
    #     agent = Qoff(args, env, env_test, logger)
    # elif args['--agent'] == 'dqn':
    #     agent = DQN(args, env, env_test, logger)
    # elif args['--agent'] == 'dqng':
    #     agent = DQNG(args, env, env_test, logger)
    # elif args['--agent'] == 'dqngm':
    #     agent = DQNGM(args, env, env_test, logger, short_logger)
    # elif args['--agent'] == 'acdqngm':
    #     agent = ACDQNGM(args, env, env_test, logger)
    # elif args['--agent'] == 'ddpg':
    #     agent = DDPG(args, env, env_test, logger)
    # elif args['--agent'] == 'ddpgg':
    #     agent = DDPGG(args, env, env_test, logger)
    # elif args['--agent'] == 'ddpggm':
    #     agent = DDPGGM(args, env, env_test, logger)
    else:
        raise RuntimeError
    # elif args['agent'] == 'qlearning':
    #     agent = Qlearning.Qlearning(args, sess, env, env_test, logger)
    # elif args['agent'] == 'qlearning_off':
    #     agent = Qlearning_offpolicy.Qlearning_offPolicy(args, sess, env, env_test, logger)
    # elif args['agent'] == 'qlearningfd':
    #     with open(os.path.join(args['log_dir'],
    #                            'qlearning_Taxi-v1_1',
    #                            '20180628141516_051865',
    #                            'policy.pkl'), 'rb') as input:
    #         Q_tutor = pickle.load(input)
    #     agent = QlearningfD.QlearningfD(args, sess, env, env_test, Q_tutor, logger)
    env_step = 0
    episode_step = 0
    demo = [int(f) for f in args['--demo'].split(',')]
    imit_steps = int(float(args['--freq_demo']) * float(args['--prop_demo']))
    max_episode_steps = int(args['--ep_steps'])
    demo_order = bool(int(args['--demoorder']))
    state = env.reset()
    agent.reset(state)
    t0 = time.time()
    while env_step < int(args['--max_steps']):

        if env_step % int(args['--freq_demo']) == 0 and demo != [-1]:

            if demo_order:
                i = 0
                while i < len(wrapper.demo_f) and wrapper.queues[wrapper.features.index(wrapper.demo_f[i])].C_avg[-1] > 0.9:
                    i += 1
                if i == len(wrapper.demo_f):
                    task = np.random.choice(wrapper.demo_f)
                else:
                    task = wrapper.demo_f[i]
            else:
                task = np.random.choice(wrapper.demo_f)

            for _ in range(25):
                demonstration = wrapper_test.get_demo(task)
                agent.process_trajectory(demonstration)
            for _ in range(imit_steps):
                agent.imit()
        # self.env.render(mode='human')
        # self.env.unwrapped.viewer._record_video = True
        # self.env.unwrapped.viewer._video_path = os.path.join(self.logger.get_dir(), "video_%07d.mp4")
        # self.env.unwrapped.viewer._run_speed = 0.125
        a = agent.act()
        state = env.step(a)[0]
        term = agent.step(state)

        env_step += 1
        episode_step += 1

        if term or episode_step >= max_episode_steps:
            agent.end_episode()
            state = env.reset()
            agent.reset(state)
            episode_step = 0

        if env_step % int(args['--eval_freq'])== 0:
            agent.log(env_step)
            t1 = time.time()
            print(t1 - t0)
            t0 = t1
