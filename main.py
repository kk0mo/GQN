from utils import evaluate_policy, Reward_adapter, str2bool
from datetime import datetime
from GSAC import GSAC_agent
import gymnasium as gym
import numpy as np
import os, shutil
import argparse
import torch


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=2, help='Humanoid-v4,HalfCheetah-v4, Hopper-v4, HumanoidStandup-v4')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--actor_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--critic_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--alpha_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.15, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.device = torch.device(opt.device) # from str to torch.device
#print(opt)


def main():
    EnvName = ['Humanoid-v4','HalfCheetah-v4', 'Hopper-v4', 'HumanoidStandup-v4']

    # Build Env
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.action_bound = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(' ---------------------------- Env ----------------------------')
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.action_bound}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}  '
          f'device:{opt.device}')
    #print(env.observation_space.shape)
    # Seed Everything
    env_seed = opt.seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        time_now = datetime.now().strftime("%d-%m-%Y/%H:%M")
        write_path = os.path.join('runs', EnvName[opt.EnvIdex], time_now)
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        writer = SummaryWriter(log_dir=write_path)
        
    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = GSAC_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(EnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        c_nt = 1
        while True:
            if c_nt > 1:
                break
            c_nt += 1
            score = evaluate_policy(env, agent, opt.device, turns=1)
            print('EnvName:', EnvName[opt.EnvIdex], 'score:', score)
        env.close()
        eval_env.close()
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                if total_steps < (4*opt.max_e_steps): 
                    a = env.action_space.sample() # warm up
                else:
                    with torch.no_grad():
                        state = torch.FloatTensor(s[np.newaxis, :]).to(opt.device)
                        a = agent.actor(state)[0].cpu().numpy()[0]
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                #r = Reward_adapter(r, opt.EnvIdex)
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for j in range(opt.update_every):
                        agent.train()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    agent.explore_noise *= opt.explore_noise_decay
                    ep_rew, ep_entropy = evaluate_policy(eval_env, agent, opt.device, turns=3)
                    if opt.write: 
                        writer.add_scalar('ep_r', ep_rew, global_step=total_steps)
                        writer.add_scalar('ep_entropy', ep_entropy, global_step=total_steps)
                    # Assuming total_steps is defined somewhere in your code
                    total_steps_suffix = "k" if total_steps < 1e6 else "m"
                    total_steps_value = total_steps / 1000 if total_steps < 1e6 else total_steps / 1e6
                    formatted_steps = f"{int(total_steps_value)}{total_steps_suffix}"
                    print(f'EnvName:{EnvName[opt.EnvIdex]}, Steps: {formatted_steps}, Episode Reward:{ep_rew}, Episode Entropy:{ep_entropy}')


                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(EnvName[opt.EnvIdex], int(total_steps / 1e5))
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()