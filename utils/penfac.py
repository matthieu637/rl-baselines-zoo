
import argparse
import configparser
from run_optuna import run_optuna

class Penfac:
    def __init__(self, policy, env, gamma=0.99, noise=0.2, momentum=1, actor_output_layer_type=2, hidden_layer_type=1,
                 alpha_a=0.0001, alpha_v=0.001, number_fitted_iteration=10, lambda_=0.8, update_each_episode=5, stoch_iter_actor=30, beta_target=0.2,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        parser = argparse.ArgumentParser()
        parser.add_argument('--env')
        clargs, _ = parser.parse_known_args()

        config = configparser.ConfigParser()
        config.read('config.ddrl.ini')
        config['agent']['gamma']=str(gamma)
        config['agent']['noise']=str(noise)
        config['agent']['momentum']=str(momentum)
        config['agent']['actor_output_layer_type']=str(actor_output_layer_type)
        config['agent']['hidden_layer_type']=str(hidden_layer_type)
        config['agent']['alpha_a']=str(alpha_a)
        config['agent']['alpha_v']=str(alpha_v)
        config['agent']['number_fitted_iteration']=str(number_fitted_iteration)
        config['agent']['lambda']=str(lambda_)
        config['agent']['update_each_episode']=str(update_each_episode)
        config['agent']['stoch_iter_actor']=str(stoch_iter_actor)
        config['agent']['beta_target']=str(beta_target)
        config['simulation']['env_name']=str(clargs.env)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        class Dummy:
            def close(self):
                pass
        self.env = Dummy()


    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="penfac", reset_num_timesteps=True):

        config = configparser.ConfigParser()
        config.read('config.ini')
        config['simulation']['total_max_steps'] = str(total_timesteps)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        run_optuna(callback)
        return self

    def get_env(self):
        return None

