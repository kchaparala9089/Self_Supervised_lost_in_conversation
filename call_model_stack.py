import subprocess
import argparse
import os
import time
import openai
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--user_system_model', type=str, default="/export/fs06/aalme2/Self_Supervised_lost_in_conversation/models/Llama-3.1-Tulu-3-8B-SFT", help='Model to call')
    argparser.add_argument('--experiment_models', type=str, nargs='+', default=["/export/fs06/aalme2/Self_Supervised_lost_in_conversation/models/Llama-3.1-Tulu-3-8B-SFT", "/export/fs06/aalme2/Self_Supervised_lost_in_conversation/models/Llama-3.1-Tulu-3-8B-DPO"], help='Models to run experiments with')
    args = argparser.parse_args()

    # set CUDA_VISIBLE_DEVICES to 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    command = "vllm serve " + args.user_system_model + " --port 8002 --max-model-len 10000 --api-key 123 --served_model_name model"
    system_log = open("user_system_model_log.txt", "w")
    user_system_process = subprocess.Popen(command.split(), stdout=system_log, stderr=system_log)
    print(f"Started user/system model subprocess with PID {user_system_process.pid}")
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8002/v1"

    print("Waiting for user/system model server to start...")
    
    time.sleep(60)  # wait for the user/system model server to start

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    client = openai.OpenAI(api_key="123", base_url="http://localhost:8002/v1")
    completion = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "user", "content": "Hello!"},
        ],
    )
    print(completion.choices[0].message)

    for experiment_model in args.experiment_models:
        experiment_model_command = "vllm serve " + experiment_model + " --port 8003 --max-model-len 10000 --api-key 123 --served_model_name model" # --gpu-memory-utilization 0.4"
        experiment_log = open(f"experiment_model_{os.path.basename(experiment_model)}_log.txt", "w")
        experiment_model_process = subprocess.Popen(experiment_model_command.split(), stdout=experiment_log, stderr=experiment_log)
        print(f"Waiting for experiment model {experiment_model} server to start...")
        time.sleep(60)
        print(f"Started experiment model subprocess with PID {experiment_model_process.pid}")
        command = f"python run_simulations.py --system_model model:localhost:8002 --user_model model:localhost:8002 --models model:localhost:8003 --N_workers 1"
        print(f"Running experiments with command: {command}")
        experiment_process = subprocess.Popen(command.split())
        experiment_process.wait()
        print(f"Experiment with model {experiment_model} completed.")
        experiment_model_process.terminate()
        time.sleep(20)
    user_system_process.terminate()

