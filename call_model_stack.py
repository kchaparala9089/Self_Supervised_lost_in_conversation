import subprocess
import argparse
import os
import time

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--user_system_model', type=str, default="Llama3.1-8B-Instruct", help='Model to call')
    argparser.add_argument('--experiment_models', type=str, nargs='+', default=["Llama3.1-8B"], help='Models to run experiments with')
    args = argparser.parse_args()

    command = "vllm serve " + args.user_system_model + " --port 8000 --model-name model --api-key 123"
    user_system_process = subprocess.Popen(command.split())
    print(f"Started user/system model subprocess with PID {user_system_process.pid}")
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
    
    time.sleep(20)  # wait for the user/system model server to start
    # assert address is reachable
    import requests
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers={"api-key": "123"}, json={"model": "model", "messages": [{"role": "user", "content": "Hello"}]})
        if response.status_code != 200:
            raise Exception(f"User/system model server not reachable, status code: {response.status_code}, response: {response.text}")
        else:
            print("User/system model server is reachable.")
    except Exception as e:
        print(f"User/system model server not reachable: {e}")
        user_system_process.terminate()
        exit(1)

    for experiment_model in args.experiment_models:
        experiment_model_command = "vllm serve " + experiment_model + " --port 8001 --model-name model --api-key 123"
        experiment_model_process = subprocess.Popen(experiment_model_command.split())
        print(f"Started experiment model subprocess with PID {experiment_model_process.pid}")
        command = f"python run_simulations.py --system_model model:localhost:8000 --user_model model:localhost:8000 --models model:localhost:8001 --N_workers 1"
        print(f"Running experiments with command: {command}")
        experiment_process = subprocess.Popen(command.split())
        experiment_process.wait()
        print(f"Experiment with model {experiment_model} completed.")
        experiment_model_process.terminate()
    user_system_process.terminate()

