import gradio as gr
from gradio_client import Client
import threading
import time
import train_gui

def run_server():
    train_gui.demo.launch(server_port=7869, prevent_thread_lock=True)

threading.Thread(target=run_server).start()
time.sleep(3) # Wait for server

try:
    print("Connecting to local Gradio...")
    client = Client("http://127.0.0.1:7869")
    print("Submitting request...")
    job = client.submit("Transformer", 64, 5, 0.0001, 5, fn_index=0)
    time.sleep(10)
    print("Job status:", job.status())
    print("Job output:", job.outputs())
except Exception as e:
    print("Client Error:", e)
