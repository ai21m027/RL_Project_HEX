
import paramiko
from scp import SCPClient
import getpass

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient('ctld-n1.cs.technikum-wien.at', 22, input('Enter user name:\n'), getpass.getpass(prompt='Password: '))

def scp_tests():
    scp = SCPClient(ssh.get_transport())
    scp.put('main_train.py', remote_path='/home/ai21m026/hex')
    scp.put('Game.py', remote_path='/home/ai21m026/hex')
    scp.put('Arena.py', remote_path='/home/ai21m026/hex')
    scp.put('Coach.py', remote_path='/home/ai21m026/hex')
    scp.put('CoachAssistant.py', remote_path='/home/ai21m026/hex')
    scp.put('MCTS.py', remote_path='/home/ai21m026/hex')
    scp.put('NeuralNet.py', remote_path='/home/ai21m026/hex')
    scp.put('utils.py', remote_path='/home/ai21m026/hex')
    scp.put('run_job.sh', remote_path='/home/ai21m026/hex')
    scp.put('run_training.sh', remote_path='/home/ai21m026/hex')
    scp.put('clear_and_run.sh', remote_path='/home/ai21m026/hex')
    scp.put('hex', recursive=True, remote_path='/home/ai21m026/hex')
    scp.put('pytorch_classification', recursive=True, remote_path='/home/ai21m026/hex')

def ssh_tests():
    stdin, stdout, stderr = ssh.exec_command('sinfo')
    result = str(stdout.read())
    print(result.replace('\\n', '\n'))


scp_tests()
# ssh_tests()
