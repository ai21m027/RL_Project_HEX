
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

def uploadGameFiles():
    scp = SCPClient(ssh.get_transport())
    scp.put('DataGenerator.py', remote_path='/home/ai21m026/hex')
    scp.put('HexGame.py', remote_path='/home/ai21m026/hex')
    scp.put('HexNet.py', remote_path='/home/ai21m026/hex')
    scp.put('MCTS.py', remote_path='/home/ai21m026/hex')
    scp.put('utils.py', remote_path='/home/ai21m026/hex')
    scp.put('start_generators.sh', remote_path='/home/ai21m026/hex')
    scp.put('x_slurm_kill_all_jobs.sh', remote_path='/home/ai21m026/hex')

def downloadNnet(remote_path, local_path):
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_path, local_path)

def ssh_tests():
    stdin, stdout, stderr = ssh.exec_command('sinfo')
    result = str(stdout.read())
    print(result.replace('\\n', '\n'))


uploadGameFiles()