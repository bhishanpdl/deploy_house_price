# Connecting to AWS
- when creating new instance give full security access (I already havd `FullAccess` group, use it)
- SSH and SCP use port 22 which I have allowed from any computer with my pem file.
- Click on instance > Connect > `ssh -i "~/.aws/2020nov18.pem" ec2-user@ec2-18-220-58-129.us-east-2.compute.amazonaws.com`

Transfer files
```bash
# terminal tab 1 (local computer)
pwd # make sure you are in correct directory
ls ~/.aws # make sure you have pem files there are they have permission chmod 400
scp -i "~/.aws/2020nov18.pem" -r app.py ec2-user@ec2-18-220-58-129.us-east-2.compute.amazonaws.com


# terminal tab 2 (remote aws host)
ssh -i "~/.aws/2020nov18.pem" ec2-user@ec2-18-220-58-129.us-east-2.compute.amazonaws.com
ls # make sure you have files copied from scp

sudo apt-get update && sudo apt-get install python3-pip # for ubuntu
#sudo yum update && sudo yum  install python3-pip # For amazon linux

pip3 install -r requirements.txt

which streamlit
streamlit run app.py

```
