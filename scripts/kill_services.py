import os

# Extract values for each application
service_port = os.environ.get('PODAGENT_SERVICE_PORT')

# Execute the commands 
print(f'kill $(lsof -t -i :{service_port})')
os.system(f'kill $(lsof -t -i :{service_port})')




