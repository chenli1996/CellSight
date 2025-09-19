#!/bin/bash

# Define the path to the SSH config file
CONFIG_FILE="$HOME/.ssh/config"
BACKUP_FILE="$HOME/.ssh/config.bak"

# Ask for the new hostname
echo -n "Enter the new hostname for 'greene-compute': "
read new_hostname

# Check if the new hostname is provided
if [ -z "$new_hostname" ]; then
    echo "No hostname entered. Exiting without changes."
    exit 1
fi

# Backup the original config file
cp $CONFIG_FILE $BACKUP_FILE

# Update the HostName within the 'greene-compute' block
awk -v newhost="$new_hostname" '
    $1=="Host" && $2=="greene-compute"{flag=1}
    $1=="Host" && $2!="greene-compute" && flag{flag=0}
    flag && $1=="HostName"{$1="    HostName"; $2=newhost; $0=$1 " " $2}
    {print}
' $BACKUP_FILE > $CONFIG_FILE

# Output the result of the operation
if [ $? -eq 0 ]; then
    echo "HostName updated successfully to '$new_hostname' with indentation in 'greene-compute'."
else
    echo "Failed to update HostName. Please check the script and the config file."
fi
