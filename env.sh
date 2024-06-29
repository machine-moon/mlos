#!/bin/bash

# Set the home directory for virtual environments
VENV_HOME="$HOME/workspaces/mlos/venvs"

# Function to log messages with a timestamp
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

# Function to display the help message
show_help() {
    echo "Usage:"
    echo "  env <command> PROJECT_NAME [options]"
    echo
    echo "Commands:"
    echo "  start [REQ_FILE]                Create and activate a virtual environment."
    echo "                                  Optionally specify a custom requirements file."
    echo "  stop                            Deactivate the virtual environment."
    echo "  clean                           Reinstall dependencies from requirements.txt."
    echo "  list                            List all existing virtual environments."
    echo "  info                            Display information about the virtual environment."
    echo "  run  COMMAND                    Run a command in the specified virtual environment."
    echo "  save                            Export the environment to a requirements.txt file."
    echo
    echo "Examples:"
    echo "  env start my_project"
    echo "  env start my_project custom_requirements.txt"
    echo "  env stop my_project"
    echo "  env clean my_project"
    echo "  env list"
    echo "  env info my_project"
    echo "  env run my_project 'pip list'"
    echo "  env save my_project custom_requirements.txt"
    echo
}


# Function to confirm an action with the user
confirm_action() {
    read -p "$1 [y/N]: " confirm
    case "$confirm" in
        [yY][eE][sS]|[yY])
            true
            ;;
        *)
            false
            ;;
    esac
}
# Function to create a virtual environment
create_venv() {
    local env_name=$1
    local req_file=$2

    # Check if the virtual environment already exists
    if [ ! -d "$VENV_HOME/$env_name/$env_name" ]; then
        log "Creating virtual environment for $env_name..."
        python3 -m venv "$VENV_HOME/$env_name/$env_name"
        
        # Activate the virtual environment
        source "$VENV_HOME/$env_name/$env_name/bin/activate"
        
        # Check if a specific requirements file is provided
        if [ -n "$req_file" ]; then
            log "Installing dependencies from the provided requirements file..."
            pip install -r "$req_file"
        # Check if a generic requirements.txt exists when no specific requirements file is provided
        elif [ -f "$VENV_HOME/$env_name/requirements.txt" ]; then
            log "requirements.txt found. Installing dependencies..."
            pip install -r "$VENV_HOME/$env_name/requirements.txt"
        else
            log "No requirements file found. Proceeding without installing dependencies."
        fi

        # Check if requirements.txt exists after dependency installation
        if [ -f "$VENV_HOME/$env_name/requirements.txt" ]; then
            # Save a new copy with a timestamp
            log "requirements.txt exists. Saving a new copy with a timestamp..."
            pip freeze > "$VENV_HOME/$env_name/requirements_$(date +"%Y-%m-%d_%H-%M-%S").txt"
        else
            # Save the current requirements as requirements.txt
            log "requirements.txt does not exist. Saving the current requirements..."
            pip freeze > "$VENV_HOME/$env_name/requirements.txt"
        fi

        deactivate
    fi

    log "Activating virtual environment for $env_name..."
    source "$VENV_HOME/$env_name/$env_name/bin/activate"
}

# Function to deactivate the current virtual environment
deactivate_venv() {
    if [ -n "$VIRTUAL_ENV" ]; then
        log "Deactivating virtual environment..."
        deactivate
    else
        log "No virtual environment is currently active or found."
    fi
}

# Function to clean the virtual environment by reinstalling dependencies
clean_venv() {
    local env_name=$1
    local req_file="$VENV_HOME/$env_name/requirements.txt"

    # Check if the requirements file exists
    if [ -f "$req_file" ]; then
        log "Cleaning virtual environment for $env_name with $req_file..."
        source "$VENV_HOME/$env_name/$env_name/bin/activate"
        pip install --force-reinstall -r "$req_file"
        deactivate
    else
        log "No requirements.txt found for $env_name."
    fi
}

# Function to list all existing virtual environments
list_venvs() {
    log "Listing all virtual environments..."
    ls "$VENV_HOME"
}

# Function to display information about a specific virtual environment
info_venv() {
    local env_name=$1

    log "Information for virtual environment $env_name:"
    source "$VENV_HOME/$env_name/$env_name/bin/activate"
    pip list
    deactivate
}

# Function to run a command in a specified virtual environment
run_in_venv() {
    local env_name=$1
    shift
    local cmd=$@

    log "Running command in virtual environment $env_name: $cmd"
    source "$VENV_HOME/$env_name/$env_name/bin/activate"
    eval $cmd
    deactivate
}

# Function to save the current state of a virtual environment to a requirements.txt file
save_venv() {
    local env_name=$1
    # Set the default requirements filename if not provided
    if [ -z "$req_filename" ]; then
        req_filename="requirements.txt"
    else
        req_filename=$2
    fi
    local req_file_path="$VENV_HOME/$env_name/$req_filename"

    log "Saving virtual environment $env_name to $req_filename..."
    source "$VENV_HOME/$env_name/$env_name/bin/activate"
    pip freeze > "$req_file_path"
    deactivate
}

# Main starts here

# Display help if the first argument is 'help'
if [ "$1" == "help" ]; then
    show_help
    exit 0
fi

# Ensure at least two arguments are provided
if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

# Parse the command and arguments
COMMAND=$1
ENV_NAME=$2
OPTION=$3

# Execute the appropriate function based on the command
case $COMMAND in
    start)
        create_venv $ENV_NAME $OPTION
        ;;
    stop)
        deactivate_venv
        ;;
    clean)
        if confirm_action "Are you sure you want to clean the virtual environment for $ENV_NAME?"; then
            clean_venv $ENV_NAME
        fi
        ;;
    list)
        list_venvs
        ;;
    info)
        info_venv $ENV_NAME
        ;;
    run)
        run_in_venv $ENV_NAME $OPTION
        ;;
    save)
        save_venv $ENV_NAME
        ;;
    *)
        show_help
        ;;
esac
