#!/bin/bash

# Function to rename directories to lowercase
rename_dirs() {
    # Find all directories, sort by depth, and process in reverse order
    find "$1" -depth -type d | while read -r dir; do
        lower_dir=$(echo "$dir" | tr '[:upper:]' '[:lower:]')
        if [ "$dir" != "$lower_dir" ]; then
            mv "$dir" "$lower_dir"
        fi
    done
}

# Start renaming from the current directory
rename_dirs .

echo "All directories have been converted to lowercase."

